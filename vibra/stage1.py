"""Stage 1: relevance matching with ranking consistency, anchored by the Procrustes map (Section 3.2)."""
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config as C
from .data import positives
from .linalg import ridge_fit, joint_procrustes


class Head(nn.Module):
    def __init__(self, W, b):
        super().__init__()
        self.lin = nn.Linear(W.shape[0], W.shape[1])
        with torch.no_grad():
            self.lin.weight.copy_(W.T)
            self.lin.bias.copy_(b)

    def forward(self, z):
        return F.normalize(self.lin(z), dim=-1)


class Branch:
    def __init__(self, z_new, old_q, G_old, V_new, pos_idx, q_ids, g_ids):
        self.z_new, self.old_q, self.G_old, self.V_new = z_new, old_q, G_old, V_new
        self.pos_idx, self.q_ids, self.g_ids = pos_idx, q_ids, g_ids
        self.cand = self.posmask = None

    def set_candidates(self, cand):
        self.cand = cand
        self.posmask = self.g_ids[cand] == self.q_ids[:, None]


def make_branches(f):
    codes, _ = pd.factorize(np.concatenate([f["image_ids"], f["cap_image_ids"]]))
    img_code = torch.as_tensor(codes[:len(f["image_ids"])], device="cuda")
    cap_code = torch.as_tensor(codes[len(f["image_ids"]):], device="cuda")
    cap2img, img2caps = positives(f["image_ids"], f["cap_image_ids"])
    return {"img": Branch(f["new_img"], f["old_img"], f["old_txt"], f["new_txt"], torch.as_tensor(img2caps[:, 0], device="cuda"), img_code, cap_code),
            "txt": Branch(f["new_txt"], f["old_txt"], f["old_img"], f["new_img"], torch.as_tensor(cap2img, device="cuda"), cap_code, img_code)}


@torch.no_grad()
def _topk(Q, G, k, chunk=8192):
    return torch.cat([torch.topk(Q[s:s + chunk] @ G.T, k, dim=1).indices for s in range(0, len(Q), chunk)])


@torch.no_grad()
def build_candidates(pos_idx, init_q, teacher_q, teacher_G, old_q, old_G, K, n_top=64, seed=0):
    """Per query: the positive, the top items of the initial map, of the new model and of the old model, then random fill; first K unique."""
    N, nG = len(pos_idx), old_G.shape[0]
    g = torch.Generator(device="cuda").manual_seed(seed)
    cat = torch.cat([pos_idx[:, None], _topk(init_q, old_G, n_top), _topk(teacher_q, teacher_G, n_top), _topk(old_q, old_G, n_top),
                     torch.randint(0, nG, (N, 2 * K + 256), device="cuda", generator=g)], 1)
    L = cat.shape[1]
    out = torch.empty(N, K, dtype=torch.long, device="cuda")
    for s in range(0, N, 8192):
        c = cat[s:s + 8192]
        sv, si = torch.sort(c, dim=1, stable=True)
        dup = torch.zeros_like(sv, dtype=torch.bool)
        dup[:, 1:] = sv[:, 1:] == sv[:, :-1]
        prio = torch.full_like(c, L + 1)
        prio.scatter_(1, si, torch.where(dup, torch.full_like(si, L + 1), si))
        keep = torch.topk(-prio.float(), K, dim=1).indices.sort(1).values
        out[s:s + 8192] = torch.gather(c, 1, keep)
    return out


def ce_multi_pos(logits, posmask):
    return -(torch.logsumexp(logits.masked_fill(~posmask, -1e9), 1) - torch.logsumexp(logits, 1))


def loss_rc(head, br, idx, tau, kd_weight):
    """Contrastive loss on the label plus the ranking consistency term toward the new model, over the candidate pool."""
    u = head(br.z_new[idx])
    cand = br.cand[idx]
    logits = torch.einsum("bd,bkd->bk", u, br.G_old[cand]) / tau
    loss = ce_multi_pos(logits, br.posmask[idx]).mean()
    with torch.no_grad():
        pT = F.softmax(torch.einsum("bd,bkd->bk", br.z_new[idx], br.V_new[cand]) / tau, 1)
    kl = (pT * (torch.log(pT + 1e-12) - F.log_softmax(logits, 1))).sum(1).mean()
    return loss + kd_weight * kl


@torch.no_grad()
def dev_metric(qi, qt, f):
    """Mean of image to text and text to image R@1 against the old gallery, ties broken by gallery index."""
    cap2img, img2caps = positives(f["image_ids"], f["cap_image_ids"])
    out = []
    for Q, G, pos in ((qi, f["old_txt"], img2caps), (qt, f["old_img"], cap2img[:, None])):
        pos = torch.as_tensor(pos, device="cuda")
        nG, ar, hit = G.shape[0], torch.arange(G.shape[0], device="cuda"), []
        for s in range(0, len(Q), 4096):
            S, p = Q[s:s + 4096] @ G.T, pos[s:s + 4096]
            v = p >= 0
            ps = torch.gather(S, 1, p.clamp(min=0)).masked_fill(~v, -float("inf"))
            best = ps.max(1).values
            bidx = torch.where(ps == best[:, None], p.clamp(min=0).masked_fill(~v, nG), torch.full_like(p, nG)).min(1).values
            r = (S > best[:, None]).sum(1) + ((S == best[:, None]) & (ar[None] < bidx[:, None])).sum(1)
            hit.append(r < 1)
        out.append(float(torch.cat(hit).float().mean() * 100))
    return dict(i2t_R1=out[0], t2i_R1=out[1], M=(out[0] + out[1]) / 2)


def train_heads(br, inits, tau, dev, seed=0, kd_weight=C.KD_WEIGHT, lr=1e-3, wd=1e-4, bs=512, epochs=C.STAGE1_EPOCHS):
    torch.manual_seed(seed)
    heads = {b: Head(*inits[b]).cuda() for b in br}
    opt = torch.optim.AdamW([p for h in heads.values() for p in h.parameters()], lr=lr, weight_decay=wd)
    N, g, best, hist = len(br["img"].z_new), torch.Generator().manual_seed(seed), None, []
    for ep in range(1, max(epochs) + 1):
        perm = torch.randperm(N, generator=g).cuda()
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            loss = sum(loss_rc(heads[b], br[b], idx, tau, kd_weight) for b in br)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep in epochs:
            with torch.no_grad():
                m = dict(dev_metric(heads["img"](dev["new_img"]), heads["txt"](dev["new_txt"]), dev), epoch=ep)
            hist.append(m)
            if best is None or m["M"] > best[0]["M"]:
                best = (m, {b: {k: v.detach().clone() for k, v in h.state_dict().items()} for b, h in heads.items()})
    return best[0], best[1], hist


def fit(tr, dev, out, seed=0, K=C.K, taus=C.TAUS):
    """Ridge initialization and temperature chosen on held out pairs; saves the selected heads to out/stage1.pt."""
    os.makedirs(out, exist_ok=True)
    t0, best_r = time.time(), None
    for lam in C.RIDGE_LAMBDAS:
        Wi, bi = ridge_fit(tr["new_img"], tr["old_img"], lam)
        Wt, bt = ridge_fit(tr["new_txt"], tr["old_txt"], lam)
        m = dev_metric(F.normalize(dev["new_img"] @ Wi + bi, dim=-1), F.normalize(dev["new_txt"] @ Wt + bt, dim=-1), dev)
        if best_r is None or m["M"] > best_r[0]["M"]:
            best_r = (m, lam, {"img": (Wi, bi), "txt": (Wt, bt)})
    inits = best_r[2]
    br = make_branches(tr)
    for b, x in br.items():
        W, bb = inits[b]
        x.set_candidates(build_candidates(x.pos_idx, F.normalize(x.z_new @ W + bb, dim=-1), x.z_new, x.V_new, x.old_q, x.G_old, K, seed=seed))
    runs = {}
    for tau in taus:
        runs[tau] = train_heads(br, inits, tau, dev, seed=seed)
        print(f"tau={tau} dev M={runs[tau][0]['M']:.2f} epoch={runs[tau][0]['epoch']}", flush=True)
    tau = max(runs, key=lambda t: runs[t][0]["M"])
    torch.save(runs[tau][1], os.path.join(out, "stage1.pt"))
    sel = dict(ridge_lambda=best_r[1], tau=tau, dev=runs[tau][0], all={str(t): r[0] for t, r in runs.items()}, sec=time.time() - t0)
    json.dump(sel, open(os.path.join(out, "stage1.json"), "w"), indent=1)
    return sel


def load_heads(path):
    sd = torch.load(path, map_location="cuda")
    heads = {}
    for b in ("img", "txt"):
        h = Head(sd[b]["lin.weight"].T, sd[b]["lin.bias"]).cuda()
        h.load_state_dict(sd[b])
        heads[b] = h.eval()
    return heads


class StageOne:
    """q0(z) = unit(beta f(z) + (1 - beta) unit(P^T z)) for image (f = image head) and text (f = text head) queries."""

    def __init__(self, tr, heads_path, beta=C.BETA):
        self.heads = load_heads(heads_path)
        self.P = joint_procrustes(tr)
        self.beta = beta
        for h in self.heads.values():
            for p in h.parameters():
                p.requires_grad_(False)

    def query(self, z, modality="img", head=None):
        f = head or self.heads[modality]
        return F.normalize(self.beta * f(z) + (1 - self.beta) * F.normalize(z @ self.P, dim=-1), dim=-1)
