"""Text queries: barycenter of the stage 1 text query and a contrastive text map M (Section 3.4)."""
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import positives
from .evaluate import ranks

ALPHAS = (0.0, 0.25, 0.5, 0.75)
nz = lambda x: F.normalize(x, dim=-1)


class MLP(nn.Module):
    def __init__(self, di, do):
        super().__init__()
        h = 4 * do
        self.net = nn.Sequential(nn.Linear(di, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, do))

    def forward(self, x):
        return self.net(x)


def make_map(kind, di, do):
    return nn.Linear(di, do) if kind == "lin" else MLP(di, do)


def train_map(Wn, Wo, kind, steps=2000, bs=8192, chunk=1 << 20, seed=0):
    """Symmetric InfoNCE between M(new caption embedding) and the old embedding of the same caption, learnable temperature."""
    torch.manual_seed(seed)
    phi = make_map(kind, Wn.shape[1], Wo.shape[1]).cuda()
    ls = nn.Parameter(torch.tensor(np.log(1 / 0.07), device="cuda"))
    opt = torch.optim.AdamW(list(phi.parameters()) + [ls], lr=1e-3 if kind == "lin" else 1e-4, weight_decay=0.01)
    rng, n, it = np.random.default_rng(seed), len(Wn), 0
    while it < steps:
        for c in rng.permutation((n + chunk - 1) // chunk):
            a, b = c * chunk, min((c + 1) * chunk, n)
            wn = torch.from_numpy(np.array(Wn[a:b])).cuda()
            wo = torch.from_numpy(np.array(Wo[a:b])).cuda()
            p = torch.randperm(len(wn), device="cuda")
            for s in range(0, len(wn) - bs + 1, bs):
                i = p[s:s + bs]
                lg = ls.exp().clamp(max=100) * nz(phi(wn[i].float())) @ wo[i].float().T
                y = torch.arange(len(i), device="cuda")
                loss = (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2
                opt.zero_grad()
                loss.backward()
                opt.step()
                it += 1
                if it >= steps:
                    break
            if it >= steps:
                break
    return phi.eval()


@torch.no_grad()
def t2i_r1(Q, f):
    cap2img, _ = positives(f["image_ids"], f["cap_image_ids"])
    return float((ranks(Q, f["old_img"], torch.as_tensor(cap2img, device="cuda")[:, None]) < 1).mean() * 100)


class TextQuery:
    def __init__(self, stage1, phi, alpha):
        self.stage1, self.phi, self.alpha = stage1, phi, alpha

    @torch.no_grad()
    def __call__(self, t):
        return nz(self.alpha * self.stage1.query(t, "txt") + (1 - self.alpha) * nz(self.phi(t)))


def fit(stage1, Wn, Wo, devh, out, alpha=None, steps=2000):
    """Trains a linear and an MLP map; the map and, unless given, alpha are chosen on COCO DEV."""
    os.makedirs(out, exist_ok=True)
    maps = {kind: train_map(Wn, Wo, kind, steps) for kind in ("lin", "mlp")}
    res = {}
    with torch.no_grad():
        for kind, phi in maps.items():
            for a in (ALPHAS if alpha is None else (alpha,)):
                res[f"{kind}|{a}"] = t2i_r1(TextQuery(stage1, phi, a)(devh["new_txt"]), devh)
    best = max(res, key=res.get)
    kind, a = best.split("|")[0], float(best.split("|")[1])
    torch.save(dict(kind=kind, alpha=a, di=int(Wn.shape[1]), do=int(Wo.shape[1]), state=maps[kind].state_dict()), os.path.join(out, "text_map.pt"))
    json.dump(dict(selected=best, devh_R1=res), open(os.path.join(out, "text_map.json"), "w"), indent=1)
    return best


def load(stage1, path):
    d = torch.load(path, map_location="cuda")
    phi = make_map(d["kind"], d["di"], d["do"]).cuda()
    phi.load_state_dict(d["state"])
    return TextQuery(stage1, phi.eval(), d["alpha"])
