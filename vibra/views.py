"""Views in the old space, noise model weights and the barycenter teacher (Section 3.3)."""
import json
import os
import torch
import torch.nn.functional as F
from . import config as C
from .data import positives
from .linalg import ridge_fit

VIEWS = ("new", "old", "aux")
nz = lambda x: F.normalize(x, dim=-1)


class Views:
    """new: stage 1 head f(z); old: old image embedding; aux: auxiliary embedding mapped by ridge regression onto the old one."""

    def __init__(self, stage1, tr, aux_tr):
        n = C.AUX_RIDGE_N
        self.stage1 = stage1
        self.W, self.b = ridge_fit(aux_tr[:n], tr["old_img"][:n], C.AUX_RIDGE_LAMBDA)

    @torch.no_grad()
    def __call__(self, f, aux):
        return {"new": nz(self.stage1.heads["img"](f["new_img"])), "old": nz(f["old_img"]), "aux": nz(aux @ self.W + self.b)}


@torch.no_grad()
def noise_model(Q, f, topn=20):
    """Separation a_v and covariance of the rescaled noise of each view on held out pairs; returns the minimum variance weights w*."""
    names = list(Q)
    _, img2caps = positives(f["image_ids"], f["cap_image_ids"])
    i2c = torch.as_tensor(img2caps, device="cuda")
    S = {v: Q[v] @ f["old_txt"].T for v in names}
    n, m = S[names[0]].shape
    pos = torch.zeros(n, m, dtype=torch.bool, device="cuda")
    r = torch.arange(n, device="cuda")[:, None].expand_as(i2c)
    pos[r[i2c >= 0], i2c[i2c >= 0]] = True
    cand = pos.clone()
    for v in names:
        cand.scatter_(1, S[v].masked_fill(pos, -1e9).topk(topn, 1).indices, True)
    neg = cand & ~pos
    a, E = {}, {}
    for v in names:
        mneg = (S[v] * neg).sum(1) / neg.sum(1)
        mpos = (S[v] * pos).sum(1) / pos.sum(1)
        a[v] = float((mpos - mneg).mean())
        E[v] = ((S[v] - mneg[:, None] - a[v] * pos) / a[v])[cand]
    Sigma = torch.cov(torch.stack([E[v] for v in names]).double())
    w = torch.linalg.solve(Sigma, torch.ones(len(names), dtype=Sigma.dtype, device=Sigma.device))
    w = (w / w.sum()).tolist()
    return {"w": dict(zip(names, w)), "a": a}


def load_weights(pair):
    return json.load(open(os.path.join(C.CONFIGS, "weights", f"{pair}.json")))


def teacher(Q, weights=None):
    """T = unit(sum_v c_v q_v) with c_v proportional to w*_v / a_v; weights=None gives the nested rule unit(q_new + unit(q_old + q_aux))."""
    if weights is None:
        return nz(Q["new"] + nz(Q["old"] + Q["aux"]))
    w, a = weights["w"], weights["a"]
    return nz(sum(w[v] / a[v] * Q[v] for v in VIEWS))
