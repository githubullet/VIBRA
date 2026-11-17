"""Retrieval against the old gallery: image to text and text to image, R@1/5/10."""
import json
import os
import numpy as np
import torch
from .data import positives

PROTOCOLS = {"coco": "original", "cxc": "cxc", "eccv": "eccv"}


@torch.no_grad()
def ranks(Q, G, pos, chunk=2048):
    out = []
    for s in range(0, len(Q), chunk):
        S = Q[s:s + chunk] @ G.T
        p = pos[s:s + chunk]
        v = p >= 0
        ps = torch.gather(S, 1, p.clamp(min=0)).masked_fill(~v, -1e9).max(1).values
        out.append((S > ps[:, None]).sum(1))
    return torch.cat(out).cpu().numpy()


def recall(r, ks=(1, 5, 10)):
    return {f"R@{k}": float((r < k).mean() * 100) for k in ks}


def standard_positives(f):
    cap2img, img2caps = positives(f["image_ids"], f["cap_image_ids"])
    return None, {"std": (None, torch.as_tensor(img2caps, device="cuda"), None, torch.as_tensor(cap2img, device="cuda")[:, None])}


def _pad(rows):
    a = np.full((len(rows), max(len(r) for r in rows)), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        a[i, :len(r)] = r
    return torch.as_tensor(a, device="cuda")


def coco_positives(f, gt_dir):
    """COCO 5K test set with the original, CxC and ECCV Caption positive maps (eccv_caption package data in gt_dir)."""
    test = set(int(x) for x in np.load(os.path.join(gt_dir, "coco_test_ids.npy")))
    cid = np.array([int(str(c).replace("coco_c", "")) for c in f["caption_ids"]])
    keep = np.isin(cid, list(test))
    cid = cid[keep]
    iid = np.array([int(str(i).replace("coco_", "")) for i in f["image_ids"]])
    crow, irow = {c: j for j, c in enumerate(cid)}, {i: j for j, i in enumerate(iid)}
    out = {}
    for tag, name in PROTOCOLS.items():
        rd = lambda part: {int(k): [int(x) for x in v] for k, v in json.load(open(os.path.join(gt_dir, f"{name}_{part}.json"))).items()}
        i2c, c2i = rd("image_to_caption"), rd("caption_to_image")
        qi, qc = sorted(i2c, key=lambda i: irow[i]), sorted(c2i, key=lambda c: crow[c])
        out[tag] = (torch.as_tensor([irow[i] for i in qi], device="cuda"), _pad([[crow[c] for c in i2c[i] if c in crow] for i in qi]),
                    torch.as_tensor([crow[c] for c in qc], device="cuda"), _pad([[irow[i] for i in c2i[c]] for c in qc]))
    return torch.as_tensor(keep, device="cuda"), out


@torch.no_grad()
def evaluate_set(f, q_img, q_txt, set_name, gt_dir=None):
    """q_img: queries for all images of f; q_txt: queries for all captions of f; galleries are the old features."""
    if set_name == "coco_5k":
        keep, P = coco_positives(f, gt_dir)
        P = {"cxc": P["cxc"]}
    else:
        keep, P = standard_positives(f)
    g_img = f["old_img"]
    g_txt = f["old_txt"] if keep is None else f["old_txt"][keep]
    q_txt = q_txt if keep is None else q_txt[keep]
    (qi, pi, qc, pc), = P.values()
    out = {}
    if q_img is not None:
        out["i2t"] = recall(ranks(q_img if qi is None else q_img[qi], g_txt, pi))
    if q_txt is not None:
        out["t2i"] = recall(ranks(q_txt if qc is None else q_txt[qc], g_img, pc))
    return out
