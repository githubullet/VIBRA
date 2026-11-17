"""Manifests, cached features and positive maps."""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from . import config as C


def load_manifest(name):
    return (pd.read_parquet(os.path.join(C.MANIFESTS, f"{name}.images.parquet")),
            pd.read_parquet(os.path.join(C.MANIFESTS, f"{name}.captions.parquet")))


def load_feats(pair, name, device="cuda"):
    d = np.load(os.path.join(C.FEATURES, pair, f"{name}.npz"), allow_pickle=True)
    out = {k: F.normalize(torch.from_numpy(d[k]).float().to(device), dim=-1) for k in ("old_img", "old_txt", "new_img", "new_txt")}
    for k in ("image_ids", "caption_ids", "cap_image_ids"):
        out[k] = d[k]
    return out


def load_aux(name, ids, device="cuda"):
    d = np.load(os.path.join(C.FEATURES, "aux_h14", f"{name}.npz"), allow_pickle=True)
    X, vid = torch.from_numpy(d["emb"]).float().to(device), d["image_ids"]
    if not (len(vid) == len(ids) and (vid == ids).all()):
        pos = {v: i for i, v in enumerate(vid)}
        X = X[torch.as_tensor([pos[i] for i in ids], device=device)]
    return F.normalize(X, dim=-1)


def positives(image_ids, cap_image_ids):
    idx = {iid: i for i, iid in enumerate(image_ids)}
    cap2img = np.array([idx[c] for c in cap_image_ids], dtype=np.int64)
    order = np.argsort(cap2img, kind="stable")
    counts = np.bincount(cap2img, minlength=len(image_ids))
    img2caps = -np.ones((len(image_ids), max(int(counts.max()), 1)), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for i in range(len(image_ids)):
        img2caps[i, :counts[i]] = order[starts[i]:starts[i] + counts[i]]
    return cap2img, img2caps
