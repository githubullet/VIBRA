"""Barycenter teacher for every training image. Usage: python scripts/build_teacher.py --pair clip_b32_to_clip_l14 [--weights derived|nested|estimate]"""
import argparse
import json
import os
import torch
import _common  # noqa: F401
from vibra import config as C
from vibra.data import load_aux, load_feats
from vibra.stage1 import StageOne
from vibra import views as V

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=list(C.PAIRS))
    ap.add_argument("--weights", default="derived", choices=["derived", "nested", "estimate"])
    a = ap.parse_args()
    run = os.path.join(C.RUNS, a.pair)
    tr = load_feats(a.pair, C.TRAIN_SET)
    aux_tr = load_aux(C.TRAIN_SET, tr["image_ids"])
    s1 = StageOne(tr, os.path.join(run, "stage1.pt"))
    views = V.Views(s1, tr, aux_tr)
    if a.weights == "estimate":
        dev = load_feats(a.pair, C.DEV_SET)
        weights = V.noise_model(views(dev, load_aux(C.DEV_SET, dev["image_ids"])), dev)
        json.dump(weights, open(os.path.join(run, "weights.json"), "w"), indent=1)
    else:
        weights = V.load_weights(a.pair) if a.weights == "derived" else None
    with torch.no_grad():
        T = V.teacher(views(tr, aux_tr), weights)
    torch.save({"teacher": T.cpu(), "aux": aux_tr.cpu()}, os.path.join(run, "teacher.pt"))
    print("teacher", tuple(T.shape), "weights", weights, flush=True)
