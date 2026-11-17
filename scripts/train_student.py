"""Low rank update of the new image encoder. Usage: python scripts/train_student.py --pair clip_b32_to_clip_l14 [--seed 0]"""
import argparse
import os
import torch
import _common  # noqa: F401
from vibra import config as C
from vibra.data import load_feats, load_manifest
from vibra.stage1 import StageOne
from vibra import student

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=list(C.PAIRS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0)
    a = ap.parse_args()
    run = os.path.join(C.RUNS, a.pair)
    tr = load_feats(a.pair, C.TRAIN_SET)
    s1 = StageOne(tr, os.path.join(run, "stage1.pt"))
    tt = torch.load(os.path.join(run, "teacher.pt"))
    dev, devh = load_feats(a.pair, C.DEV_SET), load_feats(a.pair, C.DEVH_SET)
    dev["paths"], devh["paths"] = load_manifest(C.DEV_SET)[0].path.tolist(), load_manifest(C.DEVH_SET)[0].path.tolist()
    res = student.train(C.PAIRS[a.pair].new, s1, tr, tt["teacher"].cuda(), tt["aux"].cuda(), load_manifest(C.TRAIN_SET)[0].path.tolist(),
                        dev, devh, os.path.join(run, f"student_s{a.seed}"), seed=a.seed, max_steps=a.max_steps)
    print("selected", res["selected"], flush=True)
