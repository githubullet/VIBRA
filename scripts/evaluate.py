"""R@1/5/10 against the old gallery on the five test sets.
Usage: python scripts/evaluate.py --pair clip_b32_to_clip_l14 [--seed 0] [--eccv_dir PATH_TO_eccv_caption/data]"""
import argparse
import json
import os
import torch
import _common  # noqa: F401
from vibra import config as C
from vibra.data import load_feats, load_manifest
from vibra.evaluate import evaluate_set
from vibra.stage1 import StageOne
from vibra import student, text

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=list(C.PAIRS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eccv_dir", default=None)
    ap.add_argument("--sets", default=",".join(C.TEST_SETS))
    a = ap.parse_args()
    run = os.path.join(C.RUNS, a.pair)
    s1 = StageOne(load_feats(a.pair, C.TRAIN_SET), os.path.join(run, "stage1.pt"))
    tq = text.load(s1, os.path.join(run, "text_map.pt"))
    sdir = os.path.join(run, f"student_s{a.seed}")
    step = json.load(open(os.path.join(sdir, "student.json")))["selected"]["step"]
    model = None
    if step > 0:
        model, proc = student.load(C.PAIRS[a.pair].new, os.path.join(sdir, f"ckpt_step{step}"))
    res = {}
    for s in a.sets.split(","):
        f = load_feats(a.pair, s)
        z = f["new_img"] if model is None else student.encode(model, proc, load_manifest(s)[0].path.tolist())
        with torch.no_grad():
            res[s] = evaluate_set(f, s1.query(z, "img"), tq(f["new_txt"]), s, a.eccv_dir)
        print(s, res[s], flush=True)
    json.dump(res, open(os.path.join(run, f"results_s{a.seed}.json"), "w"), indent=1)
