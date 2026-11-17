"""Stage 1 heads with relevance matching. Usage: python scripts/train_stage1.py --pair clip_b32_to_clip_l14 [--seed 0]"""
import argparse
import os
import _common  # noqa: F401
from vibra import config as C
from vibra.data import load_feats
from vibra import stage1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=list(C.PAIRS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sel = stage1.fit(load_feats(a.pair, C.TRAIN_SET), load_feats(a.pair, C.DEV_SET), os.path.join(C.RUNS, a.pair), seed=a.seed)
    print("selected", sel["tau"], sel["dev"], flush=True)
