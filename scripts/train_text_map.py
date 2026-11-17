"""Text map M on the caption corpus. Usage: python scripts/train_text_map.py --pair clip_b32_to_clip_l14"""
import argparse
import os
import numpy as np
import _common  # noqa: F401
from vibra import config as C
from vibra.data import load_feats
from vibra.stage1 import StageOne
from vibra import text

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=list(C.PAIRS))
    a = ap.parse_args()
    feats, run = os.path.join(C.FEATURES, a.pair), os.path.join(C.RUNS, a.pair)
    s1 = StageOne(load_feats(a.pair, C.TRAIN_SET), os.path.join(run, "stage1.pt"))
    Wn, Wo = (np.load(os.path.join(feats, f"corpus_{t}.npy"), mmap_mode="r") for t in ("new", "old"))
    print("selected", text.fit(s1, Wn, Wo, load_feats(a.pair, C.DEVH_SET), run), flush=True)
