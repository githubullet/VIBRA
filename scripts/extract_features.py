"""Old and new embeddings of every set for a model pair, auxiliary embeddings of the images, and the caption corpus.

Usage: python scripts/extract_features.py --pair clip_b32_to_clip_l14 [--aux] [--corpus]
"""
import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vibra import config as C
from vibra.data import load_manifest
from vibra.extract import Encoder

SETS = [C.TRAIN_SET, C.DEV_SET, C.DEVH_SET] + C.TEST_SETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=list(C.PAIRS))
    ap.add_argument("--aux", action="store_true")
    ap.add_argument("--corpus", action="store_true")
    a = ap.parse_args()
    pair = C.PAIRS[a.pair]
    out = os.path.join(C.FEATURES, a.pair)
    os.makedirs(out, exist_ok=True)
    old, new = Encoder(pair.old), Encoder(pair.new)
    for s in SETS:
        fn = os.path.join(out, f"{s}.npz")
        if os.path.exists(fn):
            continue
        imgs, caps = load_manifest(s)
        paths, texts = imgs.path.tolist(), caps.text.tolist()
        np.savez(fn, old_img=old.images(paths).numpy(), new_img=new.images(paths).numpy(), old_txt=old.texts(texts).numpy(),
                 new_txt=new.texts(texts).numpy(), image_ids=imgs.image_id.values, caption_ids=caps.caption_id.values, cap_image_ids=caps.image_id.values)
        print(a.pair, s, len(imgs), len(caps), flush=True)
    if a.corpus:
        texts = json.load(open(os.path.join(C.MANIFESTS, "cc3m_corpus.json")))
        for tag, enc in (("old", old), ("new", new)):
            fn = os.path.join(out, f"corpus_{tag}.npy")
            if not os.path.exists(fn):
                np.save(fn, enc.texts(texts, bs=1024).numpy().astype(np.float16))
    del old, new
    torch.cuda.empty_cache()
    if a.aux:
        aux_dir = os.path.join(C.FEATURES, "aux_h14")
        os.makedirs(aux_dir, exist_ok=True)
        aux = None
        for s in [C.TRAIN_SET] + C.TEST_SETS:
            fn = os.path.join(aux_dir, f"{s}.npz")
            if os.path.exists(fn):
                continue
            aux = aux or Encoder(C.AUX_H14)
            imgs, _ = load_manifest(s)
            np.savez(fn, emb=aux.images(imgs.path.tolist(), bs=128).numpy(), image_ids=imgs.image_id.values)
            print("aux", s, len(imgs), flush=True)


if __name__ == "__main__":
    main()
