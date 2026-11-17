"""Build the manifests (<set>.images.parquet: image_id, path; <set>.captions.parquet: caption_id, image_id, text) from the raw data.

Usage: python scripts/prepare_data.py --steps cc3m,flickr,coco,nocaps,xm3600,coco_devh,corpus
"""
import argparse
import glob
import io
import json
import os
import sys
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vibra import config as C

RAW = os.path.join(C.DATA, "raw")
IMG = os.path.join(C.DATA, "images")
SPLITS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits")


def write_set(name, images, captions):
    os.makedirs(C.MANIFESTS, exist_ok=True)
    images.reset_index(drop=True).to_parquet(os.path.join(C.MANIFESTS, f"{name}.images.parquet"))
    captions.reset_index(drop=True).to_parquet(os.path.join(C.MANIFESTS, f"{name}.captions.parquet"))
    print(f"{name}: {len(images)} images, {len(captions)} captions", flush=True)


def build_cc3m():
    for name in ("cc3m_t100k", "cc3m_dev"):
        split = pd.read_csv(os.path.join(SPLITS, f"{name}.csv"), dtype=str)
        want, found = set(split.key), {}
        for shard in sorted(set(split.shard)):
            out = os.path.join(IMG, "cc3m", shard)
            os.makedirs(out, exist_ok=True)
            with tarfile.open(os.path.join(RAW, "cc3m", f"{shard}.tar")) as tf:
                files = {}
                for m in tf:
                    if m.isfile():
                        stem, ext = os.path.splitext(m.name)
                        files.setdefault(stem, {})[ext] = tf.extractfile(m).read()
            for stem, d in files.items():
                key = json.loads(d[".json"]).get("key", stem) if ".json" in d else stem
                if key in want and ".jpg" in d and ".txt" in d:
                    p = os.path.join(out, f"{key}.jpg")
                    if not os.path.exists(p):
                        open(p, "wb").write(d[".jpg"])
                    found[key] = (p, d[".txt"].decode("utf-8").strip())
        keys = [k for k in split.key if k in found]
        assert len(keys) == len(split), f"{name}: {len(split) - len(keys)} images missing"
        write_set(name, pd.DataFrame(dict(image_id=keys, path=[found[k][0] for k in keys])),
                  pd.DataFrame(dict(caption_id=keys, image_id=keys, text=[found[k][1] for k in keys])))


def build_flickr():
    img_dir = os.path.join(IMG, "flickr30k")
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
        with zipfile.ZipFile(os.path.join(RAW, "flickr30k", "flickr30k-images.zip")) as z:
            for n in z.namelist():
                if n.endswith(".jpg"):
                    open(os.path.join(img_dir, os.path.basename(n)), "wb").write(z.read(n))
    ann = pd.read_csv(os.path.join(RAW, "flickr30k", "flickr_annotations_30k.csv"))
    imgs, caps = [], []
    for r in ann.itertuples():
        iid = "flickr_" + r.filename.replace(".jpg", "")
        imgs.append(dict(image_id=iid, path=os.path.join(img_dir, r.filename)))
        for sid, t in zip(json.loads(r.sentids), json.loads(r.raw)):
            caps.append(dict(caption_id=f"flickr_c{sid}", image_id=iid, text=t))
    write_set("flickr_large", pd.DataFrame(imgs), pd.DataFrame(caps))


def build_coco():
    img_dir = os.path.join(IMG, "coco_val2014")
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
        with zipfile.ZipFile(os.path.join(RAW, "coco", "val2014.zip")) as z:
            for n in z.namelist():
                if n.endswith(".jpg"):
                    open(os.path.join(img_dir, os.path.basename(n)), "wb").write(z.read(n))
    kp = {s: pd.read_parquet(os.path.join(RAW, "coco", "karpathy", "data", f"{s}-00000-of-00001.parquet")) for s in ("validation", "restval", "test")}

    def make(dfs):
        imgs, caps = [], []
        for df in dfs:
            for r in df.itertuples():
                iid = f"coco_{r.cocoid}"
                imgs.append(dict(image_id=iid, path=os.path.join(img_dir, r.filename)))
                for sid, t in zip(r.sentids, r.sentences):
                    caps.append(dict(caption_id=f"coco_c{sid}", image_id=iid, text=t.strip()))
        return pd.DataFrame(imgs), pd.DataFrame(caps)

    write_set("coco_large", *make([kp["validation"], kp["restval"]]))
    write_set("coco_5k", *make([kp["test"]]))


def build_nocaps():
    img_dir = os.path.join(IMG, "nocaps")
    os.makedirs(img_dir, exist_ok=True)
    imgs, caps = [], []
    for f in sorted(glob.glob(os.path.join(RAW, "nocaps", "data", "validation-*.parquet"))):
        for b in pq.ParquetFile(f).iter_batches(batch_size=200):
            for r in b.to_pylist():
                iid, p = f"nocaps_{r['image_open_images_id']}", os.path.join(img_dir, r["image_file_name"])
                if not os.path.exists(p):
                    open(p, "wb").write(r["image"]["bytes"])
                imgs.append(dict(image_id=iid, path=p))
                for aid, t in zip(r["annotations_ids"], r["annotations_captions"]):
                    caps.append(dict(caption_id=f"nocaps_c{aid}", image_id=iid, text=t))
    write_set("nocaps_val", pd.DataFrame(imgs), pd.DataFrame(caps))


def build_xm3600():
    img_dir = os.path.join(IMG, "xm3600")
    os.makedirs(img_dir, exist_ok=True)
    imgs, caps = [], []
    for b in pq.ParquetFile(os.path.join(RAW, "xm3600", "data", "en-00000-of-00001.parquet")).iter_batches(batch_size=200):
        for r in b.to_pylist():
            iid, p = f"xm_{r['image_id']}", os.path.join(img_dir, f"{r['image_id']}.jpg")
            if not os.path.exists(p):
                open(p, "wb").write(r["image"]["bytes"])
            imgs.append(dict(image_id=iid, path=p))
            for j, t in enumerate(r["captions"]):
                caps.append(dict(caption_id=f"{iid}_c{j}", image_id=iid, text=t.strip()))
    write_set("xm3600_en", pd.DataFrame(imgs), pd.DataFrame(caps))


def build_coco_devh():
    ann = json.load(open(os.path.join(RAW, "coco", "annotations", "captions_train2014.json")))
    images = {r["id"]: r for r in ann["images"]}
    text = {a["id"]: a["caption"].strip() for a in ann["annotations"]}
    ids = pd.read_csv(os.path.join(SPLITS, "coco_devh_images.csv")).coco_image_id.tolist()
    caps = pd.read_csv(os.path.join(SPLITS, "coco_devh_captions.csv"))
    img_dir = os.path.join(IMG, "coco_train2014")
    os.makedirs(img_dir, exist_ok=True)

    def fetch(i):
        p = os.path.join(img_dir, images[i]["file_name"])
        if not os.path.exists(p):
            open(p, "wb").write(urllib.request.urlopen(images[i]["coco_url"], timeout=60).read())
        return p

    with ThreadPoolExecutor(32) as ex:
        paths = list(ex.map(fetch, ids))
    write_set("coco_devh", pd.DataFrame(dict(image_id=[f"cocotr_{i}" for i in ids], path=paths)),
              pd.DataFrame(dict(caption_id=[f"cocotr_c{c}" for c in caps.coco_caption_id], image_id=[f"cocotr_{i}" for i in caps.coco_image_id],
                                text=[text[c] for c in caps.coco_caption_id])))


def build_corpus():
    """CC3M training captions for the text map, without exact duplicates and without any caption of the held out or test sets."""
    texts = set()
    for f in sorted(glob.glob(os.path.join(RAW, "cc3m_text", "*.parquet"))):
        texts |= set(pq.read_table(f, columns=["caption"]).column(0).to_pylist())
    held = set()
    for s in [C.DEV_SET, C.DEVH_SET] + C.TEST_SETS:
        held |= set(t.strip() for t in pd.read_parquet(os.path.join(C.MANIFESTS, f"{s}.captions.parquet")).text)
    texts = sorted(t for t in texts if t and t.strip() not in held)
    json.dump(texts, open(os.path.join(C.MANIFESTS, "cc3m_corpus.json"), "w"))
    print("corpus", len(texts), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="cc3m,flickr,coco,nocaps,xm3600,coco_devh,corpus")
    steps = ap.parse_args().steps.split(",")
    builders = dict(cc3m=build_cc3m, flickr=build_flickr, coco=build_coco, nocaps=build_nocaps, xm3600=build_xm3600, coco_devh=build_coco_devh, corpus=build_corpus)
    for s in steps:
        builders[s]()
