# VIBRA: Backward-Compatible Cross-Modal Retrieval with Consensus Barycenters

VIBRA upgrades the query encoder of an image and text retrieval system while the gallery, encoded by the old model, stays untouched and only one encoder runs per query.

## Setup

```bash
pip install -r requirements.txt
export VIBRA_DATA=data        # raw data, images and manifests
export VIBRA_FEATURES=data/features
export VIBRA_RUNS=runs
```

## Data

Place the raw data under `$VIBRA_DATA/raw`:

| Folder | Source |
|---|---|
| `cc3m/` | WebDataset shards of `pixparse/cc3m-wds` (the shards listed in `splits/cc3m_*.csv`) |
| `cc3m_text/` | caption parquet files of the CC3M training split (`google-research-datasets/conceptual_captions`) |
| `flickr30k/` | `nlphuji/flickr30k` |
| `coco/` | `val2014.zip`, `annotations/captions_train2014.json` and the Karpathy split `yerevann/coco-karpathy` under `karpathy/` |
| `nocaps/` | `lmms-lab/NoCaps` (validation) |
| `xm3600/` | English split of Crossmodal-3600 as `data/en-00000-of-00001.parquet` |

The training images, the CC3M held out pairs and the COCO pairs used for model selection are fixed by the ID lists in `splits/`. COCO 5K is evaluated with the CxC positives shipped with the `eccv_caption` package (`pip install eccv_caption`).

```bash
python scripts/prepare_data.py
```

## Pipeline

The model pairs are `clip_b32_to_clip_l14`, `clip_b32_to_clip_b16`, `clip_b32_to_siglip2_so400m`, `clip_b32_to_metaclip_l14` and `metaclip_b32_to_siglip2_so400m`.

```bash
PAIR=clip_b32_to_clip_l14
python scripts/extract_features.py --pair $PAIR --aux --corpus   # old, new and auxiliary embeddings
python scripts/train_stage1.py   --pair $PAIR                     # relevance matching (Section 3.2)
python scripts/build_teacher.py  --pair $PAIR                     # view barycenter teacher (Section 3.3)
python scripts/train_student.py  --pair $PAIR --seed 0            # consensus inheritance (Section 3.4)
python scripts/train_text_map.py --pair $PAIR                     # text queries (Section 3.4)
python scripts/evaluate.py       --pair $PAIR --seed 0 --eccv_dir $(python -c "import eccv_caption, os; print(os.path.join(os.path.dirname(eccv_caption.__file__), 'data'))")
```

`build_teacher.py` uses the noise model weights in `configs/weights/` by default; `--weights estimate` recomputes them on the held out pairs and `--weights nested` uses the fixed rule unit(q_new + unit(q_old + q_aux)).

## Code

| File | Content |
|---|---|
| `vibra/stage1.py` | candidate pools, relevance matching loss, stage 1 query with the Procrustes anchor |
| `vibra/views.py` | views in the old space, noise model weights, barycenter teacher |
| `vibra/student.py` | low rank update of the image encoder trained on the teacher |
| `vibra/text.py` | text map and text queries |
| `vibra/evaluate.py` | R@1/5/10 against the old gallery |
| `vibra/extract.py`, `vibra/data.py`, `vibra/linalg.py` | feature extraction, data loading, closed form maps |

## License

MIT
