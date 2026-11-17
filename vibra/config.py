"""Paths, model registry and hyperparameters."""
import os
from dataclasses import dataclass

DATA = os.environ.get("VIBRA_DATA", "data")
MANIFESTS = os.path.join(DATA, "manifests")
FEATURES = os.environ.get("VIBRA_FEATURES", os.path.join(DATA, "features"))
RUNS = os.environ.get("VIBRA_RUNS", "runs")
CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

CLIP_B32 = ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
METACLIP_B32 = ("facebook/metaclip-b32-fullcc2.5b", "3b0425e566b8e6a54cf4286916fc3858e8b5e469")
CLIP_L14 = ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41")
CLIP_B16 = ("openai/clip-vit-base-patch16", "57c216476eefef5ab752ec549e440a49ae4ae5f3")
SIGLIP2_SO400M = ("google/siglip2-so400m-patch14-224", "78e403963a4f6a3640d07803284752326fdf4edf")
METACLIP_L14 = ("facebook/metaclip-l14-fullcc2.5b", "d1d23eb55ad314a6091ac3351fa021508e4c677a")
AUX_H14 = ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "1c2b8495b28150b8a4922ee1c8edee224c284c0c")


@dataclass(frozen=True)
class Pair:
    old: tuple
    new: tuple


PAIRS = {
    "clip_b32_to_clip_l14": Pair(CLIP_B32, CLIP_L14),
    "clip_b32_to_clip_b16": Pair(CLIP_B32, CLIP_B16),
    "clip_b32_to_siglip2_so400m": Pair(CLIP_B32, SIGLIP2_SO400M),
    "clip_b32_to_metaclip_l14": Pair(CLIP_B32, METACLIP_L14),
    "metaclip_b32_to_siglip2_so400m": Pair(METACLIP_B32, SIGLIP2_SO400M),
}

TRAIN_SET = "cc3m_t100k"
DEV_SET = "cc3m_dev"
DEVH_SET = "coco_devh"
TEST_SETS = ["flickr_large", "coco_large", "nocaps_val", "coco_5k", "xm3600_en"]

K = 1024
TAUS = (0.03, 0.07, 0.1)
RIDGE_LAMBDAS = (1e-3, 1e-2, 1e-1)
STAGE1_EPOCHS = (1, 3, 5)
KD_WEIGHT = 1.0
BETA = 0.75
AUX_RIDGE_N = 20000
AUX_RIDGE_LAMBDA = 1e-3
LORA_RANK = 8
LORA_LR = 1e-4
LORA_EPOCHS = 2.0
MU = 1.0


def pair_dirs(pair):
    return os.path.join(FEATURES, pair), os.path.join(RUNS, pair)
