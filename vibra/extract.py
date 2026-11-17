"""Image and caption embeddings of the old, new and auxiliary encoders."""
import os
import string
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_PUNCT = str.maketrans("", "", string.punctuation)


def canonicalize(t):
    return " ".join(t.replace("_", " ").translate(_PUNCT).lower().split()).strip()


class Images(Dataset):
    def __init__(self, paths, proc):
        self.paths, self.proc = paths, proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            im = Image.new("RGB", (224, 224))
        return self.proc(images=im, return_tensors="pt")["pixel_values"][0]


class Encoder:
    def __init__(self, model_id, lowercase=False):
        name, rev = model_id
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = AutoModel.from_pretrained(name, revision=rev, torch_dtype=torch.float32).cuda().eval()
        self.proc = AutoImageProcessor.from_pretrained(name, revision=rev, use_fast=False)
        self.tok = AutoTokenizer.from_pretrained(name, revision=rev)
        self.siglip = "siglip" in name.lower()
        self.lowercase = lowercase

    @torch.no_grad()
    def images(self, paths, bs=256, workers=16):
        dl = DataLoader(Images(paths, self.proc), batch_size=bs, num_workers=workers, pin_memory=True)
        return torch.cat([F.normalize(self.model.get_image_features(pixel_values=x.cuda()).float(), dim=-1).cpu() for x in dl])

    @torch.no_grad()
    def texts(self, texts, bs=512):
        if self.siglip:
            texts, kw = [canonicalize(t) for t in texts], dict(padding="max_length", max_length=64)
        else:
            texts, kw = ([t.lower() for t in texts] if self.lowercase else list(texts)), dict(padding=True, max_length=77)
        out = []
        for s in range(0, len(texts), bs):
            t = self.tok(texts[s:s + bs], truncation=True, return_tensors="pt", **kw).to("cuda")
            out.append(F.normalize(self.model.get_text_features(**t).float(), dim=-1).cpu())
        return torch.cat(out)
