"""Consensus inheritance: a low rank update of the new image encoder trained to reproduce the teacher (Section 3.4)."""
import json
import math
import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel
from peft import LoraConfig, PeftModel, get_peft_model
from . import config as C
from .data import positives
from .evaluate import ranks
from .linalg import ridge_fit

TARGET = r".*vision_model\.encoder\.layers\.\d+\.self_attn\.(q_proj|v_proj)"


class Images(Dataset):
    def __init__(self, paths, proc):
        self.paths, self.proc = paths, proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return self.proc(images=im, return_tensors="pt")["pixel_values"][0], i


def build(model_id, seed=0, rank=C.LORA_RANK):
    torch.manual_seed(seed)
    name, rev = model_id
    model = AutoModel.from_pretrained(name, revision=rev, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=TARGET, bias="none"))
    return model.cuda(), AutoImageProcessor.from_pretrained(name, revision=rev, use_fast=False)


def load(model_id, ckpt, merge=True):
    name, rev = model_id
    m = PeftModel.from_pretrained(AutoModel.from_pretrained(name, revision=rev, torch_dtype=torch.float32), ckpt)
    if merge:
        m = m.merge_and_unload()
    return m.cuda().eval(), AutoImageProcessor.from_pretrained(name, revision=rev, use_fast=False)


def image_features(model, x):
    m = model.base_model.model if isinstance(model, PeftModel) else model
    return F.normalize(m.get_image_features(pixel_values=x).float(), dim=-1)


@torch.no_grad()
def encode(model, proc, paths, bf16=False, bs=256, workers=16):
    was = model.training
    model.eval()
    out = []
    for x, _ in DataLoader(Images(paths, proc), batch_size=bs, num_workers=workers, pin_memory=True):
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out.append(image_features(model, x))
    model.train(was)
    return torch.cat(out)


@torch.no_grad()
def i2t_r1(q, f):
    _, img2caps = positives(f["image_ids"], f["cap_image_ids"])
    return float((ranks(q, f["old_txt"], torch.as_tensor(img2caps, device="cuda")) < 1).mean() * 100)


def train(model_id, stage1, tr, T, aux_target, paths, dev, devh, out, seed=0, lr=C.LORA_LR, epochs=C.LORA_EPOCHS, rank=C.LORA_RANK,
          mu=C.MU, micro=64, max_steps=0, workers=16):
    """Minimizes ||q0(z_theta) - T||^2 + mu (1 - cos(h(z_theta), aux)); keeps the checkpoint with the best COCO DEV R@1 among those
    whose CC3M DEV R@1 stays within one point of stage 1."""
    os.makedirs(out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Wa, ba = ridge_fit(tr["new_img"], aux_target, 1e-3)
    aux_head = torch.nn.Linear(Wa.shape[0], Wa.shape[1]).cuda()
    with torch.no_grad():
        aux_head.weight.copy_(Wa.T)
        aux_head.bias.copy_(ba.reshape(-1))
    model, proc = build(model_id, seed, rank)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(Images(paths, proc), batch_size=micro, shuffle=True, num_workers=workers, generator=g, drop_last=True,
                    pin_memory=True, persistent_workers=True)
    steps_ep = len(dl)
    total = max_steps or int(steps_ep * epochs)
    eval_every = max_steps or steps_ep // 2
    lora = [p for p in model.parameters() if p.requires_grad]
    params = lora + list(aux_head.parameters())
    opt = torch.optim.AdamW([{"params": lora, "lr": lr}, {"params": list(aux_head.parameters()), "lr": 3e-4}], weight_decay=0.01)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, total - warm)))))
    q0 = lambda z: stage1.query(z, "img")
    base_dev = i2t_r1(q0(dev["new_img"]), dev)
    evals = []

    def do_eval(step):
        e = dict(step=step, dev_R1=i2t_r1(q0(encode(model, proc, dev["paths"], bf16=True, workers=workers)), dev),
                 devh_R1=i2t_r1(q0(encode(model, proc, devh["paths"], bf16=True, workers=workers)), devh))
        evals.append(e)
        print("eval", e, flush=True)
        if step > 0:
            model.save_pretrained(os.path.join(out, f"ckpt_step{step}"))

    do_eval(0)
    t0, step = time.time(), 0
    model.train()
    while step < total:
        for x, idx in dl:
            x, idx = x.cuda(non_blocking=True), idx.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = image_features(model, x)
            loss = ((q0(z) - T[idx]) ** 2).sum(1).mean() + mu * (1 - F.cosine_similarity(aux_head(z), aux_target[idx], dim=-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                print(dict(step=step, loss=float(loss), sec=round(time.time() - t0)), flush=True)
            if step % eval_every == 0 or step == total:
                do_eval(step)
            if step >= total:
                break
    feasible = [e for e in evals if e["dev_R1"] >= base_dev - 1.0]
    best = max(feasible, key=lambda e: (e["devh_R1"], -e["step"]))
    res = dict(selected=best, evals=evals, base_dev=base_dev, steps=step, train_sec=time.time() - t0)
    json.dump(res, open(os.path.join(out, "student.json"), "w"), indent=1)
    return res
