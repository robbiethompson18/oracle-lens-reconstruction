"""Stage 2: train the reconstructor (LoRA "recon" + head) on (phrase -> preceding whitened activation).

Pairs: for each recon_train position, phrase = next N tokens with N ~ U{1..32} (resampled every epoch),
target = unit whitened activation at that position. Loss = ||pred - z||^2 on unit vectors (= 2 - 2cos),
exactly the paper's "MSE between the unit-normalised whitened prediction and target".
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch

from .common import Whitening, load_config, load_model, log, read_jsonl, whiten_unit
from .reconstructor import add_recon_adapter, build_inputs, save_recon


def load_split(cfg, split):
    acts = np.load(os.path.join(cfg["paths"]["acts"], f"{split}.npy"), mmap_mode="r")
    meta = list(read_jsonl(os.path.join(cfg["paths"]["acts"], f"{split}.meta.jsonl")))
    assert len(acts) == len(meta)
    return acts, meta


def sample_pairs(meta, n_min, n_max, rng):
    pairs = []
    for i, m in enumerate(meta):
        nx = m["next_ids"]
        if len(nx) < n_min:
            continue
        n = rng.randint(n_min, min(n_max, len(nx)))
        pairs.append((i, nx[:n]))
    rng.shuffle(pairs)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    rcfg = cfg["reconstructor"]
    torch.manual_seed(0)

    model, tok = load_model(cfg["model"]["name"], grad_checkpointing=rcfg.get("grad_checkpointing", False))
    model, recon = add_recon_adapter(model, cfg, trainable=True)
    model.set_adapter("recon")
    mean_t, W_t = Whitening.load(cfg["paths"]["whitening"]).to_torch(model.device)

    params = [p for p in model.parameters() if p.requires_grad] + list(recon.head.parameters())
    n_tr = sum(p.numel() for p in params)
    log(f"[recon] trainable params {n_tr / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=rcfg["lr"], betas=(0.9, 0.95), weight_decay=rcfg.get("weight_decay", 0.0))

    acts, meta = load_split(cfg, "recon_train")
    eval_acts, eval_meta = load_split(cfg, "eval")
    eval_pairs = sample_pairs(eval_meta, rcfg["n_min"], rcfg["n_max"], random.Random(1))[: rcfg["eval_rows"]]
    bs, n_pairs = rcfg["batch_size"], rcfg.get("n_pairs")
    rng = random.Random(0)
    step, t0 = 0, time.time()
    total_steps = rcfg["epochs"] * math.ceil(min(n_pairs or len(meta), len(meta)) / bs)
    warm = rcfg.get("warmup_steps", 100)

    def evaluate():
        model.eval()
        cos = []
        with torch.inference_mode():
            for i in range(0, len(eval_pairs), bs):
                chunk = eval_pairs[i : i + bs]
                ii, mm = build_inputs(tok, [p for _, p in chunk], model.device)
                pred = recon.embed(ii, mm)
                z = whiten_unit(torch.tensor(np.stack([eval_acts[r] for r, _ in chunk]), device=model.device), mean_t, W_t)
                cos.append((pred * z).sum(-1).cpu())
        model.train()
        return torch.cat(cos).mean().item()

    model.train()
    for epoch in range(rcfg["epochs"]):
        pairs = sample_pairs(meta, rcfg["n_min"], rcfg["n_max"], rng)
        if n_pairs:
            pairs = pairs[:n_pairs]
        for i in range(0, len(pairs), bs):
            chunk = pairs[i : i + bs]
            ii, mm = build_inputs(tok, [p for _, p in chunk], model.device)
            z = whiten_unit(torch.tensor(np.stack([acts[r] for r, _ in chunk]), device=model.device), mean_t, W_t)
            pred = recon.embed(ii, mm)
            loss = ((pred - z) ** 2).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, rcfg.get("grad_clip", 1.0))
            lr = rcfg["lr"] * min(1.0, (step + 1) / warm)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                cos = 1 - loss.item() / 2
                dt = (time.time() - t0) / step
                log(f"[recon] ep{epoch} step {step}/{total_steps} loss {loss.item():.4f} cos {cos:.4f} lr {lr:.1e} {dt:.2f}s/step")
            if step % rcfg.get("eval_every", 500) == 0:
                log(f"[recon] step {step} held-out cos {evaluate():.4f}")
            if step % rcfg.get("save_every", 1000) == 0:
                save_recon(model, recon.head, cfg["paths"]["ckpt_reconstructor"])
    log(f"[recon] final held-out cos {evaluate():.4f}")
    save_recon(model, recon.head, cfg["paths"]["ckpt_reconstructor"])
    log("[recon] saved")


if __name__ == "__main__":
    main()
