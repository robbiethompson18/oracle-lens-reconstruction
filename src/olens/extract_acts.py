"""Stage 1: lens-layer activations at assistant positions + whitening.

Per split: {acts}/{split}.npy  float32 [n, d] raw residual at OUTPUT of block L,
           {acts}/{split}.meta.jsonl rows {transcript_idx, pos, next_ids (<=32 following tokens)}.
Whitening fit on `whitening.n_positions` rows of the recon_train split.
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from .common import ResidualTap, Whitening, jsonl_writer, load_config, load_model, log, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    ecfg = cfg["extract"]
    layer = cfg["model"]["layer"]
    os.makedirs(cfg["paths"]["acts"], exist_ok=True)

    model, _tok = load_model(cfg["model"]["name"])
    model.eval()
    tap = ResidualTap(model, layer)
    rows = list(read_jsonl(os.path.join(cfg["paths"]["corpus"], "transcripts.jsonl")))

    for split, target in ecfg["target_positions"].items():
        npy = os.path.join(cfg["paths"]["acts"], f"{split}.npy")
        if os.path.exists(npy):
            log(f"[extract] {split} exists, skipping")
            continue
        rng = random.Random(sum(map(ord, split)))
        meta = jsonl_writer(os.path.join(cfg["paths"]["acts"], f"{split}.meta.jsonl"))
        buf = []
        for row in rows:
            if row["split"] != split or len(buf) >= target:
                continue
            ids = row["prompt_ids"] + row["response_ids"]
            start = len(row["prompt_ids"])
            cand = list(range(start, len(ids) - 1))  # positions with >=1 following token
            if len(cand) > ecfg["positions_per_transcript"]:
                cand = sorted(rng.sample(cand, ecfg["positions_per_transcript"]))
            with torch.inference_mode():
                model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
            h = tap.acts[0].float().cpu()
            # DECISION (storage): float32. ceselder stored fp16; Qwen residuals have massive-activation
            # dimensions that can overflow fp16 and silently poison the whitening fit.
            for pos in cand:
                buf.append(h[pos].numpy())
                meta.write(json.dumps({"transcript_idx": row["idx"], "pos": pos, "next_ids": ids[pos + 1 : pos + 33]}) + "\n")
        meta.close()
        arr = np.stack(buf) if buf else np.zeros((0, cfg["model"]["hidden_size"]), np.float32)
        assert np.isfinite(arr).all()
        np.save(npy, arr)
        log(f"[extract] {split}: {len(arr)} positions, mean norm {np.linalg.norm(arr, axis=1).mean():.1f}, max |x| {np.abs(arr).max():.1f}")

    wdir = cfg["paths"]["whitening"]
    if not os.path.exists(os.path.join(wdir, "W.npy")):
        a = np.load(os.path.join(cfg["paths"]["acts"], "recon_train.npy"), mmap_mode="r")
        n = min(len(a), cfg["whitening"]["n_positions"])
        idx = sorted(random.Random(0).sample(range(len(a)), n))
        log(f"[extract] fitting whitening on {n} rows")
        Whitening.fit(np.asarray(a[idx]), cfg["whitening"]["ridge_lambda_frac"]).save(wdir)
        log("[extract] whitening saved")


if __name__ == "__main__":
    main()
