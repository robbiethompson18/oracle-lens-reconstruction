"""Stage 3: phrase dictionary. From `dict`-split positions take the 2/4/8/16/32-token continuations,
dedupe per length, drop phrases containing a newline token, embed through the reconstructor.

Output: {dictionary}/len{L}_dirs.npy float16 [m, d] unit rows; len{L}_phrases.jsonl {ids, text}.
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from .common import load_config, load_model, log, read_jsonl
from .reconstructor import add_recon_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    dcfg = cfg["dictionary"]
    out = cfg["paths"]["dictionary"]
    os.makedirs(out, exist_ok=True)

    model, tok = load_model(cfg["model"]["name"])
    model, recon = add_recon_adapter(model, cfg, trainable=False)
    model.eval()
    nl_ids = {t for t in range(len(tok)) if "\n" in tok.decode([t])}

    meta = list(read_jsonl(os.path.join(cfg["paths"]["acts"], "dict.meta.jsonl")))
    random.Random(0).shuffle(meta)
    meta = meta[: dcfg["n_starts"]]
    for L in dcfg["lengths"]:
        if os.path.exists(os.path.join(out, f"len{L}_dirs.npy")):
            log(f"[dict] len{L} exists, skipping")
            continue
        seen, phrases = set(), []
        for m in meta:
            nx = m["next_ids"]
            if len(nx) < L:
                continue
            p = tuple(nx[:L])
            if p in seen or any(t in nl_ids for t in p):
                continue
            seen.add(p)
            phrases.append(list(p))
        dirs = recon.embed_phrases(tok, phrases, model.device, batch_size=dcfg["embed_batch_size"])
        np.save(os.path.join(out, f"len{L}_dirs.npy"), dirs.numpy().astype(np.float16))
        with open(os.path.join(out, f"len{L}_phrases.jsonl"), "w") as f:
            for p in phrases:
                f.write(json.dumps({"ids": p, "text": tok.decode(p)}) + "\n")
        log(f"[dict] len{L}: {len(phrases)} phrases")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
