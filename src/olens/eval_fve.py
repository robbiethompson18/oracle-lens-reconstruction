"""Stage 7: held-out FVE of a checkpoint at fixed (N, K), greedy, plus a random-dictionary-phrase baseline
and a qualitative dump of phrases next to the text the model actually went on to produce."""

import argparse
import json
import os
import random

import numpy as np
import torch

from .common import Whitening, build_oracle_prompt, fve_of, load_config, load_model, log, read_jsonl, whiten_unit
from .reconstructor import add_recon_adapter
from .rl_oracle import rollout, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="oracle adapter dir (sft or rl)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)

    from peft import PeftModel

    from .common import EmbedInjector as Inj

    model, tok = load_model(cfg["model"]["name"])
    model = PeftModel.from_pretrained(model, args.ckpt, adapter_name="oracle")
    model, recon = add_recon_adapter(model, cfg, trainable=False)
    model.eval()
    inj = Inj(model, cfg["oracle"]["inject_alpha"])
    mean_t, W_t = Whitening.load(cfg["paths"]["whitening"]).to_torch(model.device)
    acts = np.load(os.path.join(cfg["paths"]["acts"], "eval.npy"), mmap_mode="r")
    meta = list(read_jsonl(os.path.join(cfg["paths"]["acts"], "eval.meta.jsonl")))
    rows = list(range(min(args.rows, len(acts))))
    N, K = args.n, args.k
    dict_phrases = [r["ids"] for r in read_jsonl(os.path.join(cfg["paths"]["dictionary"], f"len{N}_phrases.jsonl"))]
    rng = random.Random(0)

    fv, fv_rand, dump = [], [], []
    for s in range(0, len(rows), 16):
        r = rows[s : s + 16]
        a = torch.tensor(np.stack([acts[i] for i in r]))
        prompts = [(N, K, *build_oracle_prompt(tok, N, K)) for _ in r]
        gens = rollout(model, tok, inj, prompts, a, args.temperature, K * (N + 1) + 16, 1)
        zs = whiten_unit(a.to(model.device), mean_t, W_t).cpu().numpy()
        _, f, parsed = score(recon, tok, gens, zs, [N] * len(r), [K] * len(r), 0, 0, model.device)
        fv.extend(f.tolist())
        # random-phrase baseline: K random dictionary phrases of length N through the same scorer
        rand = [[dict_phrases[rng.randrange(len(dict_phrases))] for _ in range(K)] for _ in r]
        from .reconstructor import build_inputs

        with torch.inference_mode():
            for i, ph in enumerate(rand):
                ii, mm = build_inputs(tok, ph, model.device)
                D = recon.embed(ii, mm).float().cpu().numpy()
                fv_rand.append(fve_of(D, zs[i])[0])
        for i, idx in enumerate(r):
            dump.append(
                {
                    "row": idx,
                    "fve": f[i].item(),
                    "phrases": [tok.decode(p) for p in parsed[i]],
                    "actual_next": tok.decode(meta[idx]["next_ids"]),
                }
            )
        log(f"[eval] {s + len(r)}/{len(rows)} fve {np.mean(fv):.4f} random {np.mean(fv_rand):.4f}")
    out = {
        "ckpt": args.ckpt,
        "n": N,
        "k": K,
        "rows": len(rows),
        "temperature": args.temperature,
        "fve": float(np.mean(fv)),
        "fve_random": float(np.mean(fv_rand)),
    }
    log(f"[eval] {json.dumps(out)}")
    with open(os.path.join(args.ckpt, f"eval_n{N}_k{K}_t{args.temperature}.json"), "w") as fh:
        json.dump({**out, "examples": dump[:64]}, fh, indent=1)


if __name__ == "__main__":
    main()
