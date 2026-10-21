"""Stage 7: held-out FVE of a checkpoint at fixed (N, K), greedy, plus a random-dictionary-phrase baseline, a
true-continuation baseline (the next tokens the model actually produced, cut into N-token phrases), the CJK share of
the phrases, and a qualitative dump of phrases next to the text the model actually went on to produce.

`--acts` (repeatable) reads other activation sets, e.g. a Chinese WildChat split; `--temperature` is repeatable too."""

import argparse
import json
import os
import random
import re

import numpy as np
import torch

from .common import Whitening, build_oracle_prompt, cjk_share, fve_of, load_config, load_model, log, make_injector, read_jsonl, whiten_unit
from .reconstructor import add_recon_adapter, build_inputs
from .rl_oracle import rollout, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="oracle adapter dir (sft or rl)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--temperature", type=float, action="append", help="repeatable; default 0.0")
    ap.add_argument("--acts", action="append", help="activation dir(s) with eval.npy + eval.meta.jsonl; default paths.acts")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)

    from peft import PeftModel

    model, tok = load_model(cfg["model"]["name"])
    model = PeftModel.from_pretrained(model, args.ckpt, adapter_name="oracle")
    model, recon = add_recon_adapter(model, cfg, trainable=False)
    model.eval()
    inj = make_injector(model, cfg)
    mean_t, W_t = Whitening.load(cfg["paths"]["whitening"]).to_torch(model.device)
    N, K = args.n, args.k
    dict_phrases = [r["ids"] for r in read_jsonl(os.path.join(cfg["paths"]["dictionary"], f"len{N}_phrases.jsonl"))]
    strip = cfg["oracle"]["rl"].get("strip_leading_ws", False)

    def embed_fve(phrase_lists, zs):
        out = []
        with torch.inference_mode():
            for i, ph in enumerate(phrase_lists):
                ii, mm = build_inputs(tok, ph, model.device)
                out.append(fve_of(recon.embed(ii, mm).float().cpu().numpy(), zs[i])[0])
        return out

    for acts_dir in args.acts or [cfg["paths"]["acts"]]:
        acts = np.load(os.path.join(acts_dir, "eval.npy"), mmap_mode="r")
        meta = list(read_jsonl(os.path.join(acts_dir, "eval.meta.jsonl")))
        rows = list(range(min(args.rows, len(acts))))
        tag = "" if acts_dir == cfg["paths"]["acts"] else "_" + os.path.basename(os.path.dirname(acts_dir.rstrip("/")))
        for temperature in args.temperature or [0.0]:
            rng = random.Random(0)
            fv, fv_rand, fv_true, phrases_all, dump = [], [], [], [], []
            for s in range(0, len(rows), 32):
                r = rows[s : s + 32]
                a = torch.tensor(np.stack([acts[i] for i in r]))
                prompts = [(N, K, *build_oracle_prompt(tok, N, K)) for _ in r]
                gens = rollout(model, tok, inj, prompts, a, temperature, K * (N + 1) + 16, 1)
                zs = whiten_unit(a.to(model.device), mean_t, W_t).cpu().numpy()
                _, f, parsed = score(recon, tok, gens, zs, [N] * len(r), [K] * len(r), 0, 0, model.device, strip=strip)
                fv.extend(f.tolist())
                # random-phrase baseline: K random dictionary phrases of length N through the same scorer
                fv_rand.extend(embed_fve([[dict_phrases[rng.randrange(len(dict_phrases))] for _ in range(K)] for _ in r], zs))
                # true-continuation baseline: the (<=32) tokens that actually followed, as N-token phrases
                nxt = [meta[i]["next_ids"] for i in r]
                fv_true.extend(embed_fve([[n[j : j + N] for j in range(0, len(n), N)] for n in nxt], zs))
                for i, idx in enumerate(r):
                    phrases = [tok.decode(p) for p in parsed[i]]
                    phrases_all.extend(phrases)
                    dump.append({"row": idx, "fve": f[i].item(), "phrases": phrases, "actual_next": tok.decode(meta[idx]["next_ids"])})
                means = f"fve {np.mean(fv):.4f} random {np.mean(fv_rand):.4f} true {np.mean(fv_true):.4f}"
                log(f"[eval{tag} t={temperature}] {s + len(r)}/{len(rows)} {means}")
            out = {
                "ckpt": args.ckpt,
                "acts": acts_dir,
                "n": N,
                "k": K,
                "rows": len(rows),
                "temperature": temperature,
                "fve": float(np.mean(fv)),
                "fve_random": float(np.mean(fv_rand)),
                "fve_true_next": float(np.mean(fv_true)),
                "cjk_phrases": cjk_share(phrases_all),
                "cjk_actual_next": cjk_share(d["actual_next"] for d in dump),
                "frac_phrases_with_ideograph": float(np.mean([bool(re.search(r"[\u3400-\u9fff]", p)) for p in phrases_all])),
            }
            log(f"[eval] {json.dumps(out)}")
            with open(os.path.join(args.ckpt, f"eval{tag}_n{N}_k{K}_t{temperature}.json"), "w") as fh:
                json.dump({**out, "examples": dump}, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
