"""Diagnostic: does the SFT oracle actually read the injected activation?

For held-out teacher examples, compare the per-token NLL of the teacher phrases under the SFT adapter with the
correct activation injected vs a shuffled one (rows rolled by one), across injection scales. No gap between
correct and shuffled means the conditioning was never learned (or cannot be read at that scale).
"""

import argparse
import random

import numpy as np
import torch

from .common import load_config, load_model, log, logit_head, make_injector, unwrap_peft
from .oracle_data import TeacherStore, build_sft_example, chunked_ce, collate


def nll(model, lm_head, inj, batch, acts):
    inj.set(batch["inject_pos"], acts)
    out = unwrap_peft(model).model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    s, n = chunked_ce(out.last_hidden_state, batch["labels"], lm_head)
    inj.clear()
    return float(s), int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--alphas", default="16000,4000,1000,300,100")
    args = ap.parse_args()
    cfg = load_config(args.config)
    from peft import PeftModel

    model, tok = load_model(cfg["model"]["name"])
    model = PeftModel.from_pretrained(model, args.ckpt, adapter_name="oracle")
    model.eval()
    lm_head = logit_head(model)
    inj = make_injector(model, cfg)
    store = TeacherStore(cfg)
    rows = store.rows()[-args.rows :]  # tail of the teacher split, never seen at this (L, K) draw anyway
    for alpha in [float(a) for a in args.alphas.split(",")]:
        inj.alpha = alpha
        rng = random.Random(7)
        tot = {"correct": [0.0, 0], "shuffled": [0.0, 0]}
        for i in range(0, len(rows), 8):
            exs = [e for e in (build_sft_example(store, tok, cfg, r, rng) for r in rows[i : i + 8]) if e]
            if len(exs) < 2:
                continue
            batch = collate(exs, tok.pad_token_id)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.inference_mode():
                for name, acts in (("correct", batch["acts"]), ("shuffled", torch.roll(batch["acts"], 1, 0))):
                    s, n = nll(model, lm_head, inj, batch, acts)
                    tot[name][0] += s
                    tot[name][1] += n
        c, s = tot["correct"][0] / tot["correct"][1], tot["shuffled"][0] / tot["shuffled"][1]
        log(f"[diag] alpha {alpha:>8.0f}: NLL/token correct {c:.3f}  shuffled {s:.3f}  gap {s - c:+.3f}")
    log(f"[diag] eval acts norm mean {np.linalg.norm(np.asarray(store.acts[-8:]), axis=1).mean():.1f}")


if __name__ == "__main__":
    main()
