"""Stage 5: supervised oracle. LoRA adapter "oracle" on the shared base; the activation is injected at the
marker's embedding; loss = token CE on the teacher's phrase list.
"""

import argparse
import math
import random
import time

import torch

from .common import EmbedInjector, load_config, load_model, log, logit_head, lora_config, unwrap_peft
from .oracle_data import TeacherStore, build_sft_example, chunked_ce, collate


def save_oracle(model, ckpt):
    import os

    os.makedirs(ckpt, exist_ok=True)
    model.save_pretrained(ckpt, selected_adapters=["oracle"])
    sub = os.path.join(ckpt, "oracle")
    if os.path.isdir(sub):
        for f in os.listdir(sub):
            os.replace(os.path.join(sub, f), os.path.join(ckpt, f))
        os.rmdir(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    scfg = cfg["oracle"]["sft"]
    torch.manual_seed(0)

    from peft import get_peft_model

    model, tok = load_model(cfg["model"]["name"], grad_checkpointing=True)
    model = get_peft_model(model, lora_config(cfg["oracle"]["lora_r"], cfg["oracle"]["lora_alpha"]), adapter_name="oracle")
    model.print_trainable_parameters()
    inj = EmbedInjector(model, cfg["oracle"]["inject_alpha"])
    lm_head = logit_head(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=scfg["lr"], betas=(0.9, 0.95), weight_decay=0.0)

    store = TeacherStore(cfg)
    rows = store.rows()
    log(f"[sft] {len(rows)} labelled activations")
    bs, accum = scfg["micro_batch"], scfg["grad_accum"]
    steps_per_epoch = math.ceil(len(rows) / (bs * accum))
    total = scfg["epochs"] * steps_per_epoch
    warm = scfg.get("warmup_steps", 50)
    step, micro, t0 = 0, 0, time.time()
    model.train()
    for epoch in range(scfg["epochs"]):
        rng = random.Random(1000 + epoch)
        order = rows[:]
        rng.shuffle(order)
        for i in range(0, len(order), bs):
            exs = [e for e in (build_sft_example(store, tok, cfg, r, rng) for r in order[i : i + bs]) if e]
            if not exs:
                continue
            batch = collate(exs, tok.pad_token_id)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            inj.set(batch["inject_pos"], batch["acts"])
            out = unwrap_peft(model).model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            loss_sum, n_tok = chunked_ce(out.last_hidden_state, batch["labels"], lm_head)
            loss = loss_sum / n_tok
            (loss / accum).backward()
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            lr = scfg["lr"] * min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * step / total)) if scfg.get("cosine") else 1.0)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)
            inj.clear()
            step += 1
            if step % 10 == 0:
                log(f"[sft] ep{epoch} step {step}/{total} loss {loss.item():.4f} lr {lr:.1e} {(time.time() - t0) / step:.1f}s/step")
            if step % scfg.get("save_every", 500) == 0:
                save_oracle(model, cfg["paths"]["ckpt_oracle_sft"])
    save_oracle(model, cfg["paths"]["ckpt_oracle_sft"])
    log("[sft] saved")


if __name__ == "__main__":
    main()
