"""Stage 6: RL-refine the oracle. Reward = whitened FVE of the NNLS refit of the reconstructor directions
of the generated phrases, minus small format penalties. Policy = "oracle" adapter; reward model = frozen
"recon" adapter + head on the SAME base (early exit at the lens layer), so one 27B copy serves both.

DECISION (RL algorithm): on-policy REINFORCE with a per-prompt group-normalised baseline (what ceselder's
July code calls GRPO: no PPO ratio, no clipping, one update per rollout batch) plus a KL penalty to the SFT
adapter (WorkspaceBench used beta 0.02; ceselder used 0). Rollouts via HF generate for the pilot; a vLLM
rollout path is the obvious upgrade for the full run.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from .common import (
    EmbedInjector,
    Whitening,
    build_oracle_prompt,
    fve_of,
    load_config,
    load_model,
    log,
    parse_generated_phrases,
    unwrap_peft,
    whiten_unit,
)
from .oracle_data import sample_rl_prompt
from .reconstructor import add_recon_adapter, build_inputs
from .train_oracle_sft import save_oracle

N_EVAL_RESERVED = 512


def left_pad(rows, pad_id):
    maxlen = max(len(r) for r in rows)
    ids = torch.full((len(rows), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), maxlen), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, maxlen - len(r) :] = torch.tensor(r)
        mask[i, maxlen - len(r) :] = 1
    return ids, mask, [maxlen - len(r) for r in rows]


@torch.inference_mode()
def rollout(model, tok, inj, prompts, acts, temperature, max_new, n_samples, batch=32):
    """prompts: list of (N, K, ids, pos). Returns list of generated id lists (n_samples per prompt).

    Generation is chunked to `batch` sequences: each of the 48 Gated-DeltaNet layers keeps an fp32
    recurrent state per sequence, so 128 sequences at once OOMed an 80 GB H100 next to the 54 GB base.
    """
    model.set_adapter("oracle")
    model.eval()
    rows = [p[2] for p in prompts for _ in range(n_samples)]
    all_pos = [p[3] for p in prompts for _ in range(n_samples)]
    all_vec = acts.repeat_interleave(n_samples, dim=0)
    gens = []
    for s in range(0, len(rows), batch):
        ids, mask, offs = left_pad(rows[s : s + batch], tok.pad_token_id)
        pos = torch.tensor([all_pos[s + k] + offs[k] for k in range(len(offs))])
        inj.set(pos, all_vec[s : s + batch])
        out = model.generate(
            input_ids=ids.to(model.device),
            attention_mask=mask.to(model.device),
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0,
            max_new_tokens=max_new,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
        inj.clear()
        gens.extend(out[:, ids.shape[1] :].cpu().tolist())
        del out
    model.train()
    return gens


def score(recon, tok, gens, zs, Ns, Ks, pen_phrase, pen_token, device, batch=128):
    """Returns (rewards, fves, parsed phrases)."""
    parsed = [parse_generated_phrases(g, tok, K) for g, K in zip(gens, Ks)]
    flat = [(i, p) for i, ph in enumerate(parsed) for p in ph]
    dirs = []
    with torch.inference_mode():
        for s in range(0, len(flat), batch):
            ii, mm = build_inputs(tok, [p for _, p in flat[s : s + batch]], device)
            dirs.append(recon.embed(ii, mm).float().cpu().numpy())
    dirs = np.concatenate(dirs) if dirs else np.zeros((0, zs.shape[1]), np.float32)
    owner = np.array([i for i, _ in flat])
    rewards, fves = [], []
    for i in range(len(gens)):
        D = dirs[owner == i] if len(flat) else dirs
        fve = fve_of(D, zs[i])[0] if len(D) else 0.0
        pen = pen_phrase * abs(len(parsed[i]) - Ks[i]) + pen_token * sum(abs(len(p) - Ns[i]) for p in parsed[i])
        rewards.append(fve - pen)
        fves.append(fve)
    return np.array(rewards, np.float32), np.array(fves, np.float32), parsed


def completion_logprobs(model, lm_head, ids, mask, comp_mask, inj, pos, vec, chunk=1024):
    """Sum of token log-probs over completion tokens per row, with grad. Also returns per-token logps for KL."""
    inj.set(pos, vec)
    h = unwrap_peft(model).model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    inj.clear()
    h = h[:, :-1]
    tgt = ids[:, 1:]
    cm = comp_mask[:, 1:]
    B, T, d = h.shape
    hf, tf, cf = h.reshape(-1, d)[cm.reshape(-1)], tgt.reshape(-1)[cm.reshape(-1)], cm.reshape(-1)
    lps = []
    for i in range(0, len(tf), chunk):
        logits = lm_head(hf[i : i + chunk]).float()
        lps.append(torch.gather(logits, 1, tf[i : i + chunk, None]).squeeze(1) - torch.logsumexp(logits, -1))
    lp = torch.cat(lps) if lps else h.new_zeros(0)
    full = h.new_zeros(B * T, dtype=torch.float32)
    full[cf] = lp
    return full.view(B, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    rcfg = cfg["oracle"]["rl"]
    torch.manual_seed(0)
    rng = random.Random(0)

    from peft import PeftModel

    model, tok = load_model(cfg["model"]["name"], grad_checkpointing=True)
    model = PeftModel.from_pretrained(model, cfg["paths"]["ckpt_oracle_sft"], adapter_name="oracle", is_trainable=True)
    model.load_adapter(cfg["paths"]["ckpt_oracle_sft"], adapter_name="sft_ref", is_trainable=False)
    model, recon = add_recon_adapter(model, cfg, trainable=False)
    model.set_adapter("oracle")
    lm_head = unwrap_peft(model).lm_head
    inj = EmbedInjector(model, cfg["oracle"]["inject_alpha"])
    mean_t, W_t = Whitening.load(cfg["paths"]["whitening"]).to_torch(model.device)
    params = [p for n, p in model.named_parameters() if p.requires_grad and ".oracle." in n]
    log(f"[rl] trainable {sum(p.numel() for p in params) / 1e6:.1f}M params")
    opt = torch.optim.AdamW(params, lr=rcfg["lr"], betas=(0.9, 0.95), weight_decay=0.0)

    acts = np.load(os.path.join(cfg["paths"]["acts"], "rl.npy"), mmap_mode="r")
    eval_acts = np.load(os.path.join(cfg["paths"]["acts"], "eval.npy"), mmap_mode="r")
    log_path = os.path.join(cfg["paths"]["ckpt_oracle_rl"], "rl_log.jsonl")
    os.makedirs(cfg["paths"]["ckpt_oracle_rl"], exist_ok=True)
    P, G = rcfg["prompts_per_step"], rcfg["group_size"]

    def evaluate(n_rows):
        rows = list(range(min(n_rows, len(eval_acts))))
        N, K = rcfg["eval_n"], rcfg["eval_k"]
        fv = []
        for s in range(0, len(rows), 16):
            r = rows[s : s + 16]
            a = torch.tensor(np.stack([eval_acts[i] for i in r]))
            prompts = [(N, K, *build_oracle_prompt(tok, N, K)) for _ in r]
            gens = rollout(model, tok, inj, prompts, a, 0.0, K * (N + 1) + 16, 1)
            zs = whiten_unit(a.to(model.device), mean_t, W_t).cpu().numpy()
            _, f, _ = score(recon, tok, gens, zs, [N] * len(r), [K] * len(r), 0, 0, model.device)
            fv.extend(f.tolist())
        model.set_adapter("oracle")
        return float(np.mean(fv))

    log(f"[rl] SFT warm-start greedy FVE (N={rcfg['eval_n']},K={rcfg['eval_k']}): {evaluate(rcfg['eval_rows']):.4f}")
    t0 = time.time()
    for step in range(1, rcfg["steps"] + 1):
        rows = [rng.randrange(N_EVAL_RESERVED, len(acts)) for _ in range(P)]
        prompts = [sample_rl_prompt(tok, cfg, rng) for _ in range(P)]
        a = torch.tensor(np.stack([acts[r] for r in rows]))
        zs = whiten_unit(a.to(model.device), mean_t, W_t).cpu().numpy()
        Ns = [p[0] for p in prompts for _ in range(G)]
        Ks = [p[1] for p in prompts for _ in range(G)]
        max_new = max(K * (N + 1) for N, K, _, _ in prompts) + rcfg["max_new_tokens_slack"]
        gens = rollout(model, tok, inj, prompts, a, rcfg["temperature"], max_new, G)
        rewards, fves, parsed = score(
            recon,
            tok,
            gens,
            np.repeat(zs, G, axis=0),
            Ns,
            Ks,
            rcfg["format_penalty_per_phrase"],
            rcfg["format_penalty_per_token"],
            model.device,
        )
        r = rewards.reshape(P, G)
        adv = ((r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-4)).reshape(-1)
        model.set_adapter("oracle")

        # policy update over micro-batches of full sequences (prompt + completion)
        seqs, comp = [], []
        for i, p in enumerate(prompts):
            for j in range(G):
                g = gens[i * G + j]
                cut = len(g)
                for t, tid in enumerate(g):
                    if tid in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
                        cut = t + 1
                        break
                seqs.append(p[2] + g[:cut])
                comp.append([0] * len(p[2]) + [1] * cut)
        mb = rcfg["train_micro_batch"]
        total_tokens = sum(sum(c) for c in comp)
        kl_sum, n_mb = 0.0, 0
        for s in range(0, len(seqs), mb):
            chunk = list(range(s, min(s + mb, len(seqs))))
            ids, mask, offs = left_pad([seqs[i] for i in chunk], tok.pad_token_id)
            cmask = torch.zeros_like(ids)
            for k, i in enumerate(chunk):
                cmask[k, offs[k] :] = torch.tensor(comp[i])
            pos = torch.tensor([prompts[i // G][3] + offs[k] for k, i in enumerate(chunk)])
            vec = a[[i // G for i in chunk]]
            ids, mask, cmask = ids.to(model.device), mask.to(model.device), cmask.to(model.device).bool()
            lp = completion_logprobs(model, lm_head, ids, mask, cmask, inj, pos, vec)
            adv_t = torch.tensor(adv[chunk], device=model.device)[:, None]
            loss = -(adv_t * lp).sum() / total_tokens
            if rcfg["kl_beta"] > 0:
                with torch.no_grad():
                    model.set_adapter("sft_ref")
                    lp_ref = completion_logprobs(model, lm_head, ids, mask, cmask, inj, pos, vec)
                    model.set_adapter("oracle")
                d = lp_ref - lp
                kl = (torch.exp(d) - d - 1)[cmask[:, 1:]]  # k3 estimator per token
                loss = loss + rcfg["kl_beta"] * kl.sum() / total_tokens
                kl_sum += kl.mean().item()
                n_mb += 1
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        rec = {
            "step": step,
            "reward": float(rewards.mean()),
            "fve": float(fves.mean()),
            "n_phrases": float(np.mean([len(p) for p in parsed])),
            "len_err": float(np.mean([abs(len(q) - n) for p, n in zip(parsed, Ns) for q in p] or [0])),
            "kl": kl_sum / max(n_mb, 1),
            "s_per_step": (time.time() - t0) / step,
        }
        log(f"[rl] {json.dumps(rec)}")
        if step % rcfg["eval_every"] == 0:
            rec["eval_fve"] = evaluate(rcfg["eval_rows"])
            log(f"[rl] step {step} eval greedy FVE {rec['eval_fve']:.4f}")
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if step % rcfg["save_every"] == 0:
            save_oracle(model, cfg["paths"]["ckpt_oracle_rl"])
    save_oracle(model, cfg["paths"]["ckpt_oracle_rl"])
    log(f"[rl] final eval greedy FVE {evaluate(rcfg['eval_rows'] * 2):.4f}")


if __name__ == "__main__":
    main()
