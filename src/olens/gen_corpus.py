"""Stage 0: on-policy assistant-turn corpus from Qwen3.6-27B via vLLM.

Writes {corpus}/transcripts.jsonl rows {idx, split, prompt, prompt_ids, response_ids}.
"""

import argparse
import json
import os
import random

from .common import load_config, log


def load_prompts(cfg, tok):
    """DECISION (prompt source): the paper never specifies its prompt distribution. WorkspaceBench used
    WildChat-1M (gated) and found corpus choice made no difference after RL; we use ultrachat_200k, which
    is ungated, first user turn only, English-ish length filter, exact dedup."""
    from datasets import load_dataset

    ccfg = cfg["corpus"]
    ds = load_dataset(ccfg["prompt_dataset"], split=ccfg["prompt_split"], streaming=True)
    lang = ccfg.get("language")  # WildChat's per-conversation `language` label, e.g. "Chinese"; None = keep all
    seen, out = set(), []
    for row in ds:
        if row.get("toxic") or (lang and row.get("language") != lang):
            continue
        msgs = row.get("messages") or row.get("conversation") or []
        if not msgs:
            continue
        text = msgs[0].get("content", "")
        if not (10 <= len(text) <= 2000) or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= ccfg["n_transcripts"] * 2:
            break
    random.Random(0).shuffle(out)
    return out[: ccfg["n_transcripts"]]


def assign_splits(n, splits, seed=0):
    names, fracs = zip(*splits.items())
    rng = random.Random(seed)
    labels = []
    for _ in range(n):
        r = rng.random()
        acc = 0.0
        lab = names[-1]
        for nm, fr in zip(names, fracs):
            acc += fr
            if r < acc:
                lab = nm
                break
        labels.append(lab)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    ccfg = cfg["corpus"]
    out_path = os.path.join(cfg["paths"]["corpus"], "transcripts.jsonl")
    if os.path.exists(out_path):
        log(f"[corpus] {out_path} exists, skipping")
        return
    os.makedirs(cfg["paths"]["corpus"], exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    prompts = load_prompts(cfg, tok)
    splits = assign_splits(len(prompts), ccfg["splits"])
    log(f"[corpus] {len(prompts)} prompts")

    # DECISION (thinking): generate with enable_thinking=False. ceselder's July corpus had thinking ON,
    # so chain-of-thought tokens counted as assistant text; WorkspaceBench renders chats with thinking
    # off, and the lens is meant to read the model during an ordinary reply.
    prompt_ids = [
        tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True, enable_thinking=False)
        for p in prompts
    ]
    prompt_ids = [list(p["input_ids"]) if hasattr(p, "keys") else list(p) for p in prompt_ids]
    # drop (never truncate) over-long prompts: truncating a rendered chat would cut off the assistant header
    keep = [i for i, p in enumerate(prompt_ids) if len(p) <= ccfg["max_prompt_tokens"]]
    prompts, prompt_ids, splits = [prompts[i] for i in keep], [prompt_ids[i] for i in keep], [splits[i] for i in keep]
    log(f"[corpus] {len(keep)} prompts within {ccfg['max_prompt_tokens']} tokens")

    kwargs = {
        "model": cfg["model"]["name"],
        "dtype": "bfloat16",
        "max_model_len": ccfg["max_prompt_tokens"] + ccfg["max_new_tokens"] + 8,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 256,  # one Mamba cache block per decode sequence; 80 GB fits ~326
        "enable_prefix_caching": False,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
    }
    try:
        llm = LLM(language_model_only=True, **kwargs)
    except TypeError:
        llm = LLM(**kwargs)
    sp = SamplingParams(temperature=ccfg["temperature"], top_p=ccfg["top_p"], max_tokens=ccfg["max_new_tokens"], seed=0)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompt_ids], sp)
    n = 0
    with open(out_path, "w") as f:
        for i, (p, o) in enumerate(zip(prompt_ids, outs)):
            resp = list(o.outputs[0].token_ids)
            if len(resp) < 8:
                continue
            f.write(json.dumps({"idx": i, "split": splits[i], "prompt": prompts[i], "prompt_ids": p, "response_ids": resp}) + "\n")
            n += 1
    log(f"[corpus] wrote {n} transcripts to {out_path}")


if __name__ == "__main__":
    main()
