"""Qualitative readouts on WildChat prompts: let Qwen3.6-27B answer, then read the residual stream along the
response with an oracle lens, at several layers. Output feeds the dashboard (docs/dashboard).

Default lens = the WorkspaceBench authors' public checkpoint (`agu18dec/olens_and_ar`, `olens_s3d_rl600`), whose
read contract is: prompt `concepts_raw` with the layer number in the text, replace the marker embedding with
16000*unit(h), layers 20..60 step 4, sample T=1 top-p 0.95 top-k 64. Our own oracle (stage 5/6) uses a different
prompt (N phrases of K tokens), so `--ckpt ours` switches the prompt builder; everything else is shared.
"""

import argparse
import json
import os
import random

import torch

from .common import MARKER, MARKER_ID, EmbedInjector, ResidualTap, build_oracle_prompt, chat_ids, load_model, log, text_model

CONCEPTS_RAW = (
    "An activation vector from layer {layer} of a language model is enclosed in activation tags: "
    "<activation>{marker}</activation>. Produce distinct concepts that encode this activation, each as a '- ' bullet on its own line."
)


def wildchat_prompts(n, seed, min_chars=40, max_chars=700):
    """First user turn of English, non-toxic WildChat-1M conversations; exact dedup; fixed-seed sample."""
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True).shuffle(seed=seed, buffer_size=20000)
    seen, out = set(), []
    for row in ds:
        if row.get("language") != "English" or row.get("toxic") or row.get("redacted"):
            continue
        text = row["conversation"][0]["content"].strip()
        if not (min_chars <= len(text) <= max_chars) or text in seen:
            continue
        seen.add(text)
        out.append({"id": row["conversation_hash"], "prompt": text})
        if len(out) >= n:
            break
    return out


def left_pad(rows, pad_id):
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), L), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, L - len(r) :] = torch.tensor(r)
        mask[i, L - len(r) :] = 1
    return ids, mask, [L - len(r) for r in rows]


def generate_responses(model, tok, prompts, max_new, temperature, batch):
    out = []
    for s in range(0, len(prompts), batch):
        rows = [chat_ids(tok, p["prompt"]) for p in prompts[s : s + batch]]
        ids, mask, _ = left_pad(rows, tok.pad_token_id)
        with torch.inference_mode(), model.disable_adapter():
            gen = model.generate(
                input_ids=ids.to(model.device),
                attention_mask=mask.to(model.device),
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                max_new_tokens=max_new,
                pad_token_id=tok.pad_token_id,
            )
        stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
        for r, g in zip(rows, gen[:, ids.shape[1] :].tolist()):
            resp = []
            for t in g:
                if t in stop:
                    break
                resp.append(t)
            out.append((r, resp))
        log(f"[gen] {len(out)}/{len(prompts)}")
    return out


def capture(model, prompt_ids, resp_ids, layers, positions):
    """One forward pass (adapter off); residual at the output of each block in `layers` at `positions`."""
    taps = {L: ResidualTap(model, L) for L in layers}
    ids = torch.tensor([prompt_ids + resp_ids], device=model.device)
    with torch.inference_mode(), model.disable_adapter():
        model(input_ids=ids, use_cache=False)
    acts = {L: taps[L].acts[0, positions].float().cpu() for L in layers}
    for t in taps.values():
        t.remove()
    return acts


def build_prompt(tok, kind, layer, n, k):
    if kind == "ours":
        return build_oracle_prompt(tok, n, k)
    ids = chat_ids(tok, CONCEPTS_RAW.format(layer=layer, marker=MARKER))
    hits = [i for i, t in enumerate(ids) if t == MARKER_ID]
    assert len(hits) == 1, hits
    return ids, hits[0]


def read_cells(model, tok, inj, cells, kind, n, k, max_new, batch):
    """cells: list of (layer, act[d]). Returns decoded readout strings in the same order."""
    outs = []
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    for s in range(0, len(cells), batch):
        chunk = cells[s : s + batch]
        built = [build_prompt(tok, kind, L, n, k) for L, _ in chunk]
        ids, mask, offs = left_pad([b[0] for b in built], tok.pad_token_id)
        pos = torch.tensor([b[1] + o for b, o in zip(built, offs)])
        inj.set(pos, torch.stack([a for _, a in chunk]))
        with torch.inference_mode():
            gen = model.generate(
                input_ids=ids.to(model.device),
                attention_mask=mask.to(model.device),
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                max_new_tokens=max_new,
                pad_token_id=tok.pad_token_id,
            )
        inj.clear()
        for g in gen[:, ids.shape[1] :].tolist():
            keep = []
            for t in g:
                if t in stop:
                    break
                keep.append(t)
            outs.append(tok.decode(keep).strip())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--ckpt", default="agu18dec/olens_and_ar:olens_s3d_rl600", help="HF repo:subfolder, local dir, or ours:<dir>")
    ap.add_argument("--layers", default="36,44,52,60")
    ap.add_argument("--n-prompts", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resp-tokens", type=int, default=120, help="response tokens to generate and read along")
    ap.add_argument("--stride", type=int, default=3, help="read every stride-th response token")
    ap.add_argument("--resp-temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n", type=int, default=4, help="phrase length, 'ours' prompt only")
    ap.add_argument("--k", type=int, default=8, help="phrase count, 'ours' prompt only")
    ap.add_argument("--out", default="/work/wildchat/readouts.jsonl")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    from peft import PeftModel

    kind = "ours" if args.ckpt.startswith("ours:") else "wsbench"
    ckpt = args.ckpt.split(":", 1)[1] if kind == "ours" else args.ckpt
    if ":" in ckpt and not os.path.isdir(ckpt):
        from huggingface_hub import snapshot_download

        repo, sub = ckpt.split(":", 1)
        ckpt = os.path.join(snapshot_download(repo, allow_patterns=[f"{sub}/*"]), sub)
    model, tok = load_model(args.model)
    model = PeftModel.from_pretrained(model, ckpt, adapter_name="oracle")
    n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    assert n_lora > 0, "adapter matched no modules"
    log(f"[lens] {ckpt}: {n_lora} LoRA modules")
    model.eval()
    inj = EmbedInjector(model, 16000.0)
    assert text_model(model).embed_tokens is not None

    prompts = wildchat_prompts(args.n_prompts, args.seed)
    log(f"[wildchat] {len(prompts)} prompts")
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done = {json.loads(line)["id"] for line in fh if line.strip()}
    todo = [p for p in prompts if p["id"] not in done]
    log(f"[resume] {len(done)} done, {len(todo)} to do")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    random.seed(args.seed)
    gens = generate_responses(model, tok, todo, args.resp_tokens, args.resp_temperature, 16)
    with open(args.out, "a") as fh:
        for p, (pids, rids) in zip(todo, gens):
            # positions in the concatenated sequence: last prompt token (index -1 in response coords) and every
            # stride-th response token (reading at response token j = residual after the model has seen it)
            resp_pos = list(range(0, len(rids), args.stride))
            seq_pos = [len(pids) - 1] + [len(pids) + j for j in resp_pos]
            acts = capture(model, pids, rids, layers, seq_pos)
            cells = [(L, acts[L][i]) for i, _ in enumerate(seq_pos) for L in layers]
            texts = read_cells(model, tok, inj, cells, kind, args.n, args.k, args.max_new, args.batch)
            row = {
                "id": p["id"],
                "prompt": p["prompt"],
                "response_tokens": [tok.decode([t]) for t in rids],
                "layers": layers,
                "positions": [-1] + resp_pos,
                "readouts": [texts[i * len(layers) : (i + 1) * len(layers)] for i in range(len(seq_pos))],
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            log(f"[read] {p['id'][:8]} {len(cells)} cells; e.g. L{layers[1]}@-1: {row['readouts'][0][1][:80]!r}")


if __name__ == "__main__":
    main()
