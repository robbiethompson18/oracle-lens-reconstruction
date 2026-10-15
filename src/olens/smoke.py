"""Smoke test on the pod before spending hours: model loads with no random weights, marker is one token,
residual tap at the lens layer returns sane norms, embedding injection changes the output, early exit works,
chat template with thinking off renders as expected, kernels are the fast ones."""

import argparse
import time

import torch

from .common import MARKER_ID, EarlyExit, EmbedInjector, ResidualTap, build_oracle_prompt, load_config, load_model, log, text_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    model, tok = load_model(cfg["model"]["name"])
    log(f"loaded in {time.time() - t0:.0f}s; {torch.cuda.memory_allocated() / 1e9:.1f} GB; class {type(model).__name__}")
    tm = text_model(model)
    log(f"text decoder: {len(tm.layers)} blocks, types[40:46]={list(tm.config.layer_types)[40:46]}")
    try:
        import causal_conv1d  # noqa: F401
        import fla  # noqa: F401

        log("kernels: fla + causal_conv1d importable")
    except Exception as e:  # noqa: BLE001
        log(f"kernels: MISSING ({e}) -> slow PyTorch fallback")

    ids, pos = build_oracle_prompt(tok, 4, 10)
    log(f"oracle prompt: {len(ids)} tokens, marker at {pos}; tail={tok.decode(ids[pos - 3 : pos + 4])!r} ... {tok.decode(ids[-12:])!r}")
    assert ids[pos] == MARKER_ID

    chat = tok.apply_chat_template(
        [{"role": "user", "content": "Say hi"}], add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    log(f"chat render (thinking off): {chat[-80:]!r}")

    x = torch.tensor([tok("The capital of France is", add_special_tokens=False)["input_ids"]], device=model.device)
    tap = ResidualTap(model, cfg["model"]["layer"])
    with torch.inference_mode():
        out = model(input_ids=x, output_hidden_states=True, use_cache=False)
    h = tap.acts[0].float()
    hs = out.hidden_states[cfg["model"]["layer"] + 1][0].float()
    log(f"tap norm per token {h.norm(dim=-1).tolist()}; matches hidden_states[L+1]: {torch.allclose(h, hs)}; max|x| {h.abs().max():.0f}")
    tap.remove()
    top = out.logits[0, -1].topk(3).indices.tolist()
    log(f"next tokens: {[tok.decode([t]) for t in top]}")

    ex = EarlyExit(model, cfg["model"]["layer"])
    t1 = time.time()
    with torch.inference_mode():
        hh = ex.run(model, input_ids=x)
    log(f"early exit ok: {torch.allclose(hh[0].float(), h)} in {time.time() - t1:.2f}s")
    ex.remove()

    inj = EmbedInjector(model, cfg["oracle"]["inject_alpha"])
    pid = torch.tensor([ids], device=model.device)
    with torch.inference_mode():
        base = model(input_ids=pid, use_cache=False).logits[0, -1].float()
        inj.set(torch.tensor([pos]), h[-1:].cpu())
        injd = model(input_ids=pid, use_cache=False).logits[0, -1].float()
        inj.clear()
    log(f"injection changes final logits: max diff {(base - injd).abs().max():.2f} (0 would mean the hook did nothing)")
    emb = tm.embed_tokens.weight.float()
    log(f"embedding row norms: mean {emb.norm(dim=-1).mean():.2f}, marker {emb[MARKER_ID].norm():.2f}")
    log(f"inject alpha {cfg['oracle']['inject_alpha']}")

    t1 = time.time()
    with torch.inference_mode():
        g = model.generate(input_ids=pid, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
    log(f"generate 32 tokens: {time.time() - t1:.1f}s -> {tok.decode(g[0, pid.shape[1] :])!r}")
    log("SMOKE_OK")


if __name__ == "__main__":
    main()
