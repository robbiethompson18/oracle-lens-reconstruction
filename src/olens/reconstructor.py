"""Reconstructor: phrase -> unit whitened direction of the activation that precedes it.

Architecture: the shared base model with a LoRA adapter named "recon", early-exited at the lens layer,
plus an fp32 Linear(d, d) head on the last real token. Output is L2-normalised (whitened unit space).
"""

import os

import torch
import torch.nn.functional as F
from torch import nn

from .common import EarlyExit, lora_config

# DECISION (reconstructor input template): the paper gives none. We frame the phrase like ceselder's AR
# ("Summary of the following text: <text>...</text> <summary>") and read the last token, so the model
# has a fixed anchor position after the phrase to summarise into.
RECON_PREFIX = "Summary of the following text: <text>"
RECON_SUFFIX = "</text> <summary>"


class ReconHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=True)
        nn.init.zeros_(self.lin.bias)
        nn.init.eye_(self.lin.weight)  # identity init (ceselder): start from "the hidden state itself"
        self.norm = nn.LayerNorm(d)

    def forward(self, h):
        return F.normalize(self.lin(self.norm(h.float())), dim=-1)


class Reconstructor:
    """Wraps a (peft) model that has a "recon" adapter, an EarlyExit at `layer`, and a head."""

    def __init__(self, model, layer, head, adapter="recon"):
        self.model = model
        self.exit = EarlyExit(model, layer)
        self.head = head
        self.adapter = adapter

    def embed(self, input_ids, attention_mask):
        """[B,T] right-padded -> [B,d] unit whitened directions (fp32), with grad if enabled."""
        self.model.set_adapter(self.adapter)
        h = self.exit.run(self.model, input_ids=input_ids, attention_mask=attention_mask)
        last = attention_mask.sum(-1) - 1
        h_last = h[torch.arange(h.shape[0], device=h.device), last]
        return self.head(h_last)

    @torch.inference_mode()
    def embed_phrases(self, tok, phrase_ids_list, device, batch_size=256):
        outs = []
        for i in range(0, len(phrase_ids_list), batch_size):
            ii, mm = build_inputs(tok, phrase_ids_list[i : i + batch_size], device)
            outs.append(self.embed(ii, mm).float().cpu())
        return torch.cat(outs)


_prefix_cache = {}


def build_inputs(tok, phrase_ids_list, device):
    """Token-space inputs: prefix_ids + phrase token ids + suffix_ids, right padded."""
    if tok.name_or_path not in _prefix_cache:
        _prefix_cache[tok.name_or_path] = (
            tok(RECON_PREFIX, add_special_tokens=False)["input_ids"],
            tok(RECON_SUFFIX, add_special_tokens=False)["input_ids"],
        )
    pre, suf = _prefix_cache[tok.name_or_path]
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []  # Gemma needs BOS; Qwen has none
    rows = [bos + pre + list(p) + suf for p in phrase_ids_list]
    maxlen = max(len(r) for r in rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(rows), maxlen), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), maxlen), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    return ids.to(device), mask.to(device)


def add_recon_adapter(model, cfg, trainable):
    """Attach the "recon" LoRA adapter (new or from disk) to `model`; returns a Reconstructor."""
    from peft import PeftModel, get_peft_model

    rcfg = cfg["reconstructor"]
    ckpt = cfg["paths"]["ckpt_reconstructor"]
    d = cfg["model"]["hidden_size"]
    head = ReconHead(d).to(model.device)
    if os.path.exists(os.path.join(ckpt, "head.pt")):
        head.load_state_dict(torch.load(os.path.join(ckpt, "head.pt"), map_location=model.device))
        if hasattr(model, "peft_config"):
            model.load_adapter(ckpt, adapter_name="recon", is_trainable=trainable)
        else:
            model = PeftModel.from_pretrained(model, ckpt, adapter_name="recon", is_trainable=trainable)
    else:
        assert trainable, f"no reconstructor checkpoint at {ckpt}"
        lc = lora_config(rcfg["lora_r"], rcfg["lora_alpha"])
        model = (
            model.add_adapter("recon", lc) or model if hasattr(model, "peft_config") else get_peft_model(model, lc, adapter_name="recon")
        )
    if not trainable:
        head.requires_grad_(False)
    return model, Reconstructor(model, cfg["model"]["layer"], head)


def save_recon(model, head, ckpt):
    os.makedirs(ckpt, exist_ok=True)
    model.save_pretrained(ckpt, selected_adapters=["recon"])
    # peft writes the adapter under ckpt/recon when the name isn't "default"; flatten for load_adapter
    sub = os.path.join(ckpt, "recon")
    if os.path.isdir(sub):
        for f in os.listdir(sub):
            os.replace(os.path.join(sub, f), os.path.join(ckpt, f))
        os.rmdir(sub)
    torch.save(head.state_dict(), os.path.join(ckpt, "head.pt"))
