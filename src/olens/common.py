"""Shared pieces: config, model loading, hooks, whitening, prompt format, NNLS/FVE.

Layer convention used everywhere in this repo: "layer L" = the residual stream at the OUTPUT of
decoder block L (HF `hidden_states[L+1]`). WorkspaceBench and ceselder's Aug/Sep work use the same
convention. ceselder's July code hooked the INPUT of block 45 (= output of block 44).
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

# DECISION: injection marker. A single existing vocab token that survives as one token inside the chat
# render. ㈜ (U+321C, id 158983) is what both ceselder's olens and WorkspaceBench's lens use for Qwen3.6.
MARKER = "㈜"
MARKER_ID = 158983


# ---------------------------------------------------------------- config


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:  # "a.b.c=value"
        key, val = ov.split("=", 1)
        cur = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        val = yaml.safe_load(val)
        if isinstance(val, str) and re.fullmatch(r"[+-]?\d+(\.\d*)?[eE][+-]?\d+", val):
            val = float(val)
        cur[parts[-1]] = val
    root = cfg["paths"]["root"]
    for k, v in cfg["paths"].items():
        if k != "root" and not os.path.isabs(v):
            cfg["paths"][k] = os.path.join(root, v)
    return cfg


# ---------------------------------------------------------------- model loading


def load_model(name, dtype=torch.bfloat16, device="cuda", grad_checkpointing=False):
    """Load Qwen3.6-27B's text decoder.

    DECISION: AutoModelForCausalLM resolves the multimodal checkpoint to the text-only Qwen3_5ForCausalLM
    (transformers 5.17 remaps the `model.language_model.*` keys; verified on the pod: 53.8 GB, no missing
    keys). We assert nothing was randomly initialised, because a silent key mismatch would give a random
    model that still "runs". `text_model()` below copes with either class layout.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model, info = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, device_map={"": device}, attn_implementation="sdpa", output_loading_info=True
    )
    missing = [k for k in info.get("missing_keys", []) if not k.startswith("mtp")]
    assert not missing, f"randomly initialised weights: {missing[:5]} (+{len(missing) - 5})"
    model.config.use_cache = not grad_checkpointing
    if grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, tok


def unwrap_peft(model):
    if hasattr(model, "peft_config") and hasattr(model, "base_model"):
        return model.base_model.model
    return model


def text_model(model):
    """The Qwen3_5TextModel (owns .embed_tokens, .layers, .norm), through wrapper/peft layers."""
    model = unwrap_peft(model)
    for chain in (("model", "language_model"), ("language_model",), ("model",)):
        obj = model
        for a in chain:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "layers") and hasattr(obj, "embed_tokens"):
            return obj
    raise AttributeError("could not find the text decoder")


class StopForward(Exception):
    pass


class EarlyExit:
    """Capture the output of block `layer` and abort the forward pass there.

    DECISION (reconstructor backbone): the paper fine-tunes a full copy of the subject model as the
    reconstructor. We instead keep ONE base model with two LoRA adapters ("recon" and "oracle") and run
    the reconstructor by early-exiting at the lens layer: blocks above it never see the activation being
    predicted, WorkspaceBench's ablation found reconstructor size barely matters after RL, and sharing the
    base is what lets the RL stage fit on one 80 GB GPU (no second 54 GB copy).
    """

    def __init__(self, model, layer):
        self.acts = None
        self.active = False  # the hook stays registered; only forwards inside run() exit early
        self._h = text_model(model).layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if not self.active:
            return  # an oracle generate() on the shared base must run all 64 blocks
        self.acts = _out_tensor(output)
        raise StopForward

    def run(self, model, **kwargs):
        self.acts = None
        self.active = True
        try:
            model(**kwargs, use_cache=False)
        except StopForward:
            pass
        finally:
            self.active = False
        return self.acts

    def remove(self):
        self._h.remove()


# All Linear modules of the text decoder (attention, Gated-DeltaNet projections, MLP). The vision tower
# uses different names (qkv, proj, linear_fc1/2) so this list never touches it.
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def lora_config(r, alpha, dropout=0.0):
    from peft import LoraConfig

    # DECISION: rsLoRA r64/α16 on all decoder linears, ceselder's setting for every Qwen3.6 lens.
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS, use_rslora=True, bias="none")


# ---------------------------------------------------------------- hooks


def _out_tensor(output):
    return output[0] if isinstance(output, tuple) else output


class ResidualTap:
    """Capture the residual stream at the OUTPUT of decoder block `layer`."""

    def __init__(self, model, layer):
        self.acts = None
        self._h = text_model(model).layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        self.acts = _out_tensor(output).detach()

    def remove(self):
        self._h.remove()


class EmbedInjector:
    """Replace the marker token's input embedding with alpha * unit(activation).

    DECISION (injection): the paper says only "injected into the oracle's residual stream". Three
    schemes exist in prior work: overwrite the residual at the lens layer (ceselder July), add a
    norm-matched vector at block 1 (ceselder Aug), replace the marker's input embedding with
    16000*unit(h) (WorkspaceBench). We use the last: WorkspaceBench's ablation found raw h beats
    centred/whitened and it needs no custom vLLM kernel for rollouts (stock prompt-embeds).
    Fires only when the sequence still contains the marker (prefill); no-op on cached decode steps.
    """

    def __init__(self, model, alpha):
        self.alpha = alpha
        self.positions = None
        self.vectors = None
        self._h = text_model(model).embed_tokens.register_forward_hook(self._hook)

    def set(self, positions, vectors):
        self.positions = positions
        self.vectors = vectors

    def clear(self):
        self.positions = None
        self.vectors = None

    def _hook(self, module, args, output):
        if self.positions is None or output.shape[1] <= int(self.positions.max()):
            return None
        v = self.vectors.to(output.device, torch.float32)
        v = self.alpha * v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        out = output.clone()
        out[torch.arange(out.shape[0], device=out.device), self.positions.to(out.device)] = v.to(out.dtype)
        return out

    def remove(self):
        self._h.remove()


# ---------------------------------------------------------------- whitening


@dataclass
class Whitening:
    mean: np.ndarray  # [d]
    W: np.ndarray  # (Σ + λI)^{-1/2}, [d, d], symmetric

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(path, "mean.npy"), self.mean)
        np.save(os.path.join(path, "W.npy"), self.W)

    @classmethod
    def load(cls, path):
        return cls(mean=np.load(os.path.join(path, "mean.npy")), W=np.load(os.path.join(path, "W.npy")))

    @classmethod
    def fit(cls, acts, ridge_frac):
        """DECISION (whitening): full-covariance (Σ+λI)^{-1/2} with λ = ridge_frac * mean eigenvalue,
        fit on raw lens-layer activations of the reconstructor-training split. The paper whitens with the
        template-lens covariance (Claude-written passages); we have no template corpus, so on-policy
        activations stand in. ceselder used three incompatible whiteners over time; ours matches her July."""
        acts = acts.astype(np.float64)
        mean = acts.mean(0)
        x = acts - mean
        cov = (x.T @ x) / (len(x) - 1)
        lam = ridge_frac * float(np.trace(cov)) / cov.shape[0]
        evals, evecs = np.linalg.eigh(cov + lam * np.eye(cov.shape[0]))
        W = (evecs * (evals**-0.5)) @ evecs.T
        return cls(mean=mean.astype(np.float32), W=W.astype(np.float32))

    def to_torch(self, device):
        return torch.tensor(self.mean, device=device), torch.tensor(self.W, device=device)


def whiten_unit(acts, mean_t, W_t):
    """[n,d] raw -> [n,d] unit-norm whitened."""
    z = (acts.float() - mean_t) @ W_t
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)


# ---------------------------------------------------------------- oracle prompt

# DECISION (prompt): ceselder's July wording, with the marker wrapped in <activation> tags as in
# WorkspaceBench. N and K are stated in the prompt (paper). Thinking is disabled so the phrase list follows
# the assistant header directly; ceselder's July prompt accidentally left an open <think> block.
ORACLE_INSTRUCTION = (
    "You are an activation decoder. An internal activation vector from a language model is enclosed in "
    "activation tags: <activation>{marker}</activation>. Output exactly {k} phrases, each exactly {n} tokens "
    "long, that together describe the content of this activation. One phrase per line, no numbering, no extra text."
)


def chat_ids(tok, user_text):
    msgs = [{"role": "user", "content": user_text}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False)
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    if isinstance(out, torch.Tensor):
        out = out.tolist()
    if out and isinstance(out[0], (list, tuple)):
        out = list(out[0])
    return list(out)


def build_oracle_prompt(tok, n, k):
    """Returns (input_ids, marker_pos)."""
    ids = chat_ids(tok, ORACLE_INSTRUCTION.format(marker=MARKER, k=k, n=n))
    hits = [i for i, t in enumerate(ids) if t == MARKER_ID]
    assert len(hits) == 1, f"marker must appear exactly once as token {MARKER_ID}; got {hits}"
    return ids, hits[0]


def newline_id(tok):
    ids = tok("\n", add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, ids
    return ids[0]


def parse_generated_phrases(gen_ids, tok, k):
    """Split generated ids on the newline token; drop empties; stop at eos/im_end; cap at k."""
    nl = newline_id(tok)
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    phrases, cur = [], []
    for t in gen_ids:
        t = int(t)
        if t in stop:
            break
        if t == nl:
            if cur:
                phrases.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        phrases.append(cur)
    return phrases[:k]


# ---------------------------------------------------------------- NNLS / FVE


def nnls_coeffs(D, x):
    """Exact non-negative least squares (scipy). D [m,d], x [d] -> c [m] >= 0."""
    from scipy.optimize import nnls

    c, _ = nnls(np.asarray(D, dtype=np.float64).T, np.asarray(x, dtype=np.float64))
    return c


def fve_of(D, x):
    """Fraction of variance of x explained by a non-negative combination of the rows of D.
    Both in whitened space; x is unit-norm. Returns (fve, coeffs)."""
    if len(D) == 0:
        return 0.0, np.zeros(0)
    D = np.asarray(D, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    c = nnls_coeffs(D, x).astype(np.float32)
    resid = x - c @ D
    return float(1.0 - (resid @ resid) / (x @ x)), c


# ---------------------------------------------------------------- misc


def jsonl_writer(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", buffering=1)


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def log(msg):
    print(msg, flush=True)
