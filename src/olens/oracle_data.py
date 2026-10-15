"""Oracle SFT example assembly, shared by SFT and RL.

An example: prompt = chat(user: "<activation>㈜</activation> ... exactly K phrases, each exactly N tokens")
            inject = raw lens-layer activation at the marker's embedding (alpha * unit)
            target = teacher's first K length-N phrases, exact token ids, newline-separated, EOS-terminated.
"""

import os

import numpy as np
import torch

from .common import build_oracle_prompt, newline_id, read_jsonl


class TeacherStore:
    def __init__(self, cfg):
        self.lengths = cfg["dictionary"]["lengths"]
        self.phrases = {
            L: [r["ids"] for r in read_jsonl(os.path.join(cfg["paths"]["dictionary"], f"len{L}_phrases.jsonl"))] for L in self.lengths
        }
        self.acts = np.load(os.path.join(cfg["paths"]["acts"], "teacher.npy"), mmap_mode="r")
        self.labels = {L: {} for L in self.lengths}
        for L in self.lengths:
            for r in read_jsonl(os.path.join(cfg["paths"]["teacher"], f"labels_len{L}.jsonl")):
                if r["atoms"]:
                    self.labels[L][r["act_row"]] = r

    def rows(self):
        rows = set(self.labels[self.lengths[0]])
        for L in self.lengths[1:]:
            rows &= set(self.labels[L])
        return sorted(rows)

    def activation(self, row):
        return np.asarray(self.acts[row], dtype=np.float32)


def build_sft_example(store, tok, cfg, row, rng):
    # DECISION (SFT targets): teacher atoms in OMP selection order (greedy order = importance order),
    # K ~ U{1..k_max}, L uniform over the five lengths; (L, K) are resampled every epoch (ceselder's July
    # code froze them per row).
    L = rng.choice(store.lengths)
    lab = store.labels[L].get(row)
    if not lab:
        return None
    k = rng.randint(1, min(cfg["oracle"]["k_max"], len(lab["atoms"])))
    prompt_ids, pos = build_oracle_prompt(tok, L, k)
    nl = newline_id(tok)
    target = []
    for ai in lab["atoms"][:k]:
        target += store.phrases[L][ai] + [nl]
    target[-1] = tok.convert_tokens_to_ids("<|im_end|>")
    return {"input_ids": prompt_ids + target, "labels": [-100] * len(prompt_ids) + target, "inject_pos": pos, "act": store.activation(row)}


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    labels = torch.full((B, maxlen), -100, dtype=torch.long)
    mask = torch.zeros((B, maxlen), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": mask,
        "inject_pos": torch.tensor([b["inject_pos"] for b in batch]),
        "acts": torch.tensor(np.stack([b["act"] for b in batch])),
    }


def chunked_ce(hidden, labels, lm_head, chunk=1024):
    """Token cross-entropy over [B,T,d] hidden states without materialising [B,T,248k] logits at once.
    Shifted inside: position t predicts labels[t+1]. Returns (sum_loss, n_tokens)."""
    import torch.utils.checkpoint as ckpt

    h = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    y = labels[:, 1:].reshape(-1)
    keep = y != -100
    h, y = h[keep], y[keep]
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, len(y), chunk):

        def f(hc, yc):
            return torch.nn.functional.cross_entropy(lm_head(hc).float(), yc, reduction="sum")

        total = total + ckpt.checkpoint(f, h[i : i + chunk], y[i : i + chunk], use_reentrant=False)
    return total, len(y)


def sample_rl_prompt(tok, cfg, rng):
    """(N, K, prompt_ids, marker_pos) for an RL rollout; biased toward the eval regime."""
    ocfg = cfg["oracle"]["rl"]
    N = rng.choice(ocfg["lengths"])
    K = rng.randint(1, ocfg["k_max"])
    ids, pos = build_oracle_prompt(tok, N, K)
    return N, K, ids, pos
