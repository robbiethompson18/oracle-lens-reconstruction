"""Stage 4: teacher labels by batched non-negative OMP against the dictionary (GPU, no model).

For each `teacher` activation (unit whitened) and each phrase length L: restrict to a random
`dict_fraction` of the length-L dictionary (so the oracle cannot memorise a fixed ranking), greedily pick the
atom with the largest positive correlation with the residual, NNLS-refit all picked atoms, up to
`max_atoms`. Output {teacher}/labels_len{L}.jsonl rows {act_row, atoms, coeffs, fve}.
"""

import argparse
import json
import os

import numpy as np
import torch

from .common import Whitening, load_config, log, whiten_unit


def batched_nnls(S, x, iters=60):
    """S [B,k,d], x [B,d] -> c [B,k] >= 0 by projected gradient with a Lipschitz step."""
    G = torch.bmm(S, S.transpose(1, 2))
    b = torch.bmm(S, x.unsqueeze(-1)).squeeze(-1)
    lip = torch.linalg.eigvalsh(G)[:, -1].clamp_min(1e-6).unsqueeze(-1)
    c = torch.clamp(b, min=0) / lip
    for _ in range(iters):
        c = torch.clamp(c + (b - torch.bmm(G, c.unsqueeze(-1)).squeeze(-1)) / lip, min=0)
    return c


def omp_batch(D, X, max_atoms, dict_fraction, gen):
    """D [m,d] (fp16 GPU, unit rows), X [B,d] fp32 unit. Returns atoms [B,K] (-1 pad), coeffs, fve prefix."""
    B, d = X.shape
    m = D.shape[0]
    mask = torch.rand(B, m, device=X.device, generator=gen) < dict_fraction
    R = X.clone()
    atoms = torch.full((B, max_atoms), -1, dtype=torch.long, device=X.device)
    coeffs = torch.zeros(B, max_atoms, device=X.device)
    fves = torch.zeros(B, max_atoms, device=X.device)
    alive = torch.ones(B, dtype=torch.bool, device=X.device)
    S = torch.zeros(B, max_atoms, d, device=X.device)
    x2 = (X**2).sum(-1)
    for k in range(max_atoms):
        corr = (R.to(D.dtype) @ D.T).float()
        corr[~mask] = -1e30
        best_val, best_idx = corr.max(-1)
        alive = alive & (best_val > 1e-4)
        if not alive.any():
            break
        sel = torch.where(alive, best_idx, torch.zeros_like(best_idx))
        S[:, k] = D[sel].float()
        S[~alive, k] = 0.0
        atoms[alive, k] = best_idx[alive]
        mask[torch.arange(B, device=X.device), sel] = False
        c = batched_nnls(S[:, : k + 1], X)
        coeffs[:, : k + 1] = torch.where(alive.unsqueeze(-1), c, coeffs[:, : k + 1])
        R = torch.where(alive.unsqueeze(-1), X - torch.bmm(c.unsqueeze(1), S[:, : k + 1]).squeeze(1), R)
        fves[:, k] = torch.where(alive, 1 - (R**2).sum(-1) / x2, fves[:, k - 1] if k else torch.zeros_like(x2))
    return atoms.cpu(), coeffs.cpu(), fves.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    tcfg = cfg["teacher"]
    dev = "cuda"
    mean_t, W_t = Whitening.load(cfg["paths"]["whitening"]).to_torch(dev)
    acts = np.load(os.path.join(cfg["paths"]["acts"], "teacher.npy"), mmap_mode="r")
    n = min(len(acts), tcfg["n_activations"])
    os.makedirs(cfg["paths"]["teacher"], exist_ok=True)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234)
    bs = tcfg["omp_batch_size"]
    for L in cfg["dictionary"]["lengths"]:
        out_path = os.path.join(cfg["paths"]["teacher"], f"labels_len{L}.jsonl")
        if os.path.exists(out_path + ".done"):
            log(f"[teacher] len{L} done, skipping")
            continue
        D = torch.tensor(np.load(os.path.join(cfg["paths"]["dictionary"], f"len{L}_dirs.npy")), device=dev)
        fve_sum = 0.0
        with open(out_path, "w") as out:
            for i in range(0, n, bs):
                X = whiten_unit(torch.tensor(np.asarray(acts[i : i + bs]), device=dev), mean_t, W_t)
                atoms, coeffs, fves = omp_batch(D, X, tcfg["max_atoms"], tcfg["dict_fraction"], gen)
                for j in range(len(X)):
                    valid = (atoms[j] >= 0).nonzero().squeeze(-1)
                    out.write(
                        json.dumps(
                            {
                                "act_row": i + j,
                                "atoms": atoms[j][valid].tolist(),
                                "coeffs": [round(float(c), 5) for c in coeffs[j][valid]],
                                "fve": [round(float(f), 5) for f in fves[j][valid]],
                            }
                        )
                        + "\n"
                    )
                fve_sum += fves[:, -1].sum().item()
                if (i // bs) % 10 == 0:
                    log(f"[teacher] len{L} {i}/{n} fve{tcfg['max_atoms']}={fves[:, -1].mean().item():.3f}")
        open(out_path + ".done", "w").close()
        log(f"[teacher] len{L} mean fve{tcfg['max_atoms']} = {fve_sum / n:.4f}")
        del D
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
