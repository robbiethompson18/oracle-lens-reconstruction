# Code smells and debt

- 2026-10-15: `causal_conv1d` cannot be built on the RunPod image (no system nvcc; the pip-packaged
  CUDA 13 toolchain ships mismatched cicc/ptxas versions, and no prebuilt wheel exists for torch 2.13).
  transformers falls back to a PyTorch conv for the Gated-DeltaNet layers. Training is fine; token-by-token
  decode (RL rollouts, eval) runs at ~0.4 s/token single-stream. Fix: a pod image with nvcc, or vLLM for
  rollouts.
- 2026-10-15: `FLA_TILELANG=0` is required (set in `sky/run_pilot.sh`); flash-linear-attention's default
  backward backend compiles CUDA at runtime and dies without nvcc. The Triton backend works.
- 2026-10-15: `sky/run_pilot.sh` is a bash stage runner with env vars inline; a `--stage` CLI in Python
  would be nicer once the stages settle.
- 2026-10-15: pilot dictionary split had 29k positions, fewer than `dictionary.n_starts` (40k); the config
  should derive split sizes from one scale knob instead of five hand-set numbers.
