---
base_model: Qwen/Qwen3.8-27B
library_name: peft
license: apache-2.0
tags: [interpretability, oracle-lens, activation-verbalization, lora]
---
# Oracle lens (SFT only, pre-RL) for Qwen3.8-27B, layer 42

Part of a reconstruction of the Anthropic "oracle lens" (Transformer Circuits, *Workspace*, 2026), here on
`Qwen/Qwen3.8-27B`, layer 42 (output of block 42 of 64 = HF `hidden_states[43]`). Same pilot-scale recipe and
config as the Qwen3.6-27B run (6k ultrachat transcripts, ~29k labelled activations, one SFT epoch, 40 REINFORCE
steps, one H100), so the two are directly comparable.
Code, configs and every design decision: https://github.com/robbiethompson18/oracle-lens-reconstruction

Held-out greedy FVE (N=8 tokens/phrase, K=10 phrases, 256 eval rows):

| checkpoint | Qwen3.8-27B (this) | Qwen3.6-27B (same recipe) |
|---|---|---|
| random phrases | 0.0015 | 0.0011 |
| oracle after SFT | 0.050 | 0.058 |
| oracle after RL | 0.186 | 0.236 |

All three are rsLoRA adapters (r=64, alpha=16) on every Linear of the text decoder; load with `peft` on top of the
base model. The oracle expects the lens activation injected by replacing the input embedding of the marker token
`㈜` (id 158983) with `16000 * a/|a|`; see `src/olens/common.py::EmbedInjector` and `oracle_data.py` for the prompt.
