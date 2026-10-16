---
base_model: Qwen/Qwen3.6-27B
library_name: peft
license: apache-2.0
tags: [interpretability, oracle-lens, activation-verbalization, lora]
---
# Oracle lens (SFT only, pre-RL) for Qwen3.6-27B, layer 42

Part of a reconstruction of the Anthropic "oracle lens" (Transformer Circuits, *Workspace*, 2026) on
`Qwen/Qwen3.6-27B`, layer 42 (output of block 42 = HF `hidden_states[43]`). Pilot-scale run on one H100:
~6k ultrachat transcripts, ~29k labelled activations, one SFT epoch, 40 REINFORCE steps.
Code, configs and every design decision: https://github.com/robbiethompson18/oracle-lens-reconstruction

Held-out greedy FVE (N=8 tokens/phrase, K=10 phrases, 256 eval rows):

| checkpoint | FVE |
|---|---|
| random phrases | 0.0011 |
| oracle after SFT | 0.058 |
| oracle after RL | 0.236 |

Reference (ceselder, July 2026, same recipe, full fine-tune): SFT ~0.11, RL 0.21-0.22.

All three are rsLoRA adapters (r=64, alpha=16) on every Linear of the text decoder; load with `peft` on top of the
base model. The oracle expects the lens activation injected by replacing the input embedding of the marker token
`㈜` (id 158983) with `16000 * a/|a|`; see `src/olens/common.py::EmbedInjector` and `oracle_data.py` for the prompt.
