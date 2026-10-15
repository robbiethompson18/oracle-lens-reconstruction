# Compute plan: oracle lens on Qwen/Qwen3.6-27B

Written 2026-10-15. Prices and versions were fetched that day; links are inline.

## 0. TL;DR

- Qwen3.6-27B = `model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration` (vision tower + text decoder). Text decoder: 64 layers, 48 Gated-DeltaNet + 16 full-attention (every 4th: 3, 7, ..., 63), hidden 5120, vocab 248,320, 262k ctx, Apache-2.0. ~26.8B text params -> 53.6 GB bf16.
- Stages (a) extraction, (b) reconstructor SFT, (c) oracle SFT all fit on **1xH200** (or 1xH100 with micro-batch <=16 and chunked CE). GRPO (d) wants **2xH200** (vLLM + frozen reconstructor on one, policy on the other) or **1xB200** in colocate mode; 2xH100 only works if the reconstructor shares the policy's base weights as a second PEFT adapter.
- Estimated wall-clock on H200: (a) ~1-2 h, (b) 6-13 h, (c) 13-25 h, (d) 15-33 h for 300-500 steps at 1,024 completions/step.
- RunPod Secure Cloud today: H100 SXM $3.99, H200 $5.29, B200 $7.99, A100-80GB $1.79 /GPU-h (Community: $2.69 / $3.59 / $5.98 / $1.39). Pipeline total ~ **$450 (low) / $800 (mid) / $1,600 (high incl. reruns)**. Author-scale (ceselder's NLA RL: 6xB200 x 800 steps x 440 s) would be ~ $4.7k, so our budget assumes ~4x smaller RL batches and fewer steps.
- Stack: vLLM 0.28-0.31 (TRL 1.16 supports 0.22-0.31; `vllm-metamodels` tested on 0.27.1/0.28.0), transformers 5.17 in the vLLM venv (vLLM 0.31 pins `<5.18`), PEFT 0.21.2, TRL 1.16.0, flash-linear-attention 0.5.2 + causal-conv1d 1.7.0 (mandatory for speed), torch 2.13.0 (vLLM pin). Two venvs like the existing repo pattern.
- Hard parts: (1) per-request activation injection in vLLM needs the `vllm-metamodels` fork/plugin (stock vLLM only has `--enable-prompt-embeds`, which covers embedding-replacement injection, not layer-L steering); (2) TRL needs a custom `rollout_func` to attach a vector per prompt; (3) 248k-vocab logits blow up memory at batch 32 x 512 (16-24 GB) -> chunked CE; (4) loading the multimodal checkpoint into `Qwen3_5ForCausalLM` has no key-remap in the modeling file -> verify weights actually loaded; (5) RunPod driver roulette for cu130 torch.

## 1. Qwen3.6-27B facts

Sources: [config.json](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json), [model card](https://huggingface.co/Qwen/Qwen3.6-27B), [transformers qwen3_5 docs](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5), [modeling_qwen3_5.py](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py), all read 2026-10-15.

| Field | Value |
|---|---|
| `architectures` | `Qwen3_5ForConditionalGeneration`; `model_type: qwen3_5`; `text_config.model_type: qwen3_5_text` |
| Multimodal | Yes (vision tower: depth 27, hidden 1152, out 5120, ~0.4B params). `language_model_only: false`. Image/video token ids 248056/248057. |
| Text decoder | `Qwen3_5TextModel`: 64 layers, `layer_types` = 3x `linear_attention` then 1x `full_attention` (`full_attention_interval: 4`; full-attn layers are 3, 7, 11, ..., 63) |
| hidden_size / intermediate | 5120 / 17408 (SiLU) |
| Full attention | 24 q heads, 4 kv heads, head_dim 256, output gate (`attn_output_gate: true`; q_proj emits 2x width, half is a sigmoid gate), partial RoPE 0.25 (64 dims), rope_theta 1e7, interleaved mRoPE sections [11,11,10] |
| Gated DeltaNet | 16 k heads x 128, 48 v heads x 128, conv kernel 4, `mamba_ssm_dtype: float32` |
| vocab | 248,320 (padded), untied lm_head |
| context | 262,144 native; 1,010,000 with YaRN |
| MTP | 1 MTP layer shipped (`mtp_num_hidden_layers: 1`); HF ignores `^mtp.*` keys |
| License | Apache-2.0 |
| Release | ~2026-04-21/22 (aggregators disagree by a day; card cites April 2026) |
| `transformers_version` in config | 4.57.1 (what it was saved with); card says "latest transformers"; docs say Qwen3.6 shares classes with Qwen3.5 |
| Serving | card: `vllm>=0.19.0`, `sglang>=0.5.10`; [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B): vLLM >=0.17; BF16 needs 1xH200 or 2xH100 TP2; FP8 checkpoint `Qwen/Qwen3.6-27B-FP8` fits a 40 GB card; `--language-model-only` skips the vision tower |

Parameter count (text only, my arithmetic from config): embeddings 248,320x5120 x2 (untied) = 2.54B; MLP 64 x 3x5120x17408 = 17.1B; full-attn 16 x (q 5120x12288 + k,v 2x5120x1024 + o 6144x5120) = 1.7B; GDN 48 x (in_proj_qkv 5120x10240 + in_proj_z 5120x6144 + out_proj 6144x5120 + small) = 5.5B. Total ~ 26.8B -> **53.6 GB bf16**. Vision + MTP add ~0.8B (~1.6 GB) if you load the whole wrapper.

### Gotchas for hooking the residual stream

- **Classes.** `Qwen3_5ForCausalLM` wraps `Qwen3_5TextModel` as `self.model` (params `model.layers.*`). The checkpoint stores `model.language_model.layers.*`. The modeling file has no `_checkpoint_conversion_mapping`; the docs quickstart nonetheless uses `Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-9B")`, so transformers v5's central conversion mapping presumably handles it. **Verify on first load**: no "newly initialized" warning, and logits match `Qwen3_5ForConditionalGeneration(...).model.language_model`. The safe path (what the oracle-lens README calls "the text decoder of the Qwen3.6 multimodal wrapper") is to load `AutoModelForImageTextToText` and take `.model.language_model` + `.lm_head`. Cost of the vision tower is ~0.8 GB, negligible. Community LoRA cards confirm the name mismatch bites PEFT adapters trained under one class and loaded under the other ([hotdogs/Qwen3.6_thinkingcap_lora](https://friendli.ai/models/hotdogs/Qwen3.6_thinkingcap_lora)). Pick one class and keep it for all three LoRAs.
- **`output_hidden_states` indexing.** `hidden_states[0]` = embeddings, `hidden_states[i]` = output of decoder block `i-1`. ceselder's NLA card says "L42 = floor(2*64/3), AR backbone has 43 blocks" -> the target is the output of block index 42 = `hidden_states[43]`. Decide and document the convention before extracting 1M vectors. The final RMSNorm is applied only to `last_hidden_state`; mid-stack entries are raw residuals. Layer 42 is a `linear_attention` block, 43 is `full_attention`; the residual stream is the same tensor either way.
- **Kernels.** The DeltaNet path uses `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` from `fla` and `causal_conv1d_fn/update` from `causal_conv1d`, via `use_kernel_func_from_hub_with_fallback`. Without both packages it **silently falls back** to the PyTorch reference (slower, more memory; fp32 recurrent path). Install `flash-linear-attention>=0.5.2` and `causal-conv1d>=1.7.0` (`--no-build-isolation`), Python 3.12 (ms-swift's Qwen3.5 guide; fla issue #121/#758). `use_kernels=True` (`Atlas-Inference/gdn`) is only for GB10/Strix Halo.
- **PEFT targets.** In transformers the GDN module has separate `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` (plain `nn.Linear`, no bias) plus `conv1d` (depthwise), `A_log`, `dt_bias`, `norm`. No `in_proj_qkvz` / `in_proj_ba` in HF (those fused names exist in vLLM's packed mapping and are split when LoRA is on). ceselder's target list `q/k/v/o_proj, in_proj_a/b, in_proj_qkv/z, out_proj, gate/up/down_proj` therefore matches every Linear in the decoder; `target_modules="all-linear"` gives the same set minus lm_head, but if you load the wrapper it also catches the vision tower's linears, so use the explicit list. `in_proj_a/b` are 5120->48 (rank 64 > out dim; harmless but pointless, consider dropping).
- r=64 all-linear LoRA size: sum(in+out) over targets ~ 7.3M x 64 ~ **470M params ~ 0.94 GB bf16**; fp32 master + Adam ~ 5.6 GB.
- vLLM: Gated DeltaNet state cannot be prefix-cached (`vllm-metamodels` README); CUDA-graph capture size interacts with the mamba cache (`--max-cudagraph-capture-size`, recipe); keep `max_num_seqs <= 1024` with CUDA graphs on vLLM 0.19 hybrids (fork bug note).

## 2. Memory and throughput

### Assumptions

- Training FLOPs per token with LoRA + gradient checkpointing ~ 6N: forward 2N, backward-through-frozen-weights 2N, recompute 2N. N = 26.8e9 -> **1.6e11 FLOP/token**. LoRA weight-grad FLOPs are negligible.
- Achieved bf16 throughput (MFU 30-40%, hybrid with Triton fla kernels, moderate batch): H100 SXM 989 TFLOPS peak -> ~350 TFLOPS; H200 same compute, more HBM (4.8 vs 3.35 TB/s) -> ~380; B200 2.25 PFLOPS -> ~750 (fla/causal-conv1d Triton kernels are less tuned on Blackwell, so discounted); A100 312 -> ~110.
- -> **tok/s (train, LoRA+ckpt)**: A100 ~700, **H100 ~2,000** (1,300-2,800), **H200 ~2,300**, **B200 ~4,500** (3,500-5,500). No published single-GPU LoRA benchmark exists for a 27B hybrid; NGC's 8xH100 Llama-3-70B bf16 LoRA = 14.4k tok/s/node = 1.8k/GPU, which scales to ~4.6k/GPU at 27B; my number is conservative because of checkpointing and GDN kernels. Measure in the first hour and rescale.
- Inference (vLLM, bf16 27B, batched 256-1024 seqs): decode is weight-bandwidth bound: 54 GB / 4.8 TB/s = 11 ms/step on H200 -> theoretical 20k tok/s at batch 256; realistic **4-8k tok/s** on H200, ~3-5k on H100, ~8-12k on B200.

### Memory (bf16 base 53.6 GB + LoRA r=64)

| Item | Size |
|---|---|
| Base weights bf16 | 53.6 GB (55 with vision+MTP) |
| LoRA params + grads + fp32 Adam | ~5-7 GB |
| Checkpoint boundaries, B x T tokens | B*T*5120*2 B x 64 layers: B=32,T=512 -> 10.7 GB; B=32,T=128 -> 2.7 GB |
| One layer's recompute (MLP 17408-wide, GDN chunks) | ~2-3 GB at 16k tokens |
| **Logits**, 248k vocab | B*T*248320*(2 bf16 + 4 fp32 for CE): B=32,T=512 -> **24 GB**; T=128 -> 6 GB |
| vLLM KV for GRPO rollouts | attention KV = 16 layers x 4 heads x 256 x 2 x 2 B = 32 KB/token; GDN state per seq ~ 48x128x128x4 B ~ 3 MB -> 1,024 seqs x 200 tok ~ 6.5 GB + 3 GB |

Per stage:

| Stage | Peak GPU memory | Fits 1xH100 80? | 1xH200 141? |
|---|---|---|---|
| (a) extraction (inference, B=32x512) | 54 + ~4 | yes | yes |
| (b) reconstructor LoRA (spans <=12 tok + template ~ 40 tok, no lm_head; value head 5120^2) | 54 + 7 + small; B=256 of 40-token seqs ~ 10k tokens -> +7 GB | yes | yes |
| (c) oracle SFT (B=32, T~128-256, LM loss) | 54 + 7 + 3-6 ckpt + 6-12 logits ~ 70-80 GB | marginal: micro-batch 8-16 + chunked/fused CE | yes |
| (d) GRPO policy GPU (micro 8-16 x 190 tok) | 54 + 7 + 3 + 3 ~ 67-72 GB (+1 GB if reconstructor adapter shares the base) | yes, tight | yes |
| (d) vLLM GPU | 54 + 10 KV + overhead ~ 66 GB | yes (`gpu_memory_utilization 0.9`) | yes |
| (d) separate reconstructor copy | +55 GB | no | only together with vLLM on H200 (54+54+10 = 118 <= 141, `gpu_memory_utilization ~ 0.45`) |

Key trick: the reconstructor is `base + LoRA_AR + value head` and the policy is `base + LoRA_oracle`; both can live on **one** copy of the base as two PEFT adapters (train one, freeze the other, `set_adapter` to switch). That removes a 54 GB copy and makes 2xH100 viable. Caveat: ceselder's skiplens AR is a *truncated* backbone (63 layers, final norm stripped; the NLA AR has 43 blocks), so on a shared base you run only the first k blocks (slice `model.layers[:k]`), which is cheap. And if the oracle is warm-started from a merged `av_base` (as in the NLA card), the bases differ; avoid merging and stack adapters instead.

Reference model: with PEFT, `beta>0` uses `disable_adapter()` on the same weights, no extra copy. TRL default `beta=0` loads no ref at all.

### Wall-clock per stage (arithmetic shown)

(a) **Extraction, ~1M positions.** Assume 20k rollouts x ~200 tokens = 4M generated tokens (positions sampled from them). vLLM on 1xH200 at 5k tok/s -> 800 s. Forward pass for activations: 4M x 2N = 2.1e17 FLOP / 380e12 = 560 s. Storage 1M x 5120 x 2 B = 10 GB (20 GB fp32). **~1-2 h** including model download (56 GB; ~1 min with hf_xet on a good pod, 10+ min on a slow one) and whitening stats. Negligible cost.

(b) **Reconstructor, 50-100M tokens.** 50M / 2,300 = 21,700 s = **6.0 h**; 100M -> **12 h** on H200 (H100 7-14 h, B200 3-6 h). Warning: spans are ~40 tokens, so batch 32 = 1,300 tokens/step and the step is launch-latency bound (64 layers x dozens of kernels, ~0.3-0.5 s) -> use B>=128-256 so the GPU is compute-bound. ceselder's skiplens AR: 776k spans, 1 epoch, batch 32, lr 1e-4, dense `--ar-all-idx` (every causal prefix supervised; this is the token-efficient objective).

(c) **Oracle SFT, 1M examples.** Assume 100-200 tokens/example (template + marker + ~60-token explanation) -> 100-200M tokens -> 100M/2,300 = 12 h, 200M -> 24 h -> **13-25 h** on H200 (B200 7-13 h). The NLA card used r=128 for its AV warmstart stage.

(d) **GRPO.** Per step with P=64 prompts x G=16 = 1,024 completions, ~60-token prompt, ~128-token completion:
- rollouts: 131k tokens on vLLM@H200 at ~5k tok/s -> ~30 s (steering at the prompt marker position only -> CUDA graphs can stay on with `vllm-metamodels`; `vllm-lens` forces eager -> ~2x slower decode).
- reward: reconstructor forward on 1,024 phrases x 40 tok = 41k tokens x 2N = 2.2e15 / 380e12 ~ 6 s.
- policy update: 1,024 x 190 tok = 195k tokens x 6N = 3.1e16 / 380e12 ~ 80 s (B200 ~45 s). Old/ref logprob forwards (if `num_iterations>1` or beta>0): +2 x 195k x 2N ~ +50 s.
- weight sync: TRL merges the LoRA and pushes **all 54 GB** over NCCL to the server, then unmerges: ~10-20 s on NVLink, ~30 s on PCIe-only community hosts.
- -> **~3-4 min/step** -> 300 steps = 15-20 h, 500 steps = 25-33 h on 2xH200; ~2.5 min/step on 1xB200 colocate (gen and train serialize anyway in synchronous GRPO).
- Calibration: ceselder's `qwen3.6-27b-nla-rl` ran GRPO with group 8, ~504 prompts/step (4,032 completions, 4x ours) plus a co-trained AR critic at **440 s/step on 6xB200**, 800 steps. That is ~9 completions/s on 6 B200 with critic training vs my ~5/s on 2 H200 without, consistent. The oracle-lens RL checkpoint (`agu18dec/oracle-lens-ddp600`) used group 64, beta_KL 0.02, lr 5e-5, 600 steps, DDP-8; GPU type not stated (the main `agu18dec/oracle-lens` card is gated, HTTP 401).

Does 1xH200 suffice for the SFT stages? **Yes** for (a)(b)(c) with batch 32 x 512 if you chunk the CE loss. 1xH100 also works for all three with micro-batch <=16 (+ grad accumulation). GRPO needs 2 GPUs (or 1xB200): vLLM must not share a device with the trainer in server mode (TRL doc warning); colocate mode on one 180 GB B200 is the single-GPU option (`vllm_gpu_memory_utilization~0.35`, `vllm_enable_sleep_mode`).

## 3. Where to run

### Existing launch pattern (`/Users/robbie/repos/loracles-project`)

SkyPilot 0.14.0 on RunPod, no Docker, no SkyPilot workdir (uploads to the API server fail, so code goes over ssh):

```
uvx --from 'skypilot[runpod]' sky launch -c <name> -y --detach-run pod.yaml   # optionally --region US, --retry-until-up
tar cf - *.py *.sh | ssh -T <name> 'mkdir -p ~/sp && tar xf - -C ~/sp'        # subliminal_pilot/deploy_v2.sh
ssh -T <name> 'tmux new-session -d -s v2 "bash ~/sp/go.sh 2>&1 | tee go.log"'  # long jobs under tmux
```

`pod.yaml` shape (`loracles-dash/loracle/upd/pod.yaml`, `sleeper/pod_train.yaml`, `subliminal_pilot/sky_eval.yaml`): `cloud: runpod`, `accelerators: {H200-SXM:1, H100-SXM:1, H100:1, H100-NVL:1, A100-80GB-SXM:1, A100-80GB:1, RTXPRO6000:1}` (dict = ordered fallback list; 4- and 8-GPU variants in `loracle/upd/pod4.yaml`, `jlens/upd/pod8.yaml` with `{H100-SXM:8, H200-SXM:8, ...}`), `disk_size: 150-400`, `setup:` installs uv -> `~/venv` (py3.12) -> `vllm peft safetensors huggingface_hub[hf_xet]`, or `torch --index-url .../cu128` + transformers/peft when vLLM isn't needed, then `hf download`. Env: `HF_HOME=/hf HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0`. Multi-GPU pods run one worker per `CUDA_VISIBLE_DEVICES` (`boot.sh`). `echo/setup.sh` keeps a second venv `/root/vllm` so vLLM's torch pin doesn't fight the training venv; reuse that.

Known problems recorded in the repo:
- **Driver roulette**: EU RunPod hosts shipped a CUDA 12.8 driver (<580), which breaks the cu130 torch that vLLM pulls in (`subliminal_pilot/deploy_v2.sh`, `pods_ledger.md` 2026-10-12 10:17). `subliminal_pilot/relaunch_good.sh` loops `sky down` / `sky launch --region US` until `nvidia-smi --query-gpu=driver_version` >= 580. Reuse it. Alternative: install vLLM's cu12.9 wheel variant (vLLM 0.29+ ships cu129 wheels/images alongside the cu130 default) to lower the driver floor.
- **Balance**: RunPod ran out of credit on 2026-10-12 23:20 UTC and stopped every pod; SkyPilot lost the records. Top up before a 30 h run and watch for stopped pods in the console (`pods_ledger.md`).
- Some hosts were ~8x slower (~12% GPU util, `subl2-whale`); the fix was relaunching elsewhere.
- Pods launched outside SkyPilot are invisible to `sky status`.
- Measured datapoint: Qwen3-14B LoRA r=16, batch 32, ~1 s/step on H200 (`subliminal_pilot/README.md`); this project is ~2x the model and 4x the LoRA rank.

### Current prices (per GPU-hour, on-demand, fetched 2026-10-15)

| GPU | RunPod Secure | RunPod Community | Lambda (1x/8x) | Nebius on-demand (spot) | Vast.ai verified (mid-2026 snapshots) | Prime Intellect (aggregator, unverified) |
|---|---|---|---|---|---|---|
| H100 SXM 80 GB | **$3.99** | $2.69 | $4.29 / $3.99 | $4.50 (from $0.79) | $1.80-2.50 | ~$2.43 |
| H100 PCIe | $2.89 | $1.99 | $3.29 | - | $1.50-2.27 | - |
| H200 141 GB | **$5.29** | $3.59 | not listed | $5.40 (from $0.79) | $3.50-5.50 | ~$3.14 |
| B200 180 GB | **$7.99** | $5.98 | $6.99 / $6.69 | $8.50 (from $0.99) | $6-12, thin supply | ~$3.49 |
| A100 80 GB | $1.79 | $1.39 | $2.79 (8x SXM) | - | - | - |
| RTX PRO 6000 96 GB | $2.49 | $1.69 | - | - | - | - |

Sources: [RunPod pricing](https://www.runpod.io/pricing) (both tiers in the page's structured data; Secure prices rose Oct 5-12 per [pricingsaas](https://pricingsaas.com/companies/runpod/diffs/2026W42): H100 SXM 3.49->3.99, H200 4.59->5.29, B200 6.79->7.99); [Lambda](https://lambda.ai/pricing); [Nebius](https://nebius.com/prices) (raised on-demand ~20% on 2026-10-01 per [fourweekmba](https://fourweekmba.com/ai-nebius-lifts-on-demand-gpu-prices-up-to-21-from-1-october/)); Vast.ai figures from [Spheron's July 2026 comparison](https://www.spheron.network/blog/vastai-pricing-2026/) and [DeployBase](https://deploybase.ai/articles/vast-ai-gpu-cloud-pricing-complete-guide-vs-hr-for-every-gpu) (Vast's own live grid is JS-rendered); Prime Intellect numbers via [apis.io](https://apis.io/plans/prime-intellect/prime-intellect-plans-pricing/)/[toolradar](https://toolradar.com/tools/prime-intellect/pricing), contradictory and unverified (its /pricing page 404s). Market median for H100 SXM on-demand across 16 providers was $4.20 on 2026-10-07 ([Spheron](https://www.spheron.network/blog/h100-price-per-hour-2026/)).

Availability caveats: RunPod stopped accepting new Community Cloud hosts (Sept 2026), Community has no SLA and is where the old-driver hosts live; Secure Cloud US had H200-SXM singles on demand in the ledger's runs. B200 supply on RunPod is region-limited; Vast B200 is nearly absent. Storage: RunPod volume $0.10/GB-mo running, $0.20 idle; 300 GB for a week ~ $7. Nebius spot is cheap but preemptible (bad for a 30 h GRPO run; TRL checkpoints fine, but each vLLM restart + 54 GB reload costs minutes).

### Cost table (RunPod Secure, recommended config)

| Stage | Config | Hours (low-high) | $/h | $ (low-high) |
|---|---|---|---|---|
| (a) extraction | 1xH200 | 1-3 | 5.29 | 5-16 |
| (b) reconstructor, 50-100M tok | 1xH200 | 6-13 | 5.29 | 32-69 |
| (c) oracle SFT, 100-200M tok | 1xH200 | 13-25 | 5.29 | 69-132 |
| (d) GRPO, 300-500 steps | 2xH200 | 15-33 | 10.58 | 159-349 |
| (d) alt | 1xB200 colocate | 12-25 | 7.99 | 96-200 |
| Setup, downloads, smoke tests, idle | 1xH200 | 6-12 | 5.29 | 32-63 |
| **Compute subtotal** | | | | **~$300-630** |
| Debug reruns (x1.5-2.5 is realistic for a new stack with a vLLM fork) | | | | **$450 / $800 / $1,600** (low/mid/high) |

Community Cloud is ~30% cheaper (H200 $3.59) but adds driver roulette; A100s ($1.79) are ~3x slower per GPU so roughly break-even per token, and don't fit GRPO. B200 is the best $/token for (b)(c) only if the fla kernels run well on Blackwell (unverified): (b)+(c) at 4,500 tok/s = 11-17 h x $7.99 = $90-135 vs $100-200 on H200, a wash, so prefer H200 for the known-good kernel path and move to B200 only after measuring.

## 4. Software stack

PyPI versions on 2026-10-15: vllm 0.31.0 (2026-10-05, pins torch==2.13.0, transformers>=5.10.4,<5.18); transformers 5.20.0 (2026-10-15); peft 0.21.2 (2026-10-01); trl 1.16.0 (2026-10-15, vllm 0.22-0.31); flash-linear-attention 0.5.2 (2026-07-27); causal-conv1d 1.7.0 (2026-08-20); verl 0.9.1 (2026-09-20, pins vllm==0.24.0, torch 2.11, transformers<5.11); skyrl-train 0.3.1 (2026-01-02, vllm 0.11, stale); torch 2.14.1.

Recommendation:
- **Training venv**: torch 2.13 (cu129 or cu130 depending on host driver), transformers 5.17.x (shareable with vLLM), peft 0.21.2, trl 1.16.0, flash-linear-attention 0.5.2, causal-conv1d 1.7.0, accelerate. Verify `fla`/`causal_conv1d` import *inside* this venv; a failed build silently costs 2-5x on the 48 GDN layers.
- **vLLM venv**: vLLM 0.28.0 (tested end-to-end by `vllm-metamodels`; inside TRL's 0.22-0.31 window; has model-runner-V2 hidden-state hooks). Try 0.31.0 after, since the fork's hooks read vLLM internals and its README says newer releases need rechecking. Launch flags for TRL: `VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3.6-27B --language-model-only --weight-transfer-config '{"backend":"nccl"}' --logprobs-mode processed_logprobs --max-logprobs -1 --max-model-len 1024 --max-num-seqs 1024`. No prefix caching (useless on GDN).
- **Steering in vLLM**: stock vLLM has no per-request residual-stream injection. What exists: (i) `--enable-prompt-embeds` ([docs](https://docs.vllm.ai/en/latest/features/prompt_embeds/)): pass `(seq, hidden)` embeddings via `EmbedsPrompt` or base64 over `/v1/completions`, which is exactly "replace the marker token's embedding" (NLA-style `EMBED_LAYER_INDEX`) with zero fork; untested on hybrids and it bypasses the chat template; (ii) [UKGovernmentBEIS/vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens) (MIT, pip plugin, per-request `SteeringVector` in `SamplingParams.extra_args`/`vllm_xargs`, norm-match + layer indices, auto-discovers Mamba/linear-attention hybrids, but forces `enforce_eager`, TP-duplicated saves, cloudpickle = RCE on the server); (iii) [ceselder/vllm-metamodels](https://github.com/ceselder/vllm-metamodels): drop-in for vllm-lens 1.1.0, batched indexed injection ("3-59x faster"), `mode=add|replace`, `norm_match`, `position_indices` for the marker, keeps CUDA graphs because steering is applied at prompt positions only; benchmarked on Qwen3.6-27B; tested on vLLM 0.16/0.19/0.27.1/0.28.0; notes a LoRA crash on rank-64 adapters in 0.27.1 (so push merged weights, don't serve the LoRA) and `max_num_seqs<=1024` on 0.19 hybrids; (iv) [IBM vllm-hook/MIA](https://github.com/ibm/vllm-hook) (Apache-2.0, per-request `steer` in `vllm_xargs`, validated vLLM 0.29 + torch 2.13 + V2 runner, 4-8% TPOT overhead, CUDA-13 driver required). Use (iii); (i) is the fallback if the oracle's injection is at the embedding.
- **RL framework**: TRL 1.16 `GRPOTrainer` with `peft_config`, `use_vllm=True, vllm_mode="server"` (or `colocate` on 1xB200). Reward: `reward_funcs=[fn]` where `fn(completions, act_idx, **kwargs)` looks up the stored activation by dataset column `act_idx` (TRL forwards extra dataset columns as kwargs; keep tensors out of the dataset, memmap them), runs the frozen reconstructor (second adapter on the trainer's base, or a separate copy on the vLLM GPU), returns -MSE/cosine in whitened space. Rollouts: override `rollout_func` so each prompt carries its vector via `vllm_xargs` (the stock path only sends token ids to `/v1/completions`); it must return `prompt_ids`, `completion_ids`, `logprobs`. Weight sync: TRL merge -> NCCL push -> unmerge of the full model each step (vLLM PR #30739 removed DP for dense models; use TP if you ever need 2 GPUs of generation). verl 0.9.1 has Qwen3.5 GRPO recipes and LoRA fixes but pins vLLM 0.24 (older than the fork's tested set); skyrl is stale. TRL is the lowest-friction choice; verl only for async/large scale.
- **Loss memory**: 248k vocab -> use a chunked or fused linear-CE for (c) and the GRPO logprobs (TRL has `use_liger_kernel`; whether Liger's fused CE patches `qwen3_5` is unverified; check `liger_kernel.transformers` model mapping, else implement a 2k-row chunked CE).

### Hard parts, ranked

1. **Per-request injection + CUDA graphs + hybrid GDN inside vLLM** (the fork is pinned to specific vLLM internals; each vLLM bump is a re-test). Budget a day.
2. **TRL plumbing**: custom `rollout_func` + reward with activation lookup + importance-sampling correction (vLLM logprobs vs trainer logprobs; the fork notes batch-composition-dependent bf16 differences).
3. **Memory on 80 GB cards**: logits and the second 27B copy; solved by chunked CE and the shared-base two-adapter trick, otherwise use H200/B200.
4. **Class/key naming**: pick `ForConditionalGeneration.model.language_model` vs `Qwen3_5ForCausalLM` once; verify weights loaded; adapter param names must match across (b), (c), (d) and the vLLM merge.
5. **Kernels**: fla + causal-conv1d must import; measure tok/s in the first 10 minutes and compare to ~2k/s on H100.
6. **RunPod ops**: driver >=580 (cu130) or cu129 wheels; balance; checkpoint every ~50 GRPO steps and push adapters to HF because pods vanish.
