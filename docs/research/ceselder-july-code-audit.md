# ceselder/oracle-lens-qwen3.6-27b: code audit (July 2026 bundle)

Source: https://huggingface.co/ceselder/oracle-lens-qwen3.6-27b, commits 2026-07-21 ("initial
commit", "Add files using upload-large-folder tool") through 2026-07-22 ("oracle-lens bundle
(attempt 1)" x3). Files read in full (20 files, 2994 lines): README.md and every file under
`code/oracle_lens/`. Local copy (untrusted, never executed):
`/private/tmp/claude-501/-Users-robbie-repos/d49a4578-b224-4434-8165-053b55a415f5/scratchpad/dl/ceselder_oracle_lens_code/`

Big caveat up front: **the YAML config (`configs/oracle_lens.yaml`) is not in the repo**, and
neither are the launchers (`run_vllm.sh`), the script that built the `oracle_rl_wrap` vLLM
checkpoint, or the LoRA-merge script. Every numeric hyperparameter below is either (a) hard-coded
in the Python, (b) stated in the README, (c) recoverable from a code comment, or (d) marked
UNKNOWN. I searched ceselder's public GitHub repos (`EasyNLA`, `modulation-lens`,
`olens-1layer` [private/404]) and none contain the yaml.

Repo artifacts and sizes (from the HF API):

| path | bytes | what |
|---|---|---|
| `oracle_lens_model/*.safetensors` (12 shards) | ~53.8 GB | merged oracle, `Qwen3_5ForCausalLM`, bf16, 64 layers, d=5120, vocab 248320, transformers 5.5.4 |
| `reconstructor/pytorch_model.bin` | 53,844,723,875 | full bf16 27B copy + fp32 `head` (5120x5120 + bias) |
| `whitening/W.npy` | 104,857,728 | fp32 [5120,5120] = (Σ+λI)^{-1/2} |
| `whitening/mean.npy` | 20,608 | fp32 [5120] |
| `code/oracle_lens/*.py` | ~140 KB | pipeline |
| `PROBE.txt` | 6 | literally the word "probe" (upload rate-limit test) |

Subject model facts from `oracle_lens_model/config.json`: hybrid architecture, `layer_types` is
`linear_attention` x3 then `full_attention`, repeating (full attention at indices 3,7,...,63).
Layer 45 (the workspace layer) is a `linear_attention` (Gated DeltaNet) block; the activation
read is the input to block 45 = output of block 44 (also linear-attn; 43 is full attention).
`eos_token` = `<|im_end|>`, `pad_token` = `<|endoftext|>`. The chat template defaults to
thinking mode: with `add_generation_prompt=True` and `enable_thinking` undefined it emits
`<|im_start|>assistant\n<think>\n` (chat_template.jinja lines 147-152).

---

## 1. Pipeline, stage by stage

Stages are numbered in the file docstrings 0..7. All multi-GPU stages are written for
`--nproc-per-node=8` / `--num_gpus 8`, and comments name the GPU as H200 (141 GB).

### Stage 0: corpus generation (`gen_corpus.py`)

- Launch: `torchrun --nproc-per-node=8 -m oracle_lens.gen_corpus`. Each rank holds a full bf16
  replica (`device_map={"": local_rank}`), processes `range(rank, N, world)`.
- Prompts: `load_prompts` tries `cfg.corpus.prompt_dataset` then falls back in order to
  `lmsys/lmsys-chat-1m`, `HuggingFaceH4/ultrachat_200k`, `Open-Orca/OpenOrca` (streaming
  `train`). Takes the **first user message only** (`_first_user_prompt`, handles
  `conversation`/`messages`/`conversations` or `prompt`/`instruction`/`question`/`text`
  columns). Filters: `language == "English"` if the row has a `language` field (only lmsys
  does), `10 <= len(text) <= 4000` chars, dedup exact text. Shuffled with `random.Random(0)`.
  Count: `corpus.n_transcripts` (UNKNOWN).
- Splits: `assign_splits` assigns each prompt a split label from `corpus.splits` (dict
  name->fraction; UNKNOWN fractions). Split names used downstream: `recon_train`, `dict`,
  `teacher`, `rl`, `eval`.
- Generation: HF `model.generate`, `do_sample=True`, `temperature=corpus.temperature`,
  `top_p=corpus.top_p`, `max_new_tokens=corpus.max_new_tokens`, prompt truncated to
  `corpus.max_prompt_tokens`, batch `corpus.batch_size_per_gpu`, left padding. All UNKNOWN
  values. Single-turn chat via `apply_chat_template([{"role":"user",...}],
  add_generation_prompt=True)` with default thinking mode, so **every response begins inside a
  `<think>` block**; the CoT tokens are part of `response_ids`.
- Output: `{corpus}/transcripts_rank{r}.jsonl` rows `{idx, split, prompt, response,
  prompt_ids, response_ids}`. Resumable (counts existing lines).

### Stage 1: activation extraction + whitening (`extract_acts.py`)

- Launch: `torchrun --nproc-per-node=8 -m oracle_lens.extract_acts --splits recon_train,dict,teacher,rl,eval`.
- For each transcript with `len(response_ids) >= 8`: run `prompt_ids + response_ids` through the
  subject model (`use_cache=False`), capture the residual stream **entering block
  `model.layer`** (=45) via `ResidualTap` (a `forward_pre_hook` on `layers[45]`, `args[0]`).
- Candidate positions: `range(len(prompt_ids), len(ids) - 1)`, i.e. every assistant position
  (including `<think>` CoT) that has at least one following token. Per transcript at most
  `extract.positions_per_transcript` (UNKNOWN) sampled with a per-(split,rank) seeded RNG.
- Per-rank cap: `ceil(extract.target_positions[split] / world)` (UNKNOWN per split).
- Stored as **float16** raw (unwhitened) rows in `{acts}/{split}/shard_{rank}.npy` plus
  `shard_{rank}.meta.jsonl` rows `{transcript_idx, pos, next_ids: ids[pos+1 : pos+33]}` (the
  next 32 tokens, which is all the downstream phrase sampler ever needs).
- Whitening (rank 0, after a barrier): take `whitening.n_positions` (UNKNOWN) random rows from
  the `recon_train` shards, `Whitening.fit(acts, whitening.ridge_lambda_frac)`:

  ```python
  mean = acts.mean(0); x = acts - mean
  cov = (x.T @ x) / (len(x) - 1)
  lam = ridge_frac * float(np.trace(cov)) / cov.shape[0]      # ridge = frac * mean eigenvalue
  evals, evecs = np.linalg.eigh(cov + lam * np.eye(d))
  W = (evecs * (evals ** -0.5)) @ evecs.T                      # (Σ + λI)^{-1/2}, symmetric
  ```
  `ridge_lambda_frac` UNKNOWN. Computed in float64, saved float32. W is symmetric so the code's
  inconsistent use of `@ W.T` (most places) vs `@ W` (`oracle_data.recon_act`) is harmless.

### Stage 2: reconstructor training (`train_reconstructor.py`, `reconstructor.py`)

See section 2 for architecture/loss. Training details:

- Launch: `deepspeed --num_gpus 8 -m oracle_lens.train_reconstructor`. Full fine-tune of a 27B
  copy under DeepSpeed **ZeRO-3**, bf16, `overlap_comm`, `stage3_gather_16bit_weights_on_model_save`.
  CPU optimizer offload off by default ("At small micro-batch the 27B ZeRO-3 states fit on
  H200"). Gradient checkpointing **off** by default ("phrases are ~40 tokens; gradient
  checkpointing just wastes time recomputing").
- Optimizer: AdamW, `lr=reconstructor.lr` (UNKNOWN), `weight_decay=reconstructor.weight_decay`
  (UNKNOWN), betas (0.9, 0.95), `WarmupLR` from 0 to lr over `reconstructor.warmup_steps`
  (UNKNOWN), no decay after warmup, grad clip `reconstructor.grad_clip` (UNKNOWN).
- Batch: `micro_batch_per_gpu` x 8 x accum = `batch_size_global` (both UNKNOWN). Epochs
  `reconstructor.epochs` (UNKNOWN).
- Pairs (`PairDataset`): one pair per extracted `recon_train` position. For each row, N is
  drawn once at dataset construction: `n = rng.randint(n_min, min(n_max, len(next_ids)))`
  (`n_min`, `n_max` UNKNOWN; paper is 1..32, and `next_ids` is capped at 32 so n_max <= 32
  effectively). Index shuffled and truncated to `reconstructor.n_pairs` (UNKNOWN). Phrase =
  `next_ids[:n]` as **token ids** (exact N tokens). Target = `(act - mean) @ W.T`, unit-normalized.
- Checkpoint every 500 micro-steps and at end via `engine.save_16bit_model(..., "pytorch_model.bin")`.
- Optional in-training FVE eval (`fve_eval_every`): embeds `fve_eval_dict`=20000 length-4
  phrases from the `dict` split, OMP-decomposes `fve_eval_acts`=1000 `eval`-split activations
  with K=`fve_eval_k`=16 over the full candidate set (`dict_fraction=1.0`), logs mean FVE after
  16 atoms. `recon_fve_probe.py` is the standalone single-GPU version (defaults 2000 acts,
  60000 dict, length 4, K 16; docstring says the full dict+teacher pipeline is "~1.5h").

### Stage 3: dictionary (`build_dictionary.py`)

- Launch: `torchrun --nproc-per-node=8`. Loads the reconstructor on every rank (54 GB each).
- Phrases: shuffle all `dict`-split meta rows (`random.Random(0)`), keep the first
  `dictionary.n_starts` (UNKNOWN; docstring targets "~3.4M-entry" dictionary like the paper,
  which implies ~1M starts). For each `L in dictionary.lengths` (docstring: `{2,4,8,16,32}`),
  take `next_ids[:L]` where `len(next_ids) >= L`, dedupe on the id tuple, and **drop any phrase
  containing a token whose decoded text contains "\n"** (`newline_token_ids`), because the
  oracle output format is one phrase per line.
- Embedding: `build_reconstructor_inputs` (token-space), batch `dictionary.embed_batch_size`
  (UNKNOWN), rank-strided `phrases[rank::world]`, stored fp16, rank 0 interleaves back.
- Output: `{dictionary}/len{L}_dirs.npy` fp16 [m, 5120] and `len{L}_phrases.jsonl` `{ids, text}`.

### Stage 4: OMP teacher (`teacher_omp.py`)

See section 4 for the algorithm. Run details: `torchrun` 8 ranks, no model loaded (pure GPU
linear algebra). Activations: first `teacher.n_activations` (UNKNOWN; paper 1M) rows of the
`teacher` split, rank-strided, batches of `teacher.omp_batch_size` (UNKNOWN). For each length
L the whole `len{L}_dirs.npy` is loaded onto the GPU (fp16; for 1M x 5120 that is ~10 GB).
`max_atoms = teacher.max_atoms` (the log line is literally `fve16=` so 16),
`dict_fraction = teacher.dict_fraction` (paper: random half; value UNKNOWN but the docstring
says "random half"). Output `{teacher}/labels_len{L}_rank{r}.jsonl` rows
`{act_row, atoms (ordered dict indices), coeffs (rounded 5 dp), fve (prefix FVE after each
atom)}`, with a `.done` marker per length for resume.

### Stage 5: oracle SFT (`oracle_data.py`, `train_oracle_sft.py`, `train_oracle_sft_lora.py`)

Example construction (`build_sft_example`), per labeled activation row:

- Only rows labeled at **all** lengths are used (`rows_with_labels` = intersection).
- `L = rng.choice(lengths)`; `k = rng.randint(1, min(oracle.k_max, len(atoms)))` (`k_max`
  UNKNOWN; almost certainly 16 to match `max_atoms`).
- Prompt: `build_oracle_prompt(tok, L, k, oracle.placeholder_token)` (section 3).
- Target token ids: teacher's first `k` atoms **in OMP selection order** (not sorted by
  coefficient), each phrase's exact token ids followed by the single `"\n"` token; the final
  `\n` is replaced by `tok.eos_token_id` (= `<|im_end|>`). Labels are -100 on the prompt.
- Injected vector: the raw activation, or if `oracle.sft.inject_recon` is true the
  "reconstructed-activation warm-start (Euan's method)":

  ```python
  # â in raw residual space = mean + (ẑ·‖z_real‖)·W⁻¹
  z = (a - mean) @ W; zn = ‖z‖
  zhat = Σ c_i · dir(atom_i)          # norm ~0.3 since OMP fit the unit-normalized z
  â = mean + (zhat * zn) @ inv(W)     # inv(W) = (Σ+λI)^{1/2}
  ```
  The comment explains why: "injecting the raw activation a (which is ~80% phrase-unexplainable
  residual -> noisy inverse, caps SFT at ~0.12 FVE)". Whether the final run used it is UNKNOWN.
- The per-example RNG is `random.Random(seed*1_000_003 + i)`, so **(L, k) for row i is
  identical every epoch**; multi-epoch SFT re-sees exactly the same examples.

Full-FT trainer (`train_oracle_sft.py`): `deepspeed --num_gpus 8`, ZeRO-3 bf16, AdamW betas
(0.9,0.95), `lr=oracle.sft.lr`, WarmupLR over `oracle.sft.warmup_steps`, grad clip
`oracle.sft.grad_clip`, `micro_batch_per_gpu` default 4, `batch_size_global`, `epochs` (all
UNKNOWN). Gradient checkpointing ON (the `for_training=True` default). Loss: token CE on target
tokens only, with `ResidualInjector` (overwrite at layer 45) set before each forward. Checkpoint
every 500 micro-steps to `ckpt_oracle_sft/pytorch_model.bin`.

LoRA trainer (`train_oracle_sft_lora.py`): a *different* injection scheme: `AdditiveInjector`
at `oracle.inject_layer` (docstring default 1), norm-matched additive, designed to match
`vllm-lens SteeringVector(norm_match=True)`. peft LoRA `r=oracle.lora.r`,
`alpha=oracle.lora.alpha`, `dropout` default 0, `target_modules=oracle.lora.target_modules`
(plain LoRA, not rsLoRA here), torchrun DDP with `find_unused_parameters=True`, AdamW
`lr=oracle.sft.lr`, micro default 8, manual grad accumulation, clip `oracle.sft.grad_clip`,
checkpoint every 2000 steps. This path appears to be a **dead end**: the README and the final RL
script use overwrite injection, and `parity_vllm.py` was written specifically to avoid the
retrain this path implied.

### Stage 6: RL, two generations

**6a. Full-FT GRPO with HF rollouts (`rl_oracle.py`)**: produced the 0.168 checkpoint.

- Launch `deepspeed --num_gpus 8`. Policy = full-FT SFT checkpoint loaded into a fresh base
  (`strict=False`). DeepSpeed **ZeRO-2** ("keeps params local for fast HF-generate rollouts"),
  bf16, **Adam states offloaded to CPU**, grad clip 1.0, AdamW betas (0.9,0.95),
  `lr=oracle.rl.lr` (UNKNOWN). Memory note in docstring: "policy bf16 (54 GB, ZeRO-2 full
  params) + sharded bf16 grads + Adam states offloaded to CPU + frozen reconstructor bf16
  (54 GB)" per H200 rank: each of the 8 GPUs holds its own reconstructor copy.
- Per step: `B_local = prompts_per_step // world` prompts per rank; for each, a random row from
  the `rl` split (rows >= 512; the first 512 rows are the eval holdout), `N = rng.choice(oracle.lengths)`,
  `K = rng.randint(1, k_max)`. Rollouts: `G = group_size` samples per prompt, HF `generate`,
  `do_sample=True`, `temperature=oracle.rl.temperature`, `top_p=1.0`,
  `max_new_tokens = K*(N+1) + max_new_tokens_slack`. Comments reveal the sizes tried:
  "up to B_local*G*k_max ~ 1000+ seqs at 128x8" and "ran clean at 32x8", i.e. **prompts x group
  = 32x8 then 128x8** (whether per-rank or global is ambiguous; `B_local` is per rank so 128x8
  is likely global 128 prompts, 16 per rank).
- Reward (`compute_rewards`): parse phrases (split on the `\n` token, stop at `<|im_end|>`/eos,
  cap at K), embed with the reconstructor in chunks of `recon_batch`=256, then
  `reward = FVE(NNLS refit) - format_penalty_per_phrase * |#phrases - K|
  - format_penalty_per_token * Σ_phrases |len(phrase) - N|` (both penalty weights UNKNOWN).
  Empty output gets FVE 0.
- Advantage: per-prompt group normalization `(r - mean_G) / (std_G + 1e-4)`. **No KL term, no
  reference model, no PPO ratio/clipping**; one optimizer step per rollout batch, so it is
  on-policy REINFORCE with a group baseline (what the file calls GRPO).
- Loss: `-(adv * logp * completion_mask).sum() / total_completion_tokens_this_rank * n_accum`,
  logprob computed frugally as `gather(logits) - logsumexp(logits)` to avoid a full
  [B, T, 248k] log-softmax. Micro-batch `train_micro_batch` (UNKNOWN) with the injector kept
  set through backward (gradient checkpointing recompute must re-fire the injection).
- Also logs within-group vs between-group reward variance to decide whether to scale G or B.
- Eval every `eval_every` steps: greedy or sampled decode on the 512 reserved rows at
  `(eval_n, eval_k)` (README: N=8, K=10), all-reduced mean FVE (not clamped at 0).
- `eval_only` mode measures the SFT warm-start per length, greedy vs sampled.

**6b. LoRA GRPO with vLLM rollouts (`train_rl_vllm_oracle.py`)**: the published model.

- Launched under torchrun with K actor ranks; GPU layout hard-coded: cuda:0 vLLM engine (rank 0
  only), cuda:1..K the LoRA actor (one rank each, one 27B copy each), cuda:(K+1) the
  reconstructor (rank 0 only). `qual_decode.py` says "the live RL run on cuda:0-2", so the
  actual run used **K=1** (3 GPUs: vLLM, actor, reconstructor); the docstring example is K=4.
- Actor: base model + the 0.168 full-FT RL checkpoint, then peft `LoraConfig(r=lora_r
  default 64, lora_alpha=lora_alpha default 16, target_modules="all-linear", use_rslora=True,
  lora_dropout=0.0, bias="none")`. README: lr 3e-5. AdamW betas (0.9,0.95), grad clip 1.0,
  gradient checkpointing ON ("micro=16 over the 248k-vocab logits needs it to fit 140GB", so
  `train_micro_batch`=16).
- vLLM: `LLM(model=/workspace/oracle-lens/run/ckpts/oracle_rl_wrap, tokenizer=<base>,
  dtype=bfloat16, gpu_memory_utilization=0.85, max_model_len=1024, enforce_eager=True,
  enable_prefix_caching=False, additional_config={"gdn_prefill_backend": "triton"})`, env
  `VLLM_USE_FLASHINFER_SAMPLER=0`. Single GPU, **no tensor parallelism**. Injection via
  `vllm_lens` `SteeringVector(activations=a.view(1,1,-1), layer_indices=[44],
  position_indices=[placeholder_pos], norm_match=False, overwrite=True, scale=1.0)` passed in
  `SamplingParams.extra_args["apply_steering_vectors"]`; one SamplingParams per rollout
  (`temperature=oracle.rl.temperature`, `top_p=1.0`, `max_tokens=K*(N+1)+slack`).
- Weight sync: `sync_actor_to_vllm(core, llm, only_adapted=True, name_map=_VLLM_NAME_MAPS["qwen3_5_wrapper"], ipc=True)`
  from `nla.train_rl_vllm` (EasyNLA), pushing the merged LoRA delta over CUDA IPC each step
  ("on-policy, 0.5s").
- Same reward/advantage/loss as 6a; rank 0 computes rewards and `broadcast_object_list`s
  `[gen_ids, adv, Ns, Ks, acts, r_mean, f_mean]`; every rank trains a strided 1/world shard;
  manual all-reduce of LoRA grads (loss pre-scaled by `world`, then SUM/world).
- Eval every `eval_every` steps: greedy, `temperature=0.0`, `max_tokens=K*(N+1)+16`, on
  **only 64 activations** (`range(0, 512, 8)`), FVE clamped to >= 0. The README's "peak 0.2205"
  is almost certainly from this 64-sample curve, so it has roughly +/-0.03 noise.
- Checkpoints: `core.save_pretrained(ckpt_oracle_rl_lora)` every `save_every`; `qual_decode.py`
  references `oracle_rl_lora_s180_preserved`, i.e. a **step-180** adapter was kept, which gives
  the order of magnitude of RL steps (hundreds, not thousands).

### Stage 7: evaluation and readout

- `eval_fve.py`: 8 ranks, each hosting oracle + reconstructor ("bf16 pair fits on one H200"),
  shards `eval.n_positions` rows of the `eval` split, **sampled** decode (`temperature=1.0`
  default in `oracle_decode`), `(eval.n, eval.k)`, writes `eval_fve.json`. Not clamped.
- `qual_decode.py`: greedy decode on `rl` holdout rows at N=8 K=10 next to the text the model
  was actually generating (`next_ids`), plus a random-dictionary-phrase baseline (README: ~0.001).
- `decode.py`: Figure-64-style readout of arbitrary prompts; needs 3 GPUs; `--samples` up to 5
  with best-FVE selection ("paper uses up to 5"), temperature 1.0 when sampling, else greedy.
- `parity_vllm.py`: the gate that let them skip retraining: vLLM overwrite@44 on 32 rows at
  N=8 K=10 greedy must give mean FVE > 0.12 ("HF ResidualInjector@45 reference ~0.16-0.17").

---

## 2. Reconstructor architecture, exactly

```python
class Reconstructor(nn.Module):
    def __init__(self, base_model, hidden_size):
        self.base = base_model                                   # full Qwen3.6-27B copy, trained
        self.head = nn.Linear(hidden_size, hidden_size, bias=True)   # 5120 -> 5120, N(0, 0.02), zero bias
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=..., attention_mask=..., output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1]                                # final layer (HF: post final RMSNorm)
        last = attention_mask.sum(-1) - 1                        # last real token (right padding)
        h_last = h[arange(B), last]
        pred = self.head(h_last.to(self.head.weight.dtype))
        return F.normalize(pred.float(), dim=-1)                 # unit vector in whitened space
```

- Model: a full copy of the subject (`AutoModelForCausalLM.from_pretrained(cfg.model.name)`),
  all parameters trained (ZeRO-3 full FT), plus one affine head. The lm_head is unused.
- Read: **last token of the final hidden state** (`hidden_states[-1]`, which in HF's
  convention has the final norm applied). Only the last position is read.
- Input: `prompt_template.format(phrase=...)`. The template string is UNKNOWN. The token-space
  builder tokenizes `prompt_template.format(phrase="")` with `add_special_tokens=False` as a
  prefix and appends the raw phrase token ids, so the template must end with `{phrase}` (any
  suffix is silently dropped) and phrase tokens are exactly the subject model's own tokens.
  `embed_phrases` (text-space) tokenizes the formatted string whole with `max_len=40` and is
  not used by the shipped pipeline (all callers use `build_reconstructor_inputs`).
- Target: `z = (a - mean) @ W.T; z /= ‖z‖`, where `a` is the fp16-stored raw residual at the
  position *preceding* the phrase (input to block 45).
- Loss (`train_reconstructor.py:201`): `loss = ((pred - target) ** 2).sum(-1).mean()` with both
  unit vectors, i.e. `2 - 2 cos` per example, averaged over the micro-batch. This is exactly the
  paper's "mean-squared error between the unit-normalized Σ^{-1/2}-whitened prediction and
  target". Cosine is logged every 20 steps.
- Inference: base in bf16, head cast to fp32, output fp32 then stored fp16 in the dictionary.

---

## 3. Injection scheme and prompt

Final scheme (README, `common.ResidualInjector`, `parity_vllm.py`, `train_rl_vllm_oracle.py`):
**overwrite**, not additive.

```python
class ResidualInjector:   # forward_pre_hook on get_decoder_layers(model)[layer], layer = 45
    def _hook(self, module, args):
        hidden = args[0]
        if hidden.shape[1] <= int(self.positions.max()):
            return                       # KV-cached decode step: no-op
        hidden = hidden.clone()
        hidden[arange(B), self.positions] = self.vectors.to(hidden.device, hidden.dtype)
        return (hidden,) + args[1:]
```

- Where: the residual stream **entering block 45** of the oracle, the same place the activation
  was read from the subject. In vLLM/vllm-lens terms (which hook block outputs) that is
  `layer_indices=[44]`, `overwrite=True, norm_match=False, scale=1.0`.
- What: the **raw** residual vector `a` (not whitened, not normalized), cast to bf16.
- Fires only on prefill; the hook is a no-op on cached decode steps. Under gradient
  checkpointing the injector must stay set through `backward` so the recompute matches.
- Marker: `oracle.placeholder_token`, a single existing vocabulary token (UNKNOWN which; it is
  looked up with `tok.convert_tokens_to_ids` and located with `ids.index(ph_id)`, so it must be
  a token that appears exactly once in the rendered prompt).

Prompt (`common.py`), verbatim:

```python
ORACLE_SYSTEM = (
    "You are an activation decoder. An internal activation vector from a language "
    "model has been placed at the marked position. Output exactly {k} phrases, each "
    "exactly {n} tokens long, that together describe the content of this activation. "
    "One phrase per line, no numbering, no extra text."
)
def build_oracle_prompt(tok, n, k, placeholder_token):
    user = placeholder_token + "\n" + ORACLE_SYSTEM.format(k=k, n=n)
    msgs = [{"role": "user", "content": user}]
    ids = chat_ids(tok, msgs, add_generation_prompt=True)
```

So the rendered prompt is `<|im_start|>user\n<PLACEHOLDER>\nYou are an activation decoder ...
no extra text.<|im_end|>\n<|im_start|>assistant\n<think>\n` (the trailing `<think>\n` comes
from the Qwen3.6 template default; `enable_thinking` is never set). The oracle therefore learns
to emit its phrase list *inside* an open think block and terminate with `<|im_end|>`. Despite
the variable name there is no system role; it is all in the user turn. Output format: K lines,
each phrase exactly N tokens, newline-separated, `<|im_end|>` terminated. Parsing
(`parse_generated_phrases`) splits on the single `\n` token id, drops empty phrases, stops at
`<|im_end|>`/eos, caps at K.

The alternative `AdditiveInjector` (`inject_additive.py`): `forward_hook` on the **output** of
block `layer` (default 1), `h'_p = h_p + ‖h_p‖ · v/‖v‖` (Karvonen/NLA style, equals vllm-lens
`norm_match=True`). Only used by `train_oracle_sft_lora.py`; the docstring there calls the
overwrite scheme "the OLD (HF-only) path", but the shipped model and README use overwrite.

---

## 4. FVE metric and OMP teacher, exactly

Whitened target: `z = (a - mean) @ W.T`, `z = z / max(‖z‖, 1e-8)` with `W = (Σ+λI)^{-1/2}`.
Everything downstream operates on unit `z`.

NNLS refit and FVE (`common.py`), shared by RL reward, eval, decode:

```python
def nnls_refit(D, x, iters=200):           # D [m, d] unit rows, x [d]
    G = D @ D.T; b = D @ x
    L = torch.linalg.eigvalsh(G)[-1].clamp_min(1e-6)   # Lipschitz constant
    c = torch.clamp(b, min=0) / L
    for _ in range(iters):
        c = torch.clamp(c + (b - G @ c) / L, min=0)     # projected gradient
    return c
def fve_of(D, x):
    if D.shape[0] == 0: return 0.0, None
    c = nnls_refit(D.float(), x.float())
    resid = x - c @ D
    return 1.0 - (resid.norm()**2 / x.norm()**2).item(), c
```

So FVE = 1 - ‖z - Σ c_i d_i‖² / ‖z‖² with c >= 0, in whitened space, with the duplicate-phrase
case handled implicitly (duplicates share weight). The reconstructor directions `d_i` are unit
vectors, so coefficients are magnitudes. Projected gradient with a fixed step is not an exact
NNLS solver but converges for m <= 16. FVE can be negative only through non-convergence; some
eval paths clamp at 0 (`parity_vllm`, `qual_decode`, vLLM `run_eval`), others don't
(`eval_fve.py`, `rl_oracle.run_eval`, training reward).

Teacher OMP (`teacher_omp.py:omp_batch`), batched on GPU:

```python
mask = torch.rand(B, m) < dict_fraction            # per-datapoint random subset of the length-L dict
R = X.clone()
for k in range(max_atoms):
    corr = (R.to(D.dtype) @ D.T).float(); corr[~mask] = -1e30
    best_val, best_idx = corr.max(-1)
    alive = alive & (best_val > 1e-4)               # stop when no atom has positive correlation
    ...select atom, forbid reselection (mask[...] = False)
    c = batched_nnls(S[:, :k+1], X)                 # refit ALL selected atoms (60 iters) against X
    R = X - c @ S[:, :k+1]
    fves[:, k] = 1 - ‖R‖² / ‖X‖²
```

- Selection: max raw inner product between the current residual and the (unit) atoms,
  restricted to the random `dict_fraction` subset for that datapoint, no reselection.
- Refit: NNLS over all selected atoms after every selection, 60 projected-gradient iterations
  (vs 200 in `fve_of`). Coefficients may go to 0 for an earlier atom but it stays in the list.
- Records atoms in selection order, final coefficients, and the FVE prefix curve. One label
  file per length L; each activation is decomposed independently at every L.
- Matches the paper's "non-negative OMP, one phrase length at a time, random half of the
  entries, up to sixteen phrases" if `dict_fraction=0.5` and `max_atoms=16` (both consistent
  with the docstring and `fve16` log string).

---

## 5. Data sources

- Prompts: `cfg.corpus.prompt_dataset` (UNKNOWN; the fallback order is `lmsys/lmsys-chat-1m`
  (gated), `HuggingFaceH4/ultrachat_200k`, `Open-Orca/OpenOrca`), first user turn only,
  English filter where available, 10..4000 chars.
- Text: generated **on-policy by Qwen3.6-27B itself** (`cfg.model.name`, UNKNOWN whether the
  base `Qwen/Qwen3.6-27B` or an instruct variant; the README says `Qwen/Qwen3.6-27B`), via HF
  `generate` on 8 GPUs, **not vLLM** ("Stage 0: ... HF generate (no vLLM)"). Sampling
  temperature/top_p UNKNOWN. Thinking mode on, so responses include CoT.
- Activations: layer-45 input residual at assistant positions (CoT + answer), fp16.
- Split sizes: UNKNOWN (`corpus.splits`, `extract.target_positions`). Paper-scale hints in
  docstrings: "~3.4M-entry phrase dictionary", "1M" teacher activations.
- RL activations: the `rl` split, rows >= 512, re-sampled with replacement every step ("fresh
  activations" only in the sense of being unseen by SFT, not freshly generated).
- vLLM is used **only** for the LoRA-RL rollouts and its eval; everything else is HF.

---

## 6. Compute

- Hardware: an 8x H200 node (141 GB each). Evidence: every launch line uses 8 processes;
  comments "fit on H200", "fit 140GB", "bf16 pair fits on one H200" (54+54 GB).
- Parallelism: no tensor parallelism anywhere. Data parallel replicas (torchrun) for corpus,
  extraction, dictionary, teacher, eval. DeepSpeed ZeRO-3 for reconstructor and full-FT SFT,
  ZeRO-2 + CPU Adam offload for full-FT RL. DDP/manual all-reduce for the LoRA variants.
  vLLM single GPU, `enforce_eager=True`, `gpu_memory_utilization=0.85`, `max_model_len=1024`.
- Gradient checkpointing: on by default in `load_model_and_tokenizer(for_training=True)`,
  explicitly off for the reconstructor, on for LoRA RL at micro 16.
- Memory per rank in full-FT RL (docstring): 54 GB policy + sharded bf16 grads + 54 GB
  reconstructor, Adam on CPU. In the LoRA RL run: 3 GPUs (vLLM, actor with grad checkpointing,
  reconstructor).
- Wall-clock hints: dictionary + teacher "~1.5h full pipeline" (recon_fve_probe docstring);
  LoRA->vLLM sync 0.5 s/step; first vLLM load + Triton GDN compile + first rollout > 10 min
  (hence 2 h NCCL timeout); LoRA RL kept a step-180 checkpoint. No total GPU-hour figure.
- Batch sizes: everything in config (UNKNOWN) except `recon_batch=256`,
  `gen_prompts_per_call=4` (x G sequences per HF generate call), dictionary embed chunk 512
  (probe/eval), eval chunk 32, HF RL prompts x group of 32x8 and 128x8 (from comments),
  LoRA RL micro 16.

---

## 7. Bugs, hacks, TODOs, dead ends found in the code

1. **Config not shipped.** `configs/oracle_lens.yaml` is referenced everywhere; without it
   nothing runs. All `cfg[...]` keys are enumerated in section 8.
2. **Hard-coded absolute paths**: `/workspace/oracle-lens/run/ckpts/oracle_rl_wrap`
   (`train_rl_vllm_oracle.py:39`, `parity_vllm.py:27`),
   `/workspace/oracle-lens/run/ckpts/oracle_rl_lora_s180_preserved` (qual_decode docstring).
3. **Private fork dependency**: `from nla.train_rl_vllm import sync_actor_to_vllm, _VLLM_NAME_MAPS`
   with `name_map=` and `_VLLM_NAME_MAPS["qwen3_5_wrapper"]`. The public EasyNLA
   (`ceselder/EasyNLA`, `asherps/EasyNLA`) has `sync_actor_to_vllm(actor, llm, ipc=False,
   only_adapted=True)` and **no `_VLLM_NAME_MAPS`**, so the RL script depends on an unpublished
   modification.
4. **Qwen3.6 kernel instability hacks**: the fla/tilelang linear-attention kernel throws
   "Triton Error [CUDA]: invalid argument" for large batches, so HF generation is chunked to
   `gen_prompts_per_call=4` prompts (x G) per call and reconstructor forwards to <= 256
   sequences. flashinfer's `gdn_prefill` JIT fails to compile ("cuda::ptx errors"), so vLLM is
   forced to `gdn_prefill_backend: triton`; `VLLM_USE_FLASHINFER_SAMPLER=0`.
5. **torchrun/vLLM env collision**: rank 0 pops `RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR,
   MASTER_PORT, ...` before constructing the vLLM engine, else vLLM's workers join the DP
   store. NCCL timeout bumped to 2 h and an early `dist.barrier` forces comm init.
6. **Gradient checkpointing + injection**: the injector must remain set through `backward`
   (recompute must re-inject) or you get `CheckpointError`; LoRA needs
   `enable_input_require_grads()` or adapters get zero gradient. Both are noted in comments.
7. **Duplicate wandb logging** in `rl_oracle.py:328-333` (two `wb.log` calls per step).
8. **Dead code / naming**: `reconstructor.save_consolidated` has
   `"model.safetensors" if False else "pytorch_model.bin"` and is never called;
   `extract_acts` names the output `npz_path` but writes `.npy` and the docstring says `.npz`;
   `Reconstructor.embed_phrases` (text-space, `max_len=40`) is unused by the pipeline.
9. **fp16 activation storage**: residual activations are stored as float16. Qwen residual
   streams have massive-activation dimensions; any value > 65504 would overflow to inf and
   poison the whitening fit. No guard is present. Worth checking on a replication.
10. **Reconstructor template boundary**: token-space inputs are `prefix_ids + phrase_ids` where
    `prefix_ids = tok(template.format(phrase=""))`; if the template has text after `{phrase}` it
    is dropped, and BPE merges across the prefix/phrase boundary differ from tokenizing the full
    string (only matters if anyone uses `embed_phrases`).
11. **Oracle prompt in thinking mode**: `build_oracle_prompt` never passes
    `enable_thinking=False`, so the prompt ends with `<think>\n`. Works because SFT targets
    follow it directly, but any replication that renders the prompt differently (e.g. vLLM chat
    API instead of `TokensPrompt`) will mismatch.
12. **SFT examples fixed across epochs** (per-index seeded RNG) and **teacher-order targets**
    (first K atoms in greedy order, not coefficient order).
13. **Eval leakage/noise**: RL eval and `qual_decode` use the first 512 rows of the *rl* split
    (same transcripts/distribution as training activations, excluded only by row index); the
    vLLM RL eval uses 64 rows; `eval_fve.py` uses the separate `eval` split with sampling at
    temperature 1.0. The README table mixes these ("~0.21-0.22 (peak 0.2205)" is the in-run
    curve). FVE clamping at 0 is inconsistent across scripts.
14. **Dead-end path**: `train_oracle_sft_lora.py` + `AdditiveInjector@layer1` (norm-matched add,
    vllm-lens `norm_match=True` compatible) was built so vLLM rollouts would match training, then
    abandoned once `parity_vllm.py` showed overwrite@44 in vllm-lens reproduces the HF
    overwrite@45 oracle (threshold: mean greedy FVE > 0.12 on 32 rows).
15. **`inject_recon` warm start ("Euan's method")** exists as a config switch; the comment
    claims raw-activation SFT "caps SFT at ~0.12 FVE" (README reports ~0.11). Whether the shipped
    SFT used it is UNKNOWN.
16. **`load_state_dict(strict=False)`** everywhere for oracle checkpoints: silently tolerates
    missing/unexpected keys (e.g. multimodal wrapper vs text-only key names). The reconstructor
    loader asserts only that `head.*` keys are present.
17. **vLLM needs a multimodal wrapper checkpoint** (`oracle_rl_wrap`, "text=our RL weights"):
    the HF text-only `Qwen3_5ForCausalLM` weights had to be re-wrapped into
    `Qwen3_5ForConditionalGeneration` layout for vLLM; that conversion script is not in the repo.
18. `common.load_model_and_tokenizer` docstring says the HF checkpoint is the multimodal
    `Qwen3_5ForConditionalGeneration` loaded via `AutoModelForCausalLM`; the published
    `config.json` is `Qwen3_5ForCausalLM` (`model_type: qwen3_5_text`). `get_decoder_layers`
    probes three attribute chains to cope with both layouts and peft wrapping.
19. Teacher OMP's `mask` tensor is `[B, m]` bool per batch (for m ~ 1M and `omp_batch_size`
    in the hundreds that is fine) and `S` is `[B, 16, 5120]` fp32; the whole fp16 dictionary for
    one length sits on each GPU.

---

## 8. Everything needed to rerun that is NOT in the repo

Files / artifacts:
- `configs/oracle_lens.yaml` with at least these keys (all referenced in code):
  `model.{name, dtype, attn_implementation?, layer, hidden_size}`;
  `paths.{root, corpus, acts, whitening, ckpt_reconstructor, dictionary, teacher,
  ckpt_oracle_sft, ckpt_oracle_sft_lora, ckpt_oracle_rl, ckpt_oracle_rl_lora}`;
  `corpus.{prompt_dataset, n_transcripts, splits, batch_size_per_gpu, max_prompt_tokens,
  temperature, top_p, max_new_tokens}`; `extract.{positions_per_transcript, target_positions{split}}`;
  `whitening.{n_positions, ridge_lambda_frac}`;
  `reconstructor.{n_min, n_max, n_pairs, prompt_template, micro_batch_per_gpu, batch_size_global,
  grad_clip, lr, weight_decay, warmup_steps, epochs, grad_checkpointing?, cpu_offload_optimizer?,
  fve_eval_every?, fve_eval_length?, fve_eval_dict?, fve_eval_acts?, fve_eval_k?, save_ckpt?}`;
  `dictionary.{lengths, n_starts, embed_batch_size}`;
  `teacher.{n_activations, max_atoms, dict_fraction, omp_batch_size}`;
  `oracle.{k_max, placeholder_token, lengths, inject_layer}`,
  `oracle.sft.{micro_batch_per_gpu?, batch_size_global, grad_clip, lr, warmup_steps, epochs,
  cpu_offload_optimizer?, inject_recon?}`, `oracle.lora.{r, alpha, dropout?, target_modules,
  micro_batch_per_gpu?}`, `oracle.rl.{prompts_per_step, group_size, train_micro_batch, lr, steps,
  temperature, max_new_tokens_slack, format_penalty_per_phrase, format_penalty_per_token,
  eval_every, eval_n, eval_k, save_every, recon_batch?, gen_prompts_per_call?, eval_only?,
  lora_r?, lora_alpha?, vllm_gpu_mem?, vllm_max_len?, sync_ipc?}`; `eval.{n, k, n_positions}`.
  Known values: layer 45, hidden 5120, lengths {2,4,8,16,32}, eval N=8 K=10, max_atoms 16,
  LoRA r=64 alpha=16 rsLoRA all-linear lr 3e-5, train_micro_batch 16 (LoRA RL). Everything
  else UNKNOWN, including the reconstructor prompt template and the placeholder token.
- `run_vllm.sh` launcher (modes `parity`, `rl_vllm`; sets NGPU, picks the vllm-lens venv).
- The `oracle_rl_wrap` multimodal-wrapper checkpoint and the script that produces it from the
  text-only RL weights; the LoRA merge script that produced `oracle_lens_model/`.
- The corpus, activation shards, dictionary, teacher labels, and the SFT / full-FT RL (0.168)
  checkpoints. Only the reconstructor, whitening, and final merged oracle are published.
- `probes.json` for `decode.py`.

Environments and external code:
- Two venvs: a "main" one (torch, transformers 5.5.x with Qwen3.5/3.6 support, deepspeed,
  peft with `use_rslora`, datasets, numpy, pyyaml, wandb optional) and a "vllm-lens venv"
  (vllm with Qwen3.6 hybrid GDN support, `vllm-lens` from UKGovernmentBEIS, flashinfer
  optional, Triton/fla/tilelang kernels for Gated DeltaNet).
- `vllm_lens._helpers.types.SteeringVector` with `overwrite=True` support and
  `extra_args={"apply_steering_vectors": [...]}` per-request steering (PyPI `vllm-lens` 1.3.0
  as of Oct 2026; ceselder's own `vllm-metamodels` fork advertises "drop-in for vllm-lens
  1.1.0", so a 1.1.x-era API is likely what was used in July).
- `nla.train_rl_vllm.sync_actor_to_vllm(..., name_map=..., ipc=...)` and `_VLLM_NAME_MAPS`:
  a modified EasyNLA not on GitHub.
- No version pins anywhere; `transformers_version: 5.5.4` in the saved config is the only pin.
- Env vars: `WANDB_PROJECT` (optional), `VLLM_USE_FLASHINFER_SAMPLER=0` (set in code), standard
  torchrun/deepspeed vars; HF token for `lmsys/lmsys-chat-1m` if that dataset is wanted.
- Hardware: 8x H200-class GPUs (>= 140 GB each) for the full-FT stages; the LoRA RL needs 3
  such GPUs minimum (vLLM, actor, reconstructor).

---

## 9. Deviations from the Anthropic recipe

| Anthropic (Haiku 4.5) | ceselder (Qwen3.6-27B) | Notes |
|---|---|---|
| Subject = Haiku 4.5, layer unspecified | Qwen3.6-27B, layer 45 of 64 (hybrid GDN/attention; 45 is a linear-attention block) | |
| "Assistant-turn text" | Self-generated single-turn responses to LMSYS/UltraChat/OpenOrca first-user prompts, thinking mode on, so CoT tokens count as assistant text | Training distribution includes reasoning traces |
| Reconstructor: "copy of the subject model fine-tuned to map phrase -> activation" | Full-FT copy + extra 5120x5120 affine head on last-token final hidden state, output normalized; phrase wrapped in an unknown prompt template | Head and read position are design choices the paper doesn't specify |
| N uniform in 1..32 | N uniform in `n_min..min(n_max, available)` (UNKNOWN; <= 32 by data) | one pair per extracted position, `n_pairs` cap |
| Loss: cosine error in whitened metric | `‖pred - target‖²` with unit vectors = 2-2cos | Identical |
| Whitening: template-lens Σ^{-1/2} | `(Σ + λI)^{-1/2}` with `λ = frac * tr(Σ)/d`, fit on `n_positions` recon_train rows | Ridge added; frac UNKNOWN |
| Dictionary: ~1M starts, lengths 2/4/8/16/32, dedupe per length, ~3.4M vectors | Same lengths, `n_starts` UNKNOWN (targets ~3.4M), plus removal of any phrase containing a newline token | Extra filter |
| Teacher: 1M held-out activations, NN-OMP per length, random half, up to 16 | Same algorithm; `n_activations`, `dict_fraction` UNKNOWN (docstring says half), `max_atoms` 16; NNLS refit after each atom with 60 PG iterations; stop at corr <= 1e-4 | Paper doesn't say whether it refits after each selection |
| Oracle = second fine-tuned copy, activation injected, prompt specifies N and K, target = first K phrases | Same; full-FT ZeRO-3 SFT (or LoRA variant); injection = overwrite at block-45 input at a placeholder token; prompt text as in section 3 ends inside `<think>`; optional `inject_recon` warm start | Paper doesn't specify injection layer/mode; inject_recon is an addition |
| RL on fresh activations, reward = FVE minus small format penalties | GRPO-style REINFORCE (group-normalized advantage, no KL, no clipping), reward = FVE - a·|#phrases-K| - b·Σ|len-N|; activations resampled from a fixed `rl` split; **two RL phases**: full-FT with HF rollouts (0.168) then rsLoRA r64/α16 lr 3e-5 with vLLM rollouts on top (0.22) | Paper says nothing about the RL algorithm or LoRA |
| Inference: sample K phrases, reconstructor, NNLS refit | README/qual/parity/vLLM-eval use **greedy**; `eval_fve.py` samples at T=1; `decode.py` best-of up to 5 samples | Paper's figure uses up to 5 samples |
| Headline: 31% whitened FVE, held-out on-policy assistant positions | 0.168 (full-FT RL) / ~0.21-0.22 peak 0.2205 (LoRA RL) at N=8, K=10, mostly measured on 64-512 rows of the rl split; teacher ~0.19, SFT ~0.11, random phrases ~0.001 | Paper's eval N/K not stated in the quoted appendix |
| Dictionary / teacher from held-out corpora | Separate `dict`, `teacher`, `rl`, `eval` splits of the same self-generated corpus | Matches in spirit |
| (not stated) | Activations stored fp16; dictionary directions fp16; whitening fp32 | Precision choices |

Net: the structure is a faithful port of the four-stage recipe; the substantive deviations are
(1) the affine head + last-token read in the reconstructor, (2) the thinking-mode corpus and
prompt, (3) the overwrite-at-the-same-layer injection choice, (4) the ridge in the whitening,
(5) the two-phase full-FT-then-LoRA GRPO without KL, (6) greedy decoding at eval, and (7) the
much lower FVE (0.22 vs 0.31) on a different subject model with a noisier eval.
