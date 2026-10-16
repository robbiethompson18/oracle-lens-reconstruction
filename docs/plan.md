# Plan: an oracle lens for Qwen3.6-27B

Written 2026-10-15 by Robbie's Claude. Sources: five research reports in this directory
(`report_anthropic_paper.md`, `report_ceselder_july_code.md`, `report_ceselder_github.md`,
`report_workspacebench.md`, `report_compute.md`), plus the downloaded code/cards under `dl/`.
Nothing has been run on a GPU yet. No money spent.

## 0. What an oracle lens is, in one paragraph

Take the residual stream at one layer and one token position (a 5120-dim vector `a`). Whiten it
(`z = Σ^{-1/2}(a - μ)`, unit-normalise). A **reconstructor** is a fine-tuned copy of the subject
model that maps a short text phrase to the whitened direction the model's residual stream points in
just *before* it would say that phrase (a learned "template vector"). An **oracle** is a second
fine-tuned copy that, given `a` injected into its own forward pass, emits K phrases of N tokens such
that a non-negative combination of their reconstructor directions explains as much of `z`'s variance
as possible (FVE). The oracle is warm-started by distilling an NN-OMP teacher over a few-million-phrase
dictionary, then RL'd with FVE as the reward while the reconstructor stays frozen. Anthropic reports
31% whitened FVE on Haiku 4.5 at 2/3 depth. The frozen, pre-specified reconstructor plus the
linear-combination constraint is the whole point versus NLAs: it should confabulate less.

## 1. The three things "the oracle lens on Qwen3.6-27B" could mean

| | A: ceselder July (`oracle-lens-qwen3.6-27b`) | B: ceselder Aug "olens" (`olens-rl-grpo-trajectory/step_00390`) | C: WorkspaceBench authors (`agu18dec/olens_and_ar`) |
|---|---|---|---|
| Recipe | literal Anthropic appendix: reconstructor → 3.4M-phrase dictionary → NN-OMP teacher → SFT → GRPO | 4 free-form bullets "what the model is about to say", expert iteration (best-of-32 via J-lens) + GRPO | AR → inverter (CE) → OMP-distill 4 bullets → GRPO, 11 layers in one adapter |
| Layer | 45 (70% depth) | 42 (66%) | 20–60 step 4, layer in the prompt |
| Injection | overwrite residual entering block 45 | additive norm-matched at block-1 output | replace marker embedding with 16000·unit(h) |
| Output | K phrases of exactly N tokens | 4 `*` bullets | ≤4 `- ` bullets |
| Headline | wFVE 0.22 (N=8,K=10) vs paper 0.31 | wsbench 0.638 (her judging), wFVE ~0.55 | wsbench 0.69 (Opus/Sonnet) / post's Gemini numbers |
| Code public? | **yes, all 20 files** (missing: config YAML, private EasyNLA fork) | **no** (mining loop, "jacobian L42 AR" private) | **no** (`camilablank/global-workspace` is 404); checkpoints + 872 GB of ablation results public |
| Her own verdict | "OLD, not vLLM-compatible"; FVE is info-capped | "the better generalist" | n/a |

Robbie's instruction: "see discussion at bottom of both papers" → the Anthropic appendix describes A,
the AF appendix describes C. "ceselder has already made one" → her artifact literally named
"oracle lens" is A; her best is B.

**Recommendation: build A (the paper's recipe), with the cost-saving substitutions in §3, and use
her July numbers as the "you're close" gate. Then evaluate on WorkspaceBench so it can also be
compared to B and C.** Reasons: A is the only variant with a fully specified public pipeline and a
crisp scalar match criterion (held-out wFVE at fixed N,K); it is what both papers call "the oracle
lens"; B and C both need an unpublished component. The known risk: A's output format (verbatim
N-token phrases) scored lower on her own eval than B's free-form bullets, so if the end goal is
"best workspace reader" rather than "replicate the paper", C's recipe is the better target and
costs about the same. This is decision 1 below.

## 2. Decisions Robbie has to make

1. **Target lineage**: A (paper-faithful; recommended) / C (WorkspaceBench recipe) / B.
2. **Budget ceiling**: compute estimate for A is $300–630 if nothing goes wrong, realistically
   **$800, worst case $1,600** on RunPod Secure (H200 $5.29/h, prices rose 10–15% last week).
   A 1/10-scale pilot first costs ~$150 and de-risks everything. WorkspaceBench judging is extra:
   the Gemini families are ~$20–50 via OpenRouter; the two Sonnet-judged safety families are
   ~$1,300+ each and I propose skipping them.
3. **Corpus thinking mode**: ceselder July had thinking ON (CoT tokens count as assistant text); C
   and WorkspaceBench use `enable_thinking=False`. I propose OFF (matches the benchmark's chat
   render and "what the model is thinking during a normal reply"). Cheap to do both later.
4. **Reuse her code or rewrite**: her July code is MIT-less on HF (no license file in the HF repo;
   her GitHub repos EasyNLA/skip-lens/modulation-lens are MIT, vllm-metamodels and olens-1layer have
   no license). I propose: read it as reference, write our own ~1.5k-line pipeline (it is small),
   and use her `vllm-metamodels` fork only if stock vLLM prompt-embeds injection fails.
5. **Where**: RunPod via SkyPilot (your existing loracles-project pattern), US region only.
   Needs a topped-up RunPod balance (it ran dry on Oct 12 and killed every pod).
6. **Repo**: new repo `~/repos/olens` via /new-repo (Python/uv), or should this live in
   `loracles-project`? There is an empty `~/repos/olens-copy` from today 12:42; tell me what that
   was for.

## 3. Recommended recipe (lineage A with modern economies)

Everything at **layer 42** = output of decoder block 42 = HF `hidden_states[43]`, raw residual,
no norm. (Paper's L67 is 66.7% of depth = 42.7 of 64; ceselder's later work, the NLA baseline, the
J-lens checkpoints and the SAE all use 42. Her July used 45; I would not.)

Subject model: `Qwen/Qwen3.6-27B` (`Qwen3_5ForConditionalGeneration`, 64 layers: 48 Gated-DeltaNet
+ 16 full-attention at 3,7,…,63; d=5120; vocab 248,320; Apache-2.0). Load the wrapper and take
`.model.language_model`, or `Qwen3_5ForCausalLM` after verifying the weight keys actually loaded
(the checkpoint stores `model.language_model.layers.*`; a silent key mismatch gives you a random
model). Install `flash-linear-attention 0.5.2` + `causal-conv1d 1.7.0` or the GDN layers silently
fall back to slow PyTorch.

### Stage 0: on-policy corpus (~1–2 GPU-h, 1×H200)
- Prompts: first user turn of `allenai/WildChat-1M` (C used it; lmsys is gated; ultrachat as
  fallback). English, 10–4000 chars, dedup.
- vLLM `--language-model-only`, `enable_thinking=False`, T=1.0, top-p 1.0, max 512 new tokens.
- Target ~60k transcripts (~12M response tokens). Splits by transcript: recon_train 50%, dict 20%,
  teacher 20%, rl 7%, eval 3%.

### Stage 1: activations + whitening (~1 GPU-h)
- One forward per transcript, hook block 42 output, sample ≤32 assistant positions per transcript
  with `next_ids = ids[pos+1:pos+33]`. Store **fp32 or bf16, not fp16** (Qwen residuals have
  massive-activation dims; July stored fp16 with no overflow guard).
- Whitening `W = (Σ + λI)^{-1/2}`, λ = 0.1·tr(Σ)/d (C's `ridge_c 0.1`; July's value unknown), fit on
  200k recon_train rows in fp64. **Pick this convention once and keep it**: ceselder used three
  mutually incomparable whiteners (full-cov ridge in Jul, diagonal-on-L2-normed in Aug, full-cov
  eigen-floor in Sep). Our FVE numbers are only comparable to her July ones.

### Stage 2: reconstructor (~6–12 GPU-h, 1×H200)
- Paper: full-FT copy. **Substitution**: base truncated to blocks 0..42 (43 blocks, final norm →
  identity), LoRA r64/α16 rsLoRA on all linears, fp32 `Linear(5120,5120)` head on the last real
  token. This is ceselder's Aug/Sep AR shape and C's (r16). C's ablation: AR objective and size
  barely matter after RL, so the paper's full-FT is not worth 8×H200.
- Input `Summary of the following text: <text>{phrase}</text> <summary>` style template ending in
  the phrase; phrase = next N tokens, N ~ U{1..32}; target = unit whitened `z` at the preceding
  position. Loss = `||pred - z||²` on unit vectors (= 2 − 2cos, exactly the paper).
- 1 epoch over ~1M pairs (~20M span tokens; C saw post-RL quality flat from ~13M). Batch 256,
  lr 1e-4 constant (her AR sweep: 1e-5 undertrains, 1e-4 peak), warmup 100.
- Gate: held-out whitened cosine; expect ~0.2–0.25 FVE-equivalent (her ARs: 0.23–0.25 at L62,
  C's 0.226 pooled). In-training probe: OMP-16 over 20k 4-token candidates on 1k acts.

### Stage 3: dictionary (~1 GPU-h)
- 300k start positions from the `dict` split (paper: 1M; scale up later if teacher FVE is weak),
  lengths {2,4,8,16,32}, dedup per length, drop phrases containing a newline token → ~1M
  directions, bf16, ~10 GB. Embed with the reconstructor at batch 256 (her Triton error above ~32
  seqs/launch on some kernels: chunk if it bites).

### Stage 4: NN-OMP teacher (~1 GPU-h, no model)
- 300k `teacher` activations (paper 1M), per length, random 50% of the dictionary per datapoint,
  greedy max-correlation selection, NNLS refit (projected gradient, 60 iters) after every pick,
  stop at corr ≤ 1e-4 or 16 atoms. Record atoms in selection order, coefficients, FVE prefix.
- Gate: teacher FVE@16 ≈ 0.19 (her July number). If far below, the reconstructor is the problem.

### Stage 5: oracle SFT (~8–15 GPU-h, 1×H200)
- Oracle = base (all 64 blocks) + LoRA r64/α16 rsLoRA. **Injection: replace the marker token's
  input embedding with 16000·unit(a)** (C's scheme; works with stock vLLM `--enable-prompt-embeds`,
  no fork; C's ablation: raw h beats centred/whitened; a single existing CJK token ㈜ id 158983
  is the marker, chosen so it stays one token inside the chat render). Fallback if SFT FVE is
  bad: her July overwrite-at-block-42 scheme (then rollouts need vllm-metamodels).
- Prompt (one user turn, `enable_thinking=False`): the July text, "You are an activation decoder.
  An internal activation vector from a language model has been placed at the marked position.
  Output exactly {k} phrases, each exactly {n} tokens long … One phrase per line." with
  `<activation>㈜</activation>` before it.
- Per example: L ~ choice{2,4,8,16,32}, k ~ U{1..16}, target = teacher's first k atoms, exact token
  ids, `\n`-joined, EOS-terminated; CE on target only. Resample (L,k) per epoch (July fixed them).
- Try both injecting raw `a` and July's "Euan's method" (inject the OMP reconstruction `â`
  during SFT, raw `a` at RL) on the pilot; she noted raw-`a` SFT caps at ~0.12.
- Chunked cross-entropy (248k vocab × batch × seq is the memory wall). Batch 64, lr 1e-4, 1 epoch
  over ~300k examples × 5 lengths sampled = ~1M rows.
- Gate: held-out greedy wFVE at N=8,K=10 ≈ 0.11 (her SFT number).

### Stage 6: RL (~15–30 GPU-h, 2×H200 or 1×B200)
- GRPO/REINFORCE: per step 64 activations × 16 rollouts, N ~ choice{lengths}, K ~ U{1..16}
  (bias toward N=4,8 K=8,10 for the eval regime), T=1.0, `max_new = K(N+1)+16`.
- Reward = wFVE after NNLS refit of the K reconstructor directions − 0.01·|#phrases−K|
  − 0.002·Σ|len−N| (penalty weights unknown in July; tune on pilot). Group-normalised advantage,
  KL β=0.02 to the SFT adapter (C) or 0 (July); start with 0.02.
- lr 3e-5 (her LoRA phase), AdamW (0.9,0.95), clip 1.0, 300–500 steps, checkpoint + push adapter to
  HF every 25 steps (pods die).
- Rollouts: vLLM server with prompt embeds, TRL 1.16 `GRPOTrainer` with a custom `rollout_func`
  (attach the embedding per prompt) and a python reward that runs the frozen reconstructor. The
  reconstructor can share the policy's base weights as a second PEFT adapter (run only blocks
  0..42) so only one 54 GB copy is resident. Fallback: the July-style lean loop (~500 lines) with
  HF generate; slower but zero framework risk.
- Gate (the "you're close" criterion): **held-out greedy wFVE at N=8,K=10 ≥ 0.20** (her 0.21–0.22,
  measured on 64–512 rows so ±0.03), random-phrase baseline ≈ 0.001, and the sampled-decode FVE
  curve rising monotonically without the phrases degenerating (check the subject model's NLL of
  the phrases as a naturalness/hacking detector, her Sep trick).

### Stage 7: evaluation beyond FVE
- Qualitative: Figure-64-style readouts on the paper's nine prompt types (blackmail transcript,
  buggy code, poetry planning, directed modulation) at N=4,K=10, greedy and best-of-5.
- WorkspaceBench: `camilablank/workspace-bench` (MIT). Plug in as a single-layer `Method` at L42
  first (the harness allows it), produce readouts for the ~1.1k-call Gemini families + the six
  regex multi-token families + hallucination (5.6k cells), skip agentic (607k cells, Sonnet) and
  jailbreak (144k, ~$1.3k). ~30 GPU-h at her 0.32 s/cell with HF generate, far less with vLLM.
  Compare against the post's bar chart (Gemini-judged) and C's exact per-family JSON in
  `agu18dec/olens_ablations`. Needs `OPENROUTER_API_KEY`.
- Stretch: train the oracle on the 11-layer grid with the layer in the prompt (C's trick) so the
  full bench applies; the AR already handles it if trained with a `[Layer NN]` tag.

### Pilot first (~5 GPU-hours, $25 of compute, $50–100 with debugging idle)
Run stages 0–6 at 1/10 scale (6k transcripts, 80k pairs, 40k-start dictionary, 30k teacher
acts, 50 RL steps) on one GPU (`configs/pilot.yaml`). Purpose: every integration risk in §5
surfaces here, and the per-stage throughput numbers replace my estimates. Only then scale.
Update 2026-10-15: running on 1×H100 SXM ($3.99/h; no H200/B200 was available in US RunPod).
The reconstructor and oracle are two LoRA adapters on one shared base with an early exit at
block 42 for the reconstructor, so RL fits in 80 GB.

## 4. Cost and time (RunPod Secure, 2026-10-15 prices)

| Stage | GPUs | hours | $ |
|---|---|---|---|
| pilot (all stages, 1/10 scale) | 1×H100 | 5–10 | 25–50 |
| 0–1 corpus + activations | 1×H200 | 2–4 | 10–20 |
| 2 reconstructor | 1×H200 | 6–12 | 30–65 |
| 3–4 dictionary + teacher | 1×H200 | 2 | 10 |
| 5 oracle SFT | 1×H200 | 8–15 | 40–80 |
| 6 RL | 2×H200 | 15–30 | 160–320 |
| 7 wsbench readouts (no agentic/jailbreak) | 1×H200 | 10–30 | 50–160 |
| setup/idle/downloads | | 6–12 | 30–60 |
| **subtotal** | | | **~$440–875** |
| with realistic reruns (×1.5–2) | | | **$700–1,600** |

Wall-clock: pilot in ~2 days of my time; full run ~1 week elapsed if nothing explodes.
ceselder's own runs were 8×H200 / 4–8×B200 nodes; we are trading her wall-clock for cost.

## 5. Uncertainties found (and how each gets resolved)

| # | Uncertainty | Impact | Resolution |
|---|---|---|---|
| 1 | July `configs/oracle_lens.yaml` not published: lengths, k_max, n_min/max, ridge λ, dict_fraction, lrs, batches, RL temperature, format-penalty weights, reconstructor template, placeholder token all unknown | can't reproduce her numbers exactly | use C's and her Aug/Sep values where stated (λ=0.1, lr 1e-4 AR, r64/α16), tune the rest on the pilot; document ours |
| 2 | Which lineage Robbie wants (A/B/C) | changes stages 2–6 | decision 1 |
| 3 | Three incompatible whitening conventions across her work; FVE "anti-correlates with lens quality" per C's ablation | match criterion is only meaningful vs July | fix full-cov ridge; also report wsbench |
| 4 | Corpus thinking on/off, prompt source, temperature (paper never says; July had CoT on) | distribution shift vs benchmark | decision 3; default off |
| 5 | Injection scheme: overwrite@42 (July) vs add@block1 (Aug) vs embedding replace α=16000 (C); "a lens must be read with the mode it was trained with"; injection silently failed four separate ways for her | wrong choice = silent garbage | embedding replace (stock vLLM compatible); per-step assert that the injected norm and marker count are right; pilot compares raw vs â |
| 6 | Layer index convention (`hidden_states[43]` = output of block 42) and which class to load (wrapper vs CausalLM key remap) | off-by-one or random weights | unit test: logits of CausalLM == wrapper.language_model; hook on `layers[42]` output |
| 7 | GDN kernels: fla Triton error above ~32 seqs on some builds; B200 backward fault needs her patch; flashinfer GDN prefill JIT fails in vLLM (use triton backend) | throughput, crashes | H200 not B200 for the known-good path; chunk batches; pin fla 0.5.2 + causal-conv1d 1.7.0 |
| 8 | RL plumbing: TRL rollout_func + prompt embeds + per-step weight push (54 GB merged, or LoRA serve; vLLM 0.27.1 crashed on rank-64 LoRA for Qwen3.6) | days of debugging | pilot; fallback = lean HF-generate loop |
| 9 | NNLS vs ridge-lstsq composition, KL or not, lr 1e-4 vs 3e-5 | RL stability | NNLS (paper), β=0.02, lr 3e-5; sweep only if flat |
| 10 | RunPod: driver <580 on EU hosts breaks cu130 torch; balance ran out Oct 12; some hosts 8× slow | lost runs | `relaunch_good.sh` loop, US region, top up, checkpoint to HF every 25 steps |
| 11 | WorkspaceBench judges: Gemini via OpenRouter, Sonnet 5 for safety families; hallucination bank is on-policy so it transfers as-is | $ and keys | skip Sonnet families; need OpenRouter key |
| 12 | Licensing of her HF July code (no license file) | can we copy it? | treat as reference, rewrite; her GitHub repos are MIT |
| 13 | How her 0.638 wsbench number was judged (some Sep evals used word-matching, "not comparable to any bank_judge figure") | comparability | compare to C's JSON instead, which is fully specified |
| 14 | 31% paper FVE used unknown N,K; her 0.22 used N=8,K=10 greedy | which number to chase | chase her 0.22 at N=8,K=10; report N=4,K=10 too |

## 6. Public artifacts we will reuse (no training needed)

- `Qwen/Qwen3.6-27B` (Apache-2.0).
- `camilablank/workspace-bench` (MIT): eval harness, item banks, judges, floors.
- `camilablank/workspace-lenses` (MIT): J-lens/R-lens/template-lens for Qwen3.6-27B (sanity
  comparisons, and J[42] if we ever want her mining trick).
- `agu18dec/olens_and_ar` (Apache-2.0): C's oracle + AR, to run side by side as a reference.
- `ceselder/olens-rl-grpo-trajectory-qwen36-27b/step_00390` (MIT): B, same.
- `ceselder/oracle-lens-qwen3.6-27b`: A's merged oracle (54 GB) + reconstructor (54 GB) +
  whitening. Could be downloaded to score *our* phrases with *her* reconstructor as a
  cross-check of the FVE ruler (worth one H200-hour).

## 7. Order of work

1. Robbie answers §2. I create the repo, write the pipeline (stages 0–7 as separate scripts, one
   YAML, pytest on the FVE/OMP/whitening math against numpy references), and a SkyPilot pod YAML.
2. Pilot on 1×H200. Report per-stage throughput, memory, and the four gate numbers.
3. Full run. Push every checkpoint to HF under Robbie's account.
4. FVE match report vs her July table; qualitative Figure-64 panels; WorkspaceBench subset vs C.
5. Write-up.


## Pilot results (2026-10-15, 1×H100 SXM, ~7 GPU-h ≈ $28)

Held-out greedy FVE, N=8, K=10, 256 eval rows (`runs/pilot/*/eval_n8_k10_t0.0.json`):

| checkpoint | ours | ceselder July | gate |
|---|---|---|---|
| random phrases | 0.0011 | ~0.001 | ≈ 0 |
| OMP teacher, 16 atoms, len-8 dict | 0.118 | ~0.19 | ≥ 0.17 ✗ |
| oracle after SFT | 0.058 | ~0.11 | ≥ 0.10 ✗ |
| oracle after RL (40 steps) | **0.236** | 0.21–0.22 | ≥ 0.20 ✓ |

Reconstructor held-out cos 0.32; RL rollout FVE plateaued ~step 25 with KL to SFT ref ~1.3 nats/token
(β=0.02 is too weak or lr 3e-5 too high for a longer run). Rollouts: 41 s/step with chunked HF
generate (batch 32), so RL took 30 min. Two RL-stage bugs fixed: early-exit hook firing during the
oracle's generate, and GDN recurrent-state OOM at 128 concurrent rollouts.

Adapters (public): https://huggingface.co/collections/robbiethompson2018/nn-mats-sprint-oracle-lens-on-qwen36-27b-6ad1a5ec9f2f06b3b57a2cda

Next for a full run: scale dictionary/reconstructor data 10× before scaling RL; step-numbered RL saves
(done); a pod image with nvcc so fla/causal-conv1d kernels build; vLLM rollouts; then WorkspaceBench.

## Qwen3.8-27B pilot (2026-10-16, 1×H100, same config as the 3.6 pilot)

| held-out greedy FVE, N=8, K=10, 256 rows | Qwen3.6 | Qwen3.8 |
|---|---|---|
| random phrases | 0.0011 | 0.0015 |
| reconstructor held-out cos | 0.32 | 0.32 |
| OMP teacher, len-8 dict | 0.118 | 0.115 |
| oracle after SFT | 0.058 | 0.050 |
| oracle after RL (40 steps) | 0.236 | 0.186 |

Everything upstream of RL is identical within noise; the 3.8 RL eval was still rising at step 40 (0.183 at
25, 0.189 at 40) where 3.6 had plateaued. Single seed, ±0.03 eval noise: not a meaningful gap yet.
Adapters: `robbiethompson2018/oracle-lens-qwen3.8-27b-pilot{,-sft}`, `oracle-lens-reconstructor-qwen3.8-27b-pilot`.

## Gemma 4 31B-it pilot (2026-10-16, 1×H200, `configs/gemma4_pilot.yaml`, lens = block 40 of 60)

Raw-injection run: reconstructor cos 0.35, teacher len-8 FVE 0.128 (both a bit above Qwen), but oracle
SFT 0.015 / RL 0.028 (random 0.002). Diagnosis (`src/olens/diag_injection.py`): the SFT oracle's NLL on
teacher phrases drops only 0.085 nats/token with the correct vs a shuffled activation. Cause: Gemma's
L40 residual keeps 76% of its squared norm in one constant dimension (2041); raw unit vectors have mean
pairwise cosine 0.80, centred ones ~0, so raw injection hands the oracle nearly the same vector every
time. Qwen's massive dim (3994) is ~59% of the norm, which raw injection evidently tolerates.
Fix: `oracle.inject_space: centred` (subtract the whitening-stage mean before normalising); rerun of
SFT→RL→eval in `ckpt_oracle_{sft,rl}_centred`.
