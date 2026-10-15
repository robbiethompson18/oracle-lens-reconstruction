# ceselder's oracle-lens family on Qwen3.6-27B: what is public, how it was built, what it costs

Compiled 2026-10-15 from shallow clones of six GitHub repos and the READMEs / `nla_meta.yaml` / HF API metadata of ~30 HF model repos. All code was read, never executed.

Local copies:
- GitHub clones: `/private/tmp/claude-501/-Users-robbie-repos/d49a4578-b224-4434-8165-053b55a415f5/scratchpad/dl/ceselder_gh/{olens-1layer,skip-lens,modulation-lens,maem,vllm-metamodels,EasyNLA}`
- HF cards, sidecars, API JSON, and the July pipeline code: `/private/tmp/claude-501/-Users-robbie-repos/d49a4578-b224-4434-8165-053b55a415f5/scratchpad/dl/ceselder_hf/<repo-id>/` (the July code is under `oracle-lens-qwen3.6-27b/code/`)

Repo HEADs (all clones succeeded, nothing was private on GitHub):

| repo | HEAD date | sha | lines |
|---|---|---|---|
| EasyNLA | 2026-08-27 | d23cba0 | 14.0k |
| skip-lens | 2026-08-18 | ea9081d | 32.5k (incl. vendored eval JSON) |
| modulation-lens | 2026-09-05 | f91f20c | ~17k |
| maem | 2026-09-17 | 202b1ff | 49.8k |
| vllm-metamodels | 2026-09-11 | 6a0a0e5 | mostly bench JSON |
| olens-1layer | 2026-09-14 | 34b3810 | 15.3k (incl. data/*.json) |

HF 404s: `ceselder/skip-lens-ar-l42-jacobian-qwen36-27b` (private or deleted; "Invalid username or password"), `ceselder/modulation-lens-rl-8x1024-summaryAR-step_{25,50,100}` have no README. `olens-new-arch-lenses` and `olens-rl-grpo-trajectory` have no `nla_meta.yaml`.

**Big caveat up front.** The code that produced the actual "olens" checkpoints (pi0 best-of-32 mining with the J-lens, the online expert-distillation loop, the "jacobian L42 AR", the 4xB200 GRPO run, and the Sep "olens-new-arch" MSE/KL ARs) is **not on GitHub**. The skip-lens repo's last commit (Aug 18) predates it and contains only the L62 CNLA (compositional NLA) trainer; olens-1layer reads olens-new-arch artefacts from a read-only Modal volume. What follows reconstructs the olens recipe from HF model cards, sidecars, the CNLA trainer (which the olens RL is a direct descendant of), the olens-1layer `DESIGN.md` log, and the July `oracle-lens-qwen3.6-27b` code (which *is* fully shipped).

---

## 1. Timeline (HF `createdAt`, GitHub commit dates)

Everything below is Qwen/Qwen3.6-27B (`model_type qwen3_5`: 64-layer hybrid, 48 gated-DeltaNet + 16 full-attention blocks at i=3 mod 4, d=5120, vocab 248,320, MLP 17,408, head_dim 256, 24 heads / 4 KV). Layer index convention everywhere: "layer 42" = output of decoder block 42 = HF `hidden_states[43]`, raw residual, no norm.

| date | artefact | what | headline number |
|---|---|---|---|
| 2026-06-03 | `nanonla-l24-av-qwen3-8b` | Qwen3-8B NLA (EasyNLA origin) | ~70% FVE, 4xH200 3-4 h |
| 07-08 | `qwen3.6-27b-nla-L42`, `nla-qwen36-27b-matryoshka` | EasyNLA NLA on 27B L42 (AV LoRA r128 scope=all + AR 43-block + value head), 6xB200 | held-out FVE 74.5% @400 GRPO steps (AR warm-start 59.4%); matryoshka 64.2% full / 42.9% at 10 tokens |
| 07-14 | `qwen3.6-27b-nla-av` | merged AV | ~75% FVE |
| 07-19 | `qwen3.6-27b-nla-rl` | full GRPO trajectory, batch 504-522, group 8, 6xB200, ~440 s/step | 53% -> 78% FVE over 800 steps |
| **07-21** | **`oracle-lens-qwen3.6-27b`** | **A.9.2 replication at layer 45, overwrite injection, full pipeline code shipped** | wFVE (N=8,K=10): SFT ~0.11, OMP teacher ~0.19, GRPO full-FT 0.168, LoRA-on-top + vLLM 0.21-0.22 (peak 0.2205); random phrases 0.001 |
| 07-31 | AR data `ar_naive_fl_Qwen3.6-27B_L62` (50,925 rows) | start of skip-lens, at penultimate layer 62 | |
| 08-10 | `skip-lens-qwen36-27b-{naive-futurelens, futurelens-rl, cnla-warmstart, cnla-rl, cnla-longhorizon, ar-reconstructor, repeatafterme}` | futurelens (activation -> own continuation) + 4-bullet compositional NLA at L62 | AR 23.3% FVE; cnla-warmstart 4,850 rows (72,750-row sibling) |
| 08-13 | `skiplens-ar-L62-allidx` (dense all-idx AR, 776k spans); `skiplens-futurelens-rl-av12-kl03` | | AR 24.9% (~30% at full length); futurelens RL step 10: 33.1% FVE vs 6.6% untrained |
| 08-14 | `skip-lens-qwen36-27b-RL-surprisal` | GRPO + joint-logprob surprisal shaping; "MEANT TO BE FED A LAYER SUCH AS 42 AT TEST TIME" | held-out FVE 40.9% (step 95), peak ~42% (step 75) |
| **08-16** | **`skip-lens-olens-pi0-100k-qwen36-27b`** | **olens pi0: native L42 4-bullet oracle lens, best-of-32 seed round** | seed composed wFVE 0.4919; pi0 wFVE ~0.37 |
| 08-17 | `olens-online-step40-qwen36-27b` | online expert distillation, step 40 | wFVE train ~0.548 / held-out ~0.584 |
| 08-18 | skip-lens GitHub last commit | | |
| **08-19** | `olens-t15-n32-step30`; **`olens-rl-grpo-trajectory-qwen36-27b`** | BoN distill T=1.5 N=32; GRPO from "SFT olens iter_0000390", 39 ckpts | t15: wFVE ~0.54 "single-activation info cap"; GRPO reward 0.33 -> ~0.50 (best 0.573), "riding the ~0.55 single-activation reconstruction limit" |
| 08-20 | `skip-lens-qwen36-27b-agreelens-{a,d,g,p}(-l62)` | 4 token-matched futurelens arms at L42 (244,367 spans each), J-lens-agreement data filter | arm A: 10/21 AO-family wins on workspace-bench |
| 08-26/27 | `modulation-lens-meta-model`, `modulation-lens-4bullet` | single-phrase then 4-bullet modulation lens, replace-injection | 4-bullet NNLS 0.5144 vs single 0.482; **workspace-bench: olens step 390 = 0.638 vs modlens 0.480 vs J-lens 0.256** |
| 08-27 | EasyNLA last commit; maem `big_sft` | | |
| 08-30 | `modlens-4bullet-qwen36-27b-g16` | warm start from NNOMP over 1.58M-atom thinkies dictionary | probe 0.5063; wsbench 0.554 "= 87% of oracle-390's 0.638" |
| 09-04 | modulation-lens `rl/PROVENANCE.md` | RL regression found, root cause = unit-norm targets vs raw-scale mean; fixed, RL then +0.055 delta | |
| 09-05 | `modulation-lens-4bullet-rl-step50`; modulation-lens GitHub last commit; `maemm-uplift-*` | | delta 0.5927 |
| 09-06 -> 09-14 | olens-1layer (forked from "olens-new-arch" session) | shallow (1-4 layer, 5-70M param) verbalizers; RL against LoRA/affine/modlens ARs; 27B Karvonen reader baseline | 27B Karvonen SFT NNLS-8 wFVE 0.189 vs true-span ceiling 0.074 |
| 09-08 | `olens-new-arch-lenses` | `rl_mse`/`rl_kl`/`rl_msekl` LoRA arms (AR objective x RL reward) | no numbers on the card |
| 09-09 | `modulation-lens-grid-{A,B,C,D}` | space ablation (J-space x mean-sub) | wsbench word_matcher 0.170 -> 0.397 (cell A) |
| 09-09/10 | `qwen36-27b-maemm-inverter-pretrain-104m`, `-rl-8x2048` | MAEMM inverter, full-param FSDP2 | 104M pretrain ~387 B200-h, held-out 0.369; RL 16,384 rollouts/step -> 0.431 |
| 09-11 | vllm-metamodels last commit; `qwen36-27b-sae2m-l42` (2M-feature BatchTopK SAE, 8xB200 ~8 h) | | |
| 09-17/18 | maem last commit; `maemm-27b-rl-last16-lr5e-7` | | mean_all 0.600 (centred cosine) |
| 09-19 | `qwen3.6-27b-mlp-jlens` | J[42] composed with MLP weights | |

### Which is "the current best oracle lens" by her own description

Two independent cards (modulation-lens-meta-model, Aug 26-28; modlens-4bullet-g16, Aug 30) use **"oracle lens (step 390)" = 0.638 mean over 10 workspace-bench readout banks** as the reference generalist: "The oracle lens is the better generalist; this lens is the best of every lens tested at reading *directed* thought". That is `ceselder/olens-rl-grpo-trajectory-qwen36-27b` subfolder `step_00390` (the card's own load example). Its base policy is "SFT olens `iter_0000390` (best-of-N expert-distilled)", so "step 390" could in principle refer to either; the GRPO card and the fact that `olens-online-step40` / `olens-t15-n32-step30` are the only published distillation steps make the GRPO checkpoint the likely referent.

Her stated position on the metric: "The FVE reward is information-capped at the single-activation limit; the meaningful eval is downstream **workspace-bench**, not the reward number."

Nothing after Aug 19 claims to beat it as an oracle lens. The agreelens-A card calls itself "the strongest oracle lens on workspace-bench (10/21 AO family wins)" but that is a futurelens-SFT (no RL, no reconstructor) competing in the "AO family"; it is not the olens line.

### Explicit abandonments / negative results (her words)

- July L45 overwrite scheme: `inject_additive.py` docstring: "the overwrite-at-L45 scheme in `common.ResidualInjector` is the OLD (HF-only) path and is NOT vLLM-compatible." Every later lens injects additively at block 1 and reads L42 (or L62 for skip-lens).
- OPD (on-policy distillation) for the futurelens: `OPD8_RESULTS.md`: "At this scale, fixed-eight-token OPD is slower and does not improve held-out continuation loss, coherence, or hallucination over ordinary SFT." Repeat-after-me control: "This is a negative result".
- The "skip" hypothesis (train at L62, feed L42 at test): superseded by the native L42 olens ("no cross-layer transport at inference").
- Modulation-lens 400-step ScaleRL run: "REGRESSION. Keep the SFT." -> root-caused and fixed ("the reward was target-blind by construction"). Whitening in that reward: "WHITENING IS NO LONGER THE RIGHT TOOL and is deliberately off".
- olens-1layer: training shallow readers on AR vectors and feeding real h42 fails ("AR-INVERTER TRANSFER FAILS ... 5.945 vs 3.761"), fixed by a closed-form adapter P or by RL; KL-regularised RL on small readers "no gain ... Ceiling of the AR"; batch 16k "MUCH WORSE than batch 2048".
- 2M SAE v1: "Superseded" (loader bug streamed the same 125M tokens 8x).
- DESIGN.md: "Anthropic keys 401 (blocks any WorkspaceBench judging)" -- some Sep evals used a deterministic word matcher instead of the LLM judge and are "not comparable to any bank_judge figure".

---

## 2. The olens pipeline, end to end, with hyperparameters

There are really three generations. I give each, because the replication question is which one you want.

### 2a. July 2026: direct A.9.2 replication at layer 45 (`oracle-lens-qwen3.6-27b`, code shipped)

Stages, each a `torchrun --nproc-per-node=8` or `deepspeed --num_gpus 8` module under `code/oracle_lens/`. The YAML config (`configs/oracle_lens.yaml`) is **not** in the repo, so numeric hyperparameters below are only those visible in code or README.

0. `gen_corpus.py`: on-policy assistant transcripts. Prompts from `lmsys/lmsys-chat-1m` (fallbacks ultrachat, OpenOrca), English, 10-4000 chars, HF `generate` with `do_sample, temperature, top_p` from config; split into `recon_train, dict, teacher, rl, eval`.
1. `extract_acts.py`: residual stream *entering* block 45 (`forward_pre_hook` on `layers[45]`) at assistant-turn positions (`positions_per_transcript` sampled), stored fp16 with `next_ids = ids[pos+1 : pos+33]`. Whitening fit on `whitening.n_positions` rows of `recon_train`: `W = (Sigma + lambda I)^{-1/2}`, `lambda = ridge_frac * tr(Sigma)/d`, eigendecomposition in fp64. **Full-covariance whitening.**
2. `train_reconstructor.py`: **full fine-tune of a 27B copy** (DeepSpeed ZeRO-3, bf16, AdamW betas (0.9,0.95), WarmupLR), `Reconstructor = base + Linear(5120,5120,bias=True)` on the **last** hidden state (`hidden_states[-1]`, i.e. the full 64-layer stack, not truncated), output L2-normalised. Pairs: phrase = next N tokens after p with N~U[n_min,n_max], target = whitened unit activation. Loss `||pred - z||^2` (= 2 - 2cos). Reconstructor prompt `prompt_template.format(phrase=...)` (template string in the missing config). Optional inline FVE eval = OMP with K=16 over 20k candidate 4-token phrases on 1k held-out acts.
3. `build_dictionary.py`: ~3.4M phrases, lengths {2,4,8,16,32} tokens, from `n_starts` positions of the `dict` split, deduped per length, phrases containing newline tokens dropped; embedded by the reconstructor into fp16 unit directions.
4. `teacher_omp.py`: batched GPU **non-negative OMP**: per activation and per length L, restrict to a random `dict_fraction` of the length-L dictionary, greedily pick the atom with max positive correlation with the residual, NNLS-refit all coefficients (projected gradient, 60 iters), up to `max_atoms`; record ordered atoms, coeffs, FVE prefix.
5. `train_oracle_sft.py` (full-FT ZeRO-3) / `train_oracle_sft_lora.py` (LoRA + additive L1 injection): distill the teacher. Example: random length L, random `k in [1, k_max]`, prompt = `build_oracle_prompt(tok, n=L, k, placeholder)`, target = first k teacher phrases (exact token ids) joined by `\n`, EOS-terminated; CE on target tokens only. Prompt verbatim (`common.ORACLE_SYSTEM`, placed *after* the placeholder in a single user turn):
   > `You are an activation decoder. An internal activation vector from a language model has been placed at the marked position. Output exactly {k} phrases, each exactly {n} tokens long, that together describe the content of this activation. One phrase per line, no numbering, no extra text.`
   Injection: `ResidualInjector(model, layer=45)` **overwrites** the residual entering block 45 at the placeholder position with the **raw** activation. Option `inject_recon` ("Euan's method"): inject the in-span reconstruction `a_hat = mean + (sum c_i dir_i * ||z||) W^-1` instead of raw `a` during SFT, because raw `a` "is ~80% phrase-unexplainable residual -> noisy inverse, caps SFT at ~0.12 FVE".
6. `rl_oracle.py` (HF rollouts, ZeRO-2 + CPU-offloaded Adam, 8xH200 "policy bf16 54 GB + frozen reconstructor 54 GB per rank") and `train_rl_vllm_oracle.py` (vLLM rollouts on cuda:0, DDP LoRA actors on cuda:1..K, reconstructor on cuda:K+1; LoRA rsLoRA r64/alpha16 `all-linear` on top of the full-FT RL checkpoint; injection for vLLM = `SteeringVector(norm_match=False, overwrite=True)` at vLLM layer index 44 == HF pre-hook 45). GRPO: per step sample `prompts_per_step` activations with random N in lengths and K in [1,k_max], G rollouts each at `temperature`, `max_new = K(N+1)+slack`; reward = `fve_of(D, z)` (NNLS projected-gradient refit of reconstructor directions in whitened space) minus `format_penalty_per_phrase*|#phrases-K|` minus `format_penalty_per_token*sum|len-N|`; advantage `(r - mean_g)/(std_g + 1e-4)`; loss = -sum adv*logp over completion tokens / total completion tokens; AdamW betas (0.9,0.95), grad clip 1.0; no KL term; 512 eval activations reserved. Card: "lr 3e-5" for the LoRA stage.
7. `eval_fve.py`: greedy decode, N=8, K=10 -> held-out wFVE.

Result table from the card is in section 1. Note the paper's headline she cites is "31%"; she reached 0.22.

### 2b. August 2026: skip-lens / CNLA at layer 62 (GitHub `skip-lens`, HF `skip-lens-*`)

This is the public trainer the olens RL descends from. Base library = EasyNLA's `nla/` (Karvonen injection, NLACriticModel AR, GRPO trainers), plus `cnla/` (compositional 4-bullet reward) and `pretrain/` (rollout-grounded datagen).

**Data** (`pretrain/collect_ao_data.py`, `cnla/sbatch_dataset.sh`): `m-a-p/FineFineWeb` slices (`finefineweb_100k.parquet`, `finefineweb_xl.parquet`); per doc 5 positions chosen as high-entropy decision points (`--decision-points`, `--select entropy`; alternative `agreement` = J-lens/tuned-lens agree); at each: L62 (and optionally `--layers 62 42`) residual at the last context token, K sampled rollouts (`--rollouts 4 --rollout-len 16` for CNLA, `2x16` for the big futurelens set), top-15 next tokens, `teacher_input_ids`. `format_preview.py` turns the 4 rollouts into `"* c1\n* c2\n* c3\n* c4"` (newlines flattened) -> `finalize_cnla_av.py` writes the AV-SFT parquet (doc-disjoint 3% val, `min_bullets 2`) and the sidecar.

**AR** (`nla.train_sft --mode ar`): `init_critic_from_base` truncates the base to `ar_num_layers = layer_index+1` (63 for L62; 43 for L42), slices `layer_types`, replaces final RMSNorm with Identity (`--strip-final-norm`, recorded in `ar_meta.json`), drops `lm_head`, adds `Linear(d,d,bias=False)` value head identity-init **fp32** (bf16 ULP would swallow lr 1e-4 updates). `critic_predict`: last real token's hidden -> `normalize_activation(., mse_scale)` -> value head. `mse_scale` defaults to `sqrt(d_model)` unless the sidecar says otherwise; loss = `MSE(normalize(pred), normalize(gold))`; FVE = `1 - mse / E||v_norm - mu||^2` (paper baseline; the older "meannorm" baseline inflates FVE). `--ar-all-idx` = supervise every position after the template prefix (`critic_predict_all`) -- "reconstruct-as-you-read, no privileged read point". AR read template (sidecar `prompt_templates.ar`): `Summary of the following text: <text>{explanation}</text> <summary>`, `critic_suffix_ids [1272, 29, 361, 1648, 29]`. Published ARs:
- `skip-lens-qwen36-27b-ar-reconstructor`: 50,925 rows (`ar_naive_fl`, 2026-07-31), LoRA r64/alpha16 rsLoRA, 63 layers, **23.3% held-out FVE** ("best of the read-position/head ablation"). Format `ar_lora_value_head.safetensors + ar_meta.json`.
- `skiplens-ar-L62-allidx`: 775,997 on-policy <=12-token spans (`span_len_range [4,12]`, from `data/spans_onpolicy/av12_train.parquet`), 1 epoch, lr 1e-4, batch 32, `lora_scope=all` (target set q/k/v/o_proj, in_proj_a/b, in_proj_qkv/z, out_proj, gate/up/down_proj), **24.9% FVE** averaged over 4-12-token spans, ~30% at full length. Gotcha on the card: stale `ar_meta.json["target_modules"]`; derive from the `.lora_` keys.
- EasyNLA defaults for reference: AR lr 2e-5 (SFT), 8e-5 (co-trained in RL), LoRA r128/alpha16, attn-only scope ("the deliberate reconstruction bottleneck"); AR lr 1e-4 found best in the modulation-lens AR ablation ("1e-5 undertrains, inverted-U peak at 1e-4").

**AV warm start** (`nla.train_sft --mode av`): LoRA r64/alpha16 rsLoRA `--lora-scope all`, batch 16 x grad-accum 4 = 64, 300 steps for the 4,850-row CNLA set; the big futurelens pretrain used 5,000 steps / save every 500 and the 485k-pair naive set; `train_opd.py` arms used `--num-steps 200 --batch-size 4 --lr 3e-5`. Injection = `register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1)`: forward hook on block 1's output; at every position with `ids[p]==158983 and ids[p-1]==29 and ids[p+1]==510`: `out[b,p] = h_p + ||h_p|| * v/||v||` (`nla/injection.py::karvonen_inject_in_residual`); raises if the marker count != vector count ("let it fail LOUD instead of silently skipping injection"). Prompt built with `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` (otherwise Qwen3.6 opens an unclosed `<think>` block). Prompt in the L62 sidecars (verbatim):
> `You are shown an internal activation vector captured from a language model as it reads a passage of text. The vector, enclosed in <concept> tags, is taken at one position and encodes what the model is about to generate next. Output the text the model most likely produces immediately after this point.\n\n<concept>{injection_char}</concept>`

**CNLA GRPO** (`cnla/train_cnla_rl.py`, run via `sbatch_rl.sh` / `sbatch_fvecmp_cnla.sh` / `run_b300_cnla.sh`): `--base-ckpt Qwen/Qwen3.6-27B --quant none`, actor = PEFT adapter "default" (trainable, init from AV ckpt) + frozen adapter "reference" (the AV-SFT init, KL anchor); `--no-train-critic` (AR frozen); `--whitener data/cnla/whitener.pt`; `--num-steps 150-400`, `--batch-prompts 8 --group-size 128` (1024 rollouts/step; the Slurm single-GPU variant 8x16), `--max-new-tokens 110`, `--temperature 1.0` (asserted: on-policy surrogate), `--kl-beta 0.1` (k3 estimator with delta clamped at 12; default would be 0.01), `--lr 3e-5`, AdamW8bit betas (0.9,0.95), grad clip 1.0, `--logp-micro-batch 2-16`, save every 25, eval every 50 (`base_fve` + `text_judges`), hinged length penalty 0.01/token past `max_new_tokens-64`. Advantage = pure on-policy REINFORCE (`advantage * new_logp`, no ratio/clip) + beta*KL.

**CNLA reward** (`cnla/reward.py`, `cnla/cnla_rl.py`) -- this is "the whitened compositional FVE":
- Bullets are split **on the token stream** (a new bullet starts at any token whose decode contains `*`, max 4), each bullet's text = decode of its own span, so reward and advantage share tokenisation.
- Each bullet -> frozen AR -> vector; target `g = L2norm(h)`; vectors L2-normed; both whitened by a **diagonal** whitener `w(x) = (x - mu)/sigma` fitted (`fit_whitener.py`, 20k rows) on L2-normed activations.
- Composition: `alpha* = argmin ||g_w - V_w alpha||^2` solved as ridge normal equations (`(V^T V + (1e-3*mean diag + 1e-6) I) alpha = V^T g`, `torch.linalg.solve`) -- **unconstrained least squares**, not NNLS (README text says "NNOLS-select"; the trainer's reward is lstsq). `FVE = 1 - ||g_w - h_hat_w||^2 / ||g_w||^2` (whitened target is already centred).
- Per-bullet reward `r_i = FVE(all) - FVE(all \ i)` (leave-one-out marginal); optional `--loo-threshold` (`clamp(r - t, max=0)`: only redundant bullets get negative signal, "anti-KL-blowup") and `--missing-bullet-penalty` (negative x (4 - n_valid), added pre-baseline). GRPO baseline `(r - mu)/(sigma+1e-6)` over the group's valid bullets, clipped +-10, scattered onto each bullet's token span (0 outside bullets).
- Logged: `cnla/fve_full_mean`, `marginal_mean`, `n_bullets_mean`, `frac_4bullets`, `uniqueness_mean`.

### 2c. August 2026: the olens proper (native L42, best-of-N expert iteration, GRPO)

Reconstructed from the pi0 / online / t15 / GRPO cards and sidecars (all share `dataset_id olens_av, row_count 100000, layer_index 42, norm none`):

1. **Verbalizer init** = `skip-lens-qwen36-27b-RL-surprisal` (L62 futurelens; init = AV-SFT on <=12-token on-policy spans; GRPO group 16 x batch 64 = 1024 rollouts/step, `max_new_tokens 12`, lr 1e-4, KL beta 0.03 k3 to the SFT init, frozen AR reward -normalised MSE + joint/sum-logprob surprisal shaping with lambda set adaptively so its GRPO "influence ratio" is pinned at ~10%; step 95 of 100; held-out FVE 40.9%). Card: "THIS CHECKPOINT IS MEANT TO BE FED A LAYER SUCH AS 42 AT TEST TIME".
2. **Harvest** 100k L42 activations from FineFineWeb at the last real token.
3. **Candidate mining**: "Drive the RL-surprisal AV with the workspace J-lens (`J[42]->L62`, `camilablank/workspace-lenses`) to mine 32 rollouts / activation" -- i.e. feed `J[42] h42` (an L62-space transport) into the L62-trained AV, 32 samples. The t15 variant: temperature 1.5, N=32.
4. **Teacher selection**: "Score every candidate span through the jacobian L42 AR (`skip-lens-ar-l42-jacobian-qwen36-27b`, private) against the raw L42 activation; whitened NNOLS-select 4 bullets (FVE-ordered)." Seed composed wFVE **0.4919** over 100k. The code for this is not public; the closest public implementations of greedy NNOMP-select-4 are `modulation-lens/src/mine_bullets.py` (16 high-T samples, greedy NOMP on the composition objective with exact enumerated NNLS for k<=4) and `modulation-lens/stage2_nnomp.py` (batched NNOMP against a dictionary, exact NNLS by subset enumeration).
5. **SFT** a copy of the RL-surprisal checkpoint on (raw L42 -> 4 bullets): 1 epoch, 100k rows, effective batch 256 -> **pi0**, wFVE ~0.37 ("~75% of the seed teacher"); "raw FVE 0.62-0.84 on real contexts". LoRA r64/alpha16 rsLoRA.
6. **Expert iteration** ("Iteration (pi1...) closes the 0.37->0.49 gap"): online best-of-N distillation from pi0, 4-GPU DDP; published steps: `olens-online-step40` (wFVE ~0.548 train on fresh acts, ~0.584 held-out) and `olens-t15-n32-step30` (T=1.5, N=32, wFVE ~0.54, "single-activation info cap"). The SFT lineage continues to "iter_0000390".
7. **GRPO** (`olens-rl-grpo-trajectory`): base = SFT olens `iter_0000390`; reward = "whitened compositional FVE (how well the 4 emitted bullets, re-encoded through a frozen jacobian-L42 autoregressive reader, reconstruct the injected raw L42 activation)"; temp 1.0; **16 activations x 64 rollouts/step = 1024**; 400 steps; **4xB200 DDP, bit-synced** (`n_per_rank [256,256,256,256]`); lr **1e-4** ("carried over from the distillation config; high for RL (a 3e-5/1e-5 control is planned)"); `kl: 0.0` every step in `rl4gpu_fve.jsonl`; `wall_s` 92-105 s/step (`mine_s_max` ~24-34 s of that is the mining/scoring phase); reward 0.335 (step 0) -> 0.48-0.54 (steps 390-399), best 0.573; ckpt every 10. Load: `PeftModel.from_pretrained(base, repo, subfolder="step_00390")` (PEFT 0.20.0).

**Prompt (verbatim, single user turn, no system message, chat template with `enable_thinking=False`)**:
```
You are shown an internal activation vector captured from a language model as it reads a passage of text. The vector, enclosed in <concept> tags, is taken at one position and encodes what the model is about to generate next. Output four bullet points, each starting with '*', describing what the model is most likely about to generate.

<concept>㈜</concept>
```
(`㈜` = U+321C, token id 158983; left neighbour id 29 = `>`, right id 510 = `</`; the marker is at position ~40 of a ~186-token rendered prompt.)

**Injection recipe**: Karvonen norm-matched add at the **output of decoder block 1**: `h'_p = h_p + ||h_p|| * v/||v||`, `v` = raw L42 activation (no mean-centring, no whitening, no cross-layer transport). The pi0 card insists: "do not replace it with text"; "The jacobian is used only in mining; the olens reads raw L42."

**FVE definition on the cards**: "whitened fraction-of-variance-explained of the raw L42 activation vs the reconstruction of the emitted 4-bullet composition (L2-normalize, per-dim whiten, then `1 - ||h_w - h_hat_w||^2 / ||h_w||^2`)" -- i.e. the `cnla/reward.py` definition with a diagonal whitener, composition presumably the same ridge-lstsq.

**Sampling at inference** (from the related cards): `do_sample, temperature 0.7-1.0, top_p 0.95-1.0`, `max_new_tokens` ~12 per bullet; the CNLA cap was 110 tokens for 4 bullets.

### 2d. September 2026: "olens-new-arch" and olens-1layer (what changed in the recipe)

Not on GitHub except via olens-1layer, which documents the shared artefacts:
- Harvest (`src/harvest.py`, `harvest_fresh.py`): rows = (ctx ids, L42 at last ctx token, 12-token on-policy rollout at T=1 top_p=1 top_k=0 with EOS suppressed, top-128 teacher log-probs). Contexts 32-256 tokens, no BOS, from the 25.7M-token maemm corpus or fresh FineFineWeb files (disjoint by file). Harvests v1-v6 total ~84M rows (v5 = 24M, v6 = 48M); 96 shards x 250-500k rows per harvest; "Policy: NEVER epoch; data is infinite".
- AR `ar_mse_v3`: LoRA (r64/alpha16 rsLoRA, regex-targeted blocks 0..42 only: `lora_target_re(max_layer)`) + `value_head.pt` `Linear(5120,5120,bias=True)`; forward early-exits at block 42 (`Layer42Hook.capture_grad` raises `StopForward`), MSE objective ("all-pos" variant exists). Measured on the common file: raw FVE 0.21, cos 0.79, centred cos 0.45, whitened-centred FVE 0.065; "the MSE AR is a heavy shrinker to the span-conditional mean" (keeps ~13% of per-dim variance). Alternatives: `rl_kl` (downstream top-k tail KL via `topk_tail_kl`) and `rl_msekl` arms (`olens-new-arch-lenses` card: AR objective {raw MSE, downstream KL, MSE+KL} x RL reward {whitened FVE, -downstream KL}); closed-form **affine ARs** (ridge from frozen block-42 span states: last 0.088-0.091, last+mean 0.140-0.144, all-12 0.154 FVE; "an affine map gets ~2/3 of the LoRA AR"); embedding-only affine AR 0.050 ("too weak as a cheap reward; the 27B forward stays").
- Whitener `whitener_v2.pt`: **full-covariance** `W = V diag(1/sqrt(max(lambda, floor))) V^T`, floor = frac*lambda_max, fitted on raw harvested L42 activations (`common.Whitener`; `fve = 1 - ||W(pred-gold)||^2/||W(gold-mu)||^2`). So Sep uses full whitening again (as July), Aug used diagonal.
- Reward (`rl_inverter.py --reward-fn wfve`): "paper A.9.2: whitened FVE after NNLS refit (K=1 -> `max(0, cos_w)^2`) using whitener_v2". Eval (`eval_olens_rl.py`, `eval_olens_27b.py`): "paper-style inference: sample K=8 phrases -> AR -> `scipy.optimize.nnls` in whitened space -> wFVE", plus greedy / mean-sampled / best-of-8 and the 27B's NLL of the generated span (naturalness / hacking detector).
- RL recipe = "MAEMM ScaleRL bundle": CISPO (eps_max 5), batch-level advantage normalisation with zero-variance groups dropped, prompt-level loss aggregation, fp32 lm_head, AdamW betas (0.9,0.999), group 8-16, **no KL**; 2048 prompts x 16 samples per step on 4 GPUs (512x16 per rank); lr 1e-5 (small readers) / 1e-5 (27B LoRA, 256x8, 600 steps vs modlens reward).
- K=1 nearest-dictionary teacher (`seed_teacher.py`, "paper A.9.2 stage 3, K=1"): dictionary = AR vectors of 4.5M harvested spans; nearest by whitened cosine; mean best wFVE 0.155 vs true-span 0.078 over 754k activations.
- 27B Karvonen reader baseline (`train_av27b.py`): rsLoRA r64/alpha16 on all blocks, prompt `build_av_prompt` ("...<concept>㈜</concept>\n\nWrite the text that most likely follows."), trained as AR-inverter on 12M cached (AR(span), span) rows, eff batch 256, lr 3e-5 (1e-4 default in code), 41k steps on 8 GPUs; SFT-only step 5000: greedy wFVE 0.1185, NNLS-8 0.189 (ceiling 0.074 = true span), final inverter CE 1.134. Then RL'd vs the modulation-lens reward: delta 0.165 vs the modulation oracle's 0.099.

Shallow-verbalizer findings worth stealing: data > depth ("single pass over more data is THE lever"; CE still falling at 84M rows), batch 2048 >> 16k for small readers, GDN blocks read h42 as well as attention, 27B init worth -0.35 nats vs random, 1 head costs ~0.8 nats.

### 2e. Modulation lens (Aug 26 - Sep 11)

Different objective, same injection scaffolding. Single-phrase prompt (verbatim in `prompt_oneline.txt` / card), 4-bullet prompt on the `modulation-lens-4bullet` card. Scoring: each phrase/bullet is placed in a grid of 6 templates x 6 carriers (`inv_core.TEMPLATES_RECOVERED`, e.g. `Repeat this line verbatim: "{y}"\n\nAs you write it, focus on something related to: "{x}". Output only the sentence.`), L42 read at the forced carrier positions, mean over all 36 cells, projected by `J[42]`; two-mean centring (PMU = grid mean, AMU = activation-pool mean; "one shared mean puts a blank string at 0.259, two at 0.008"); exact NNLS by subset enumeration; reward cos(composition, target), plain cosine (whitening "costs 6.3x"), contrastive negative. Warm starts: self-distilled mined bullets (`mine_bullets.py`) or NNOMP top-4 over the 1.58M-atom "thinkies v3" dictionary (`mine_nnols_dict.py`, `--dedupe-cos 0.9`; "the dictionary targets are ~76% noise -- only 0.95 of 4 atoms per activation are genuinely about the text"). The frozen text->vector AR (`ar_train_lora.py`: LoRA r64/alpha16 on blocks 0..42, mean-pool, linear head, cosine loss, lr 1e-4, batch 64x8, <=12-token atoms, held-out cos 0.928 vs ridge 0.653) replaces 16 grid forwards per bullet with one, and "needs its OWN backbone truncated to read_layer+1" (full 64-layer stack halves the reward). RL: vendored maem `rl_disagg.py` with ScaleRL, 16x256 rollouts, 8xB200 (2 rollout + 6 trainer), lr 1e-5 -> 3e-5, KL 0 -> 0.1. Injection mode = **replace** (`h'_p = v`, raw magnitude, ~3.8x local block-1 norm) for the Aug lenses, Karvonen for g16; "a lens must be READ with the mode it was TRAINED with" (mismatch costs 34% of the delta; earlier card said "indistinguishable, cos 0.982").

### 2f. MAEMM (Aug 27 - Sep 18)

Inverse problem (direction -> text whose own L42 activation points along it), not a lens. Marker ` ?` appended after the generation prompt, inject `h + coeff*||h||*unit(v)` at block 1, reward = max-over-token cosine of the clean base's L42 re-read (last-5 or last-16 window), ScaleRL/CISPO full-parameter FSDP2 at lr 1e-6 / 5e-7, 8 prompts x 2048 rollouts/step; pool = real activations + 2M-SAE enc/dec rows + probes + BSF. Relevant to a lens replication only as the RL/infra donor (ScaleRL bundle, disaggregated vLLM rollouts, prefix cache, full-param publish protocol).

---

## 3. Compute

No dollar figures anywhere in code, cards, or notes. What is stated:

| stage | hardware | wall-clock / notes |
|---|---|---|
| July A.9.2 pipeline | 8xH200 node (`torchrun --nproc-per-node=8`, DeepSpeed ZeRO-3 for the two full-FT 27B copies; RL ZeRO-2 + CPU-offloaded Adam: "policy bf16 54 GB + frozen reconstructor 54 GB per rank"; vLLM path: 1 engine GPU + K actor GPUs + 1 reconstructor GPU) | not stated; "~0.5 s" LoRA sync per step |
| July NLA (EasyNLA) | 6xB200, batch 504 prompts, group 8 | ~440 s/step; 800 steps ~ 4 days |
| skip-lens datagen | Slurm `general` partition, 1 GPU/job, `--array=0-5` x 16,667 docs, 6 h limit each | ~500k pairs |
| skip-lens AV/AR SFT | 1 GPU (H200-class), 4 h limit | |
| CNLA / futurelens RL | 2 GPUs `device_map auto` (8 h limit, 8x128/step) or 1 B300 (275 GB: "actor+AR ~108 GB + a big backward") or 1 GPU 8x16 (14 h limit); vLLM configs for 6xB200 (`batch_prompts 288`, `vllm_gpu_mem 0.40`, `logp_micro_batch 8`) and 6xH200 (fp8 vLLM weights, `batch_prompts 144`) | |
| olens GRPO (step 390 lens) | **4xB200 DDP**, 1024 rollouts/step | 92-105 s/step x 400 ~ **10.5 h**; mining/scoring 24-34 s of each step |
| olens online distillation | 4-GPU DDP | not stated |
| olens-1layer harvests | Modal B200 x1 per shard, 96 shards/app ("Modal cap: 100 ephemeral apps/workspace") | ~3 h for 48M rows |
| olens-1layer readers | Modal 4-8xB200 DP, eff batch 256-2048, 15k-110k steps | "~0.03 s/step" for 1-layer; "8-GPU jobs queued for >1 h" on Modal; AR-forward-bound inverter runs "~12 h" -> precomputed AR cache |
| olens-1layer RL | 4xB200 (512x16 per rank); the 27B reward forward dominates ("8192 spans x 12 tok x 43 layers per rank per step") | |
| modulation-lens RL | Modal 8xB200 (2 vLLM rollout + 6 HF trainer), 16x256 | **104 s/step at ~7.5% MFU**; 88% of generation is prefill because the marker sits at position 40 of 282 (moving it to the end would make 86% a shared prefix, "needs SFT retrain") |
| modulation-lens AR | 1xB200 | |
| MAEMM pretrain | 8xB200 FSDP2, batch 4096, 25,391 steps | **~387 B200-hours** (6.5-9.1 s/step), 2026-09-07->09 |
| MAEMM RL | 8xB200 (2 vLLM + 6 FSDP2 trainers) | 68 s/step full-param (micro-batch 8) vs 30 s LoRA; 300 steps |
| MAEMM LoRA RL (earlier) | 5xB300 box, then 4-GPU gloo DDP | 223 -> 59 s/step; "~6.5 h (was ~25 h)" for 400 steps |
| 2M-feature SAE | 8xB200 | ~8 h |

Memory facts for a 27B replication: a B200 shows 178 GB usable (H200 141, B300 288); actor bf16 ~54 GB, 43-layer AR ~36-38 GB, vLLM copy ~54 GB (27 GB fp8); the `[mb, seq, 248k]` bf16 logits are the dominant training transient (chunked CE fixes it); FSDP2 full-param peaks 151 GB/rank at micro-batch 8.

---

## 4. Dependency stack and reproducibility pain

**Pins seen (by project):**
- EasyNLA: `transformers==4.57.1` (Qwen3-8B era; "v5's apply_chat_template break crashes the trainers"), own venv with `vllm==0.19.0 vllm-lens==1.1.0 --torch-backend=cu128` (driver >=570), `utils/patch_vllm_lens.py` mandatory (unpatched vllm-lens norm-matches against the partial residual -> ~50x too weak injection -> 42% clip-frac divergence), `peft`, `bitsandbytes` (AdamW8bit / nf4). "vLLM 0.22+ refactored GPUModelRunner -> hook silently skips injection".
- skip-lens (27B): `transformers>=5.5.1,<5.6` (qwen3_5 hybrid needs >=5.5.1), `flash-linear-attention` (GDN Triton kernels), `causal-conv1d`, `mnemonic`, FastAPI; vLLM RL needs `vllm_name_map: qwen3_5_wrapper` ("vLLM registry has no text-only Qwen3_5ForCausalLM"), `vllm_attn_backend: FLASH_ATTN` ("auto-picked FLASHINFER JIT-compiles and dies"), `ar_lora: true` (full-FT critic + Adam ~+66 GB doesn't fit). The July NLA card adds a required patch `utils/patch_transformers_qwen35_gva.py`: the fla gated-delta-rule backward "faults (misaligned address) on B200 when q/k are expanded to 48 heads".
- olens-1layer Modal image: `torch==2.8.0 transformers==5.5.4 peft==0.19.1 flash-linear-attention`.
- modulation-lens / maem Modal images: `torch==2.10.0 (cu128) vllm==0.19.0 vllm-lens==1.1.0 transformers==5.15.0 peft==0.20.0 accelerate==1.14.0 flash-linear-attention==0.5.2 wandb==0.28.2 numpy==2.4.6 safetensors==0.8.0 huggingface_hub==1.27.0 tokenizers==0.22.2`; the modlens AR was trained under `transformers==5.5.4` ("a transformers-version difference is what produced the 0.331-vs-0.759 backbone surprise -- run `calibrate` in THIS image before trusting the reward"). vLLM needs `attention_backend=TRITON_ATTN` ("FLASHINFER silently no-ops") and `language_model_only=True`; a vllm-lens install "missing its dist-info never registers its vLLM plugin entry point, so steering silently no-ops". Hopper needs `DISAGG_TRAINER_TRITON=3.7.1` (fla 0.5.2 raises on Triton 3.4-3.7.0).
- **transformers fork** `ceselder/transformers@e52940e567ab9a991a1c971c1094e340233baff3` = v5.15.0 + one commit to `cache_utils.py` (autograd-safe linear-attention state writes + batch expansion) -- required for `--prefix-cache` in maem/modulation-lens trainers and `sft/pretrain.py --prefix-share-step`. Optional, but the fast trainer paths assume it.
- **vLLM fork** `ceselder/vllm-metamodels` = `vllm-lens 1.1.0.post7` (drop-in, same dist name): indexed steering hook (37.8x faster at B=2048 on the 27B), CUDA graphs with steering (`VLLM_LENS_CUDA_GRAPHS=1`), correct full-residual `norm_match` (upstream 1.1.0 was ~8x too weak on Qwen3.6 layer 1), `mode="replace"`, in-engine `ReadoutVector` + early exit (layer-42 readout 4.7 s / 1,024 texts), `merge_lora` on publish (-29% decode), salted prefix caching, `VLLM_LENS_COMPILE=1`, shared-memory transport; tested on vLLM 0.16/0.19/0.27.1/0.28; uv.lock: vllm 0.19.0, torch 2.10.0, transformers 5.5.4, peft 0.18.1. Caveats it documents: Qwen3.6 GDN gets 0 prefix-cache hits ("Qwen3Next currently does not support 'all' prefix caching"); keep `max_num_seqs <= 1024` with CUDA graphs on 0.19 (GDN decode kernel grid limit); vLLM 0.27.1 + Qwen3.6 + rank-64 LoRA crashes vLLM's own LoRA path (merge instead); compiled 27B engine start-up 778 s on 0.19.
- External artefacts: `camilablank/workspace-lenses` (`qwen3.6-27b/j-lens/lens.pt`, `J[42]`) for the J-lens; `camilablank/workspace-bench` (MIT) for evals; Anthropic API (Sonnet 5 / Opus judges, Sonnet 4.6 gold explanations for the NLA warm start); `m-a-p/FineFineWeb`, `openbmb/Ultra-FineWeb`, `HuggingFaceFW/fineweb(-edu)`; Modal (`maemm-data`, `olens-new-arch`, `celeste-modlens-vol`, `olens-1layer` volumes; secrets `maemm-hf`, `maemm-wandb`; profile `safety-sahan`); a Slurm cluster (`/workspace-vast/celeste/...`) and a B300 box for the Aug work.

**Painful to reproduce:**
1. The olens mining/distillation code and the "jacobian L42 AR" are not public; the GRPO reward used that private AR. You can rebuild an equivalent AR from `skip-lens`/EasyNLA at L42 (43-layer truncation) but will not reproduce her numbers exactly.
2. Qwen3.6-27B is a multimodal wrapper with hybrid GDN layers: every stack needs the right transformers (>=5.5.1), fla kernels + the B200 backward patch, a text-only conversion for vLLM (`qwen3_5_wrapper` name map or a converted `Qwen3_5ForCausalLM`), and no prefix caching in vLLM. 248k vocab makes full-logit tensors the memory wall (chunked CE everywhere).
3. Injection correctness has silently failed for her at least four separate ways (vllm-lens partial-residual norm, missing dist-info, FLASHINFER, V2 model runner); every trainer now verifies `steer_apply_count` and CJK fraction per step. Use her fork and its checks.
4. The July config YAML is missing; the July reconstructor is a full-FT 27B (two 27B copies resident) -- an order of magnitude heavier than the Aug/Sep LoRA-on-truncated-backbone ARs.
5. Three different whitening conventions across Jul (full-cov ridge), Aug (diagonal per-dim on L2-normed acts), Sep (full-cov eigen-floor on raw acts) and two composition solvers (ridge lstsq vs NNLS). Numbers are not comparable across them; pick one and state it.

---

## 5. Known issues, bugs, TODOs (from code comments, cards, logs)

- olens GRPO lr 1e-4 "high for RL (a 3e-5/1e-5 control is planned)"; KL = 0 throughout.
- `skiplens-ar-L62-allidx`: stale `ar_meta.json["target_modules"]` ("derive from the `.lora_` keys").
- RL-surprisal card: "KL had drifted to ~1.5" by step 95; step 75 is the better checkpoint. futurelens-rl-av12-kl03: "Later steps over-optimize (KL climbs, held-out plateaus) -- iter 10 is the peak."
- `train_cnla_rl.py`: 4-bit AR LoRA cannot be used on a bf16 backbone (asserted); `temperature` must be 1.0; resume detection cannot catch a partial PEFT adapter load; `--loo-threshold` / `--missing-bullet-penalty` exist because plain LOO marginals caused "KL blowup" and sub-4-bullet truncation.
- modulation-lens `PROVENANCE.md`: bank rows L2-normalised for steering were fed to a reward that subtracts a raw-scale mean (||amu||=55 vs ||h||~24) -> "the objective REWARDED ignoring the activation 3:1"; the injection self-check used a random vector and "reported cos=1.0000" while the reward was broken; T=1.0 reward curves vs greedy baselines are not comparable; "Do not select on the reward alone" (readability fell 0.050 at flat reward); per-bullet token cap needed (policy "expands to fill any global cap"); step cost dominated by redundant prefill (marker position).
- modlens-4bullet-g16: in-run probe is only 32 activations (SEM ~0.027); "SFT loss is not a proxy for lens quality".
- DESIGN.md: NCCL allreduce timeout from per-rank step caps (fixed with all_reduce MIN); OOM from one rank loading 3M rows (`--max-files`); fp32 master x bf16 frozen needs autocast inside forward; "overfit" was data re-use not a model property; cache-fed checkpoints stamped the wrong AR path (fixed with `--input-ar`); v3 "common" test file overlaps the old corpus (optimistic by 0.2-0.35 nats) -> use the fresh v5 file; wandb entity changed mid-project; Modal 100-app cap; "Anthropic keys 401".
- `olens-rl-grpo-trajectory`: AR not shipped at all; `olens-new-arch-lenses`: no numbers, "See the live playground for usage".
- NLA-rl card: AR checkpointed once at the end, so per-step FVE is not exactly reproducible; adapter `base_model_name_or_path` points at the wrong place.
- July code: Qwen3.6 fla/tilelang kernels throw "Triton Error [CUDA]: invalid argument" above ~32 seqs/launch (`gen_prompts_per_call`, `recon_batch` chunking); `PROBE.txt` is a 1-word placeholder; `save_consolidated` has a dead `if False`.
- EasyNLA `train_rl_vllm.py`: IPC weight sync needs the legacy CUDA allocator (`PYTORCH_CUDA_ALLOC_CONF` unset); `--av-adapter` warm start matters (~12 pp FVE cold start otherwise); masking truncated rollouts "removed the anti-runaway gradient -> observed policy collapse, twice".
- maem: NCCL deadlocks on some boxes (gloo for LoRA grads); warm-starting RL from a deep-RL ckpt collapsed; length penalty 0.25/tok caused every collapse (use 1.0); un-gated diversity bonus collapsed; SAE stream-sharding bug.
- vllm-metamodels: vLLM >=0.27 hidden states are batch-composition sensitive at bf16; DeepSeek-V4 CUDA-graph log-probs depend on batch composition; `merge_lora` TP>1 untested.

---

## 6. Mapping to the Anthropic oracle-lens recipe (reconstructor -> phrase dictionary -> NN-OMP teacher -> oracle SFT -> RL)

| Anthropic stage | July `oracle-lens-qwen3.6-27b` | Aug olens (pi0 -> GRPO step 390) | Sep olens-new-arch / olens-1layer | modulation lens |
|---|---|---|---|---|
| Reconstructor (phrase -> whitened activation) | faithful: full-FT 27B copy + affine head on the last hidden state, unit-whitened target, loss 2-2cos, trained on next-N-token phrases N in [n_min,n_max] from on-policy transcripts | **deviation**: EasyNLA AR = base truncated to K+1 layers + fp32 `Linear(d,d)` on the `<summary>` anchor, **raw** (unwhitened) normalised-MSE objective, trained on <=12-token on-policy spans; dense `--ar-all-idx`; the olens reward AR ("jacobian L42") is private | LoRA-on-blocks-0..42 MSE AR (`ar_mse_v3`), plus KL/MSE+KL arms and closed-form affine ARs; eval in full-cov whitened space | text->modulation-vector AR (mean-pool, cosine loss, J-space), a different quantity |
| Phrase dictionary | faithful: ~3.4M phrases, lengths {2,4,8,16,32}, embedded once | **none** -- candidates are 32 on-policy rollouts of the RL-surprisal futurelens driven through `J[42]` | K=1 nearest-neighbour over 4.5M harvested spans (`seed_teacher.py`) | 1.58M-2.9M-atom "thinkies" dictionary of <=12-token spans |
| NN-OMP teacher | faithful: greedy positive-correlation selection over a random half of the dictionary, NNLS refit each step, up to `max_atoms` | **expert iteration instead of OMP**: NNOLS-select 4 of the policy's own 32 samples (greedy on the composition objective), then online best-of-N distillation (T 1.0-1.5, N 32) | K=1 NN teacher; "Lens stage (NNOMP -> SFT -> contrastive RL) still TODO" | NNOMP top-4 over the dictionary (`stage2_nnomp.py`), the closest public OMP-teacher code |
| Oracle SFT | faithful: K in [1,k_max] phrases of exactly N tokens, one per line, prompt states N and K; overwrite-inject raw `a` at L45 (option: inject the OMP reconstruction `a_hat`) | **bullets, not verbatim N-token phrases**: fixed K=4 `*`-bullets "describing what the model is most likely about to generate", free length (~27 tok/bullet cap), additive norm-matched L1 injection of raw L42 | shallow readers on (h42 -> 12-token span); 27B Karvonen reader as AR-inverter | 4 bullets "the separate things this state is holding in mind", replace injection |
| RL | faithful: GRPO, reward = whitened FVE after NNLS refit - format penalties, no KL; vLLM rollouts + LoRA stage added | GRPO, reward = whitened compositional FVE (ridge-lstsq composition, diagonal whitener), per-bullet leave-one-out advantages on bullet spans (CNLA), KL 0, lr 1e-4 | ScaleRL/CISPO, reward = `max(0,cos_w)^2` (K=1 NNLS wFVE) or modlens delta, no KL | ScaleRL, contrastive NNLS cosine in J-space, frozen AR |
| Injection | overwrite at L45 (pre-hook), later additive L1 for vLLM parity | **Karvonen additive, norm-matched, block-1 output** (`h + ||h|| v_hat`), `㈜` inside `<concept>...</concept>` | same | replace (`h'=v`) for Aug lenses, Karvonen for g16 |
| Headline | wFVE 0.22 (paper 0.31) | wFVE ~0.55-0.58 cap; workspace-bench 0.638 | 27B Karvonen SFT NNLS-8 wFVE 0.189 (ceiling 0.074) | wsbench 0.480-0.554 |

Her reading of why the Aug deviations: the reconstructor-only objective is "information-capped at the single-activation limit" (~0.55 whitened FVE for one activation at L42), so she moved the goalpost to workspace-bench readouts and to lenses whose output is a description of upcoming text rather than verbatim phrases; the expert-iteration seed exists because a phrase dictionary built from the model's own continuations at L42 gave 0.49 composed FVE directly, which no SFT-from-OMP in July reached (0.11 -> 0.22).

---

## 7. Open questions for a replication (things the public material does not settle)

1. Exact definition and training data of the "jacobian L42 AR" that scored the pi0 seed and the GRPO reward (private HF repo). Best guess from naming: an L62-style AR whose target is `J[42]`-transported L42 activations, or an AR trained at L42 with the Jacobian as a feature; the Sep work replaces it with `ar_mse_v3` (plain MSE LoRA AR on blocks 0..42).
2. The online expert-distillation loop (N, T, how many rounds, how "iter_0000390" relates to step counts, whether the teacher was refit each round) -- only two snapshots are published.
3. Whether the GRPO composition used NNLS or the CNLA ridge-lstsq; the cards say "NNOLS", the only public trainer says lstsq.
4. The July config (`lengths`, `k_max`, `n_min/n_max`, `ridge_frac`, `dict_fraction`, `max_atoms`, lrs, batch sizes, `format_penalty_*`, temperature) -- not shipped.
5. Workspace-bench judging for the 0.638 number (Opus judge prompts live in skip-lens `evals/judge_*.py` and camilablank/workspace-bench; the Sep ablations fell back to word matching).
