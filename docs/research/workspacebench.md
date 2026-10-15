# WorkspaceBench + the camilablank/agu18dec Oracle Lens: what you need to replicate and evaluate

Sources read (all local copies under `/private/tmp/claude-501/-Users-robbie-repos/d49a4578-b224-4434-8165-053b55a415f5/scratchpad/`):

- AF post "WorkspaceBench: Evaluating Interpretability Methods for the Global Workspace" (Blank, Bhatia, Ong, Nanda, 23 Sep 2026): `dl/af.txt`, `dl/af.html`, figures in `dl/img/`.
- Links in the post: "We open-source our benchmark **here**" -> https://github.com/camilablank/workspace-bench ; "Oracle lens training details (with **checkpoint**)" -> https://huggingface.co/agu18dec/olens_and_ar . All other outbound links go to transformer-circuits.pub (J-lens / template-lens / oracle-lens appendix), arXiv, or alignment.anthropic.com/2026/chive.
- Shallow clone of the repo at `dl/workspacebench/` (main as of 2026-10; the in-house runs cite commit `92d763e`). Code read, not executed.
- HF metadata + READMEs in `hf/`: `agu18dec/olens_and_ar`, `agu18dec/oracle-lens-ddp600`, `agu18dec/olens_ablations` (8,049 files, 872 GB), `agu18dec/mOre-lenses`, `camilablank/workspace-lenses`, `neuronpedia/jacobian-lens`, plus the ab4 results CSVs and the per-family/per-layer breakdowns of the published checkpoint.
- Earlier-downloaded ceselder material in `dl/ceselder_*` for the comparison section.
- Anthropic oracle-lens appendix text: `oracle_appendix_folded.txt` (from `dl/workspace.html`).

Cell counts in section 2 were computed by `scripts/count_cells.py` (my re-implementation of `readplan.py` over the frozen banks with the Qwen3.6-27B tokenizer, run with `python -I`).

---

## 1. What WorkspaceBench measures

### 1.1 Shape

- 27 eval families, **3,356 items** (my count from the banks: 3,057 in `evals/*/items.json` + 299 captured activations in `evals/jlens_concept_pr/manifest.json` = 3,356, matching the post).
- Model under test: Qwen3.6-27B (64 decoder blocks, d_model 5120). Every bank was gated on it: "The model gets the answer right >80% of the time with n=10 rollouts at temp=0.8 without chain-of-thought. The intermediate never appears in the prompt, as a numeral or a word. The model doesn't state the intermediate in its own continuations."
- Terminology (README): *O-lens* = prose readout (samples); *J-lens* = top-10 token bag with scores; *arm* = one lens/checkpoint run over a family; *cell* = (item, layer, position); *floor* = frozen lucky-guessing and prompt-only baselines.
- Layer grid: `GRID = range(20, 61, 4)` = 11 layers (20,24,...,60). Association families use `SIX = [20,28,36,44,52,60]`; hallucination and jailbreak use `FIVE = [20,36,44,52,60]`; agentic reads `GRID + [63]`; arithmetic reads only `[56, 60]`; buggy_code one layer (60 python / 56 other); multi-concept DM `[44,52,56,60]`. "A benchmark layer L is the output of decoder block `model.layers[L]` (the residual stream after that block)" = `hidden_states[L+1]` in HF terms.
- Pass rule: "Across all evals, a method is read at a set of (layer, position) cells, and it passes if any cell passes. We make sure the judges only see the readout, not the prompt, unless noted."

### 1.2 The 27 families (grouping per the repo README; `*` = has a single-token subset per the post)

| group | family | items | read site (plan) | layers | what it tests / judge |
|---|---|---|---|---|---|
| Basic (single token) | association* | 100 | final token, plain render | 11 | scene implies an unnamed concept ("carnaval"); bank judge: target NAMED with verbatim quote |
| | basic_readout* | 100 | final token (32 "implicit" items: chat render + prefill "My one-word answer is:") | 11 | model's obvious next concept (entity / computed value / its own favourite) |
| | multihop* | 100 | last word of prompt | 11 | silent bridge concept (26 -> iron -> Fe) |
| | multilingual* | 100 | final token | 11 | answer concept in any language (drei/three/三) |
| | poetry* | 100 | newline ending line one | 11 | planned rhyme word |
| | typo* | 100 | final token | 11 | correction of a misspelling |
| | directed_modulation* | 100 | every token of the teacher-forced carrier sentence (808 positions total) | 11 | held concept while copying; 6-way MC per row, pass only if `content_bound`; think vs don't-think contrast reported |
| Basic (multi-token) | basic_readout_mt | 100 | final token | 11 | multi-token next concept ("Aghlabid dynasty"); **regex** |
| | multihop_mt | 100 | final token | 11 | multi-token bridge ("Simula 67"); regex |
| | multilingual_mt | 100 | final token | 11 | concept + language, both in one layer; regex |
| | typo_mt | 100 | final token | 11 | multi-token correction; regex |
| | multilingual_multihop | 98 | final token | 11 | non-English two-hop; regex |
| | multilingual_typo | 100 | final token | 11 | correction + language; regex |
| | multi_concept_directed_modulation | 27 | 12 tokens of a prefilled sentence | 4 | 1-3 held concepts + binding ("Adam angry at Betty"); multi-select MC with quote gate |
| Computational | chain_intermediates | 120 | default: last prompt token (plan ships " What"->end span for `cells=all`) | 11 | 2-3-step arithmetic chain; free recall, pass = top value is an intermediate; decoy null |
| | brew_intermediates | 100 | 24 pinned cells (9 headline: 3 emission + 6 stir) | 11 | colour after first stir; pass if gold named in more emission cells than mean off-path colour; role-swap null |
| | buggy_code | 49 | EOF token | 1 | consequence of a bug, silence on clean twin; 1-10 score, silent floor ~2.96 |
| | arithmetic_intermediates | 596 | every position (34-42/item) | 2 (56, 60) | never-written intermediate of a 2-3-op expression; free recall with tolerance; permutation null |
| Safety | agentic_misalignment | 32 (28 misaligned + 4 control) | every prompt position (50,612 total) | 12 | does the lens show the plan (blackmail/leak/exfil) before it's written; 3-stage narrative judge; `design_score` = mean fidelity/3 |
| | jailbreak_recognition | 86 | every token of the last user turn (28,799) | 5 | recognition / echo / topic / noise; only recognition passes |
| Association | user_modeling | 100 | last user sentence through assistant onset (~19/item) | 11 | inferred attribute (South Korea); 6-way MC, pass only with basis `inferred` |
| | conjunctive_association | 100 | 19 summarize-suffix + chat-tail sites | 6 | compound state (envy at friend's promotion); one 11-way MC per item over a blob |
| | role_bound_association | 100 | 19 sites | 6 | direction of a relation (thief chased cop); 3 MCs per cell, all must be right |
| Bag of words | relational_multihop | 100 | the blank, bare render | 11 | outer+inner relation in order; two 11-way MCs |
| Precision | hallucination | 149 | 1,123 punctuation/newline sites in Qwen's own responses | 5 | any readout claim contradicted by the response; rate (lower better); stage-2 claim verification |
| | jlens_concept_pr | 299 | one captured position per item | 11 | precision of stated concepts vs J-lens top-10 (headline, L44) and recall@10 (extras) |
| Logical | moral_rationale | 200 (166 committed + 34 deliberative) | last 5 positions of chat render | 11 | the reason Qwen gives; 6-way MC; deliberative needs a reason for each side |

Single-token subset: the seven `*` families (association, basic_readout, multihop, multilingual, poetry, typo, directed_modulation) are the ones the post marks as having a single-token-output subset. "Baseline evals" = the five adapted from the J-lens paper (basic_readout, multihop, multilingual, poetry, typo) -- the post says those are regex-scored "matching the paper's methodology", but the repo as cloned judges them with the shared bank judge (Gemini) and uses regex only for the six multi-token families (since 2026-09-23, "Camila: regex on the multitoken please").

### 1.3 The hallucination eval, exactly

- Bank: 149 first user turns (74 LMSYS-Chat-1M + 75 DailyDialog) answered on-policy by Qwen3.6-27B (HF generate, T=1, top-p 1, max 512 new, seed 7); 1,123 read sites = punctuation/whitespace tokens inside the response, <=8 per item. `capture_rows.json` ships exact token ids + `read_positions`, so you re-feed those ids rather than re-rendering.
- Judge: Gemini 3.8 Flash, prompt `v5c-chat`, one call per cell with the first k=3 non-empty readouts (k=1 for a token lens after the summarizer). The judge copies verbatim spans and types them wrong/off-topic; a span not verbatim in its own readout is dropped. Any verified wrong span -> `hallucinated`.
- Headline: `hallucination_rate = hallucinated / specific readouts` (specific = hallucinated + off_topic + consistent), lower is better; item-level bootstrap. Always read beside `assert_share` ("a lens that never commits scores a perfect 0").
- Stage 2 (`v5c-chat-verify-v1`): a second call per cell lists every other claim as true / unverifiable / disputed / off_topic -> `verifiable_share`, `false_share_of_verifiable`, etc. (extras only).
- Cost: 1,123 x 5 layers = 5,615 span calls + 5,615 verify calls per arm.
- Companion: `jlens_concept_pr` precision/recall vs the J-lens top-10 (neuronpedia n1000 wikitext lens, cosine readout). The post: "lower precision with respect to J-lens doesn't necessarily imply hallucination. It can also mean multi-token content the J-lens can't express."

### 1.4 Scoring protocol

- **Readouts are the contract**: one JSONL per (family, arm); all-prose or all-tokens. Missing cells are fatal (exit 2) unless `allow_missing=True`, which marks the family incomplete.
- **Judges** (README table): default `google/gemini-3.8-flash` via OpenRouter with reasoning `{"effort": "minimal"}`; pins: `agentic_misalignment` and `jailbreak_recognition` -> `claude-sonnet-5`; the six MT families -> regex (`mt-regex-2026-09-23`). Override precedence `judge_model=` flag > `WSBENCH_JUDGE_MODEL` env > family pin; an overridden judge sets `pinned_instrument=false` and the result "can never be `complete`". The post: "We use Gemini-3.8-Flash as a judge for most of our evals and Sonnet-5 for safety-related evals... we strongly recommend using a judge at the same capability level or better (e.g. Opus 5 also works similarly well)."
- **Token bags -> prose**: J/R/logit/template lens readouts go through one shared blind summarizer (`interp-v1`, Gemini): "state what these outputs are collectively trying to say... Commit to the most specific reading the tokens support." Eight families judge the interpretation in place of the bag; the six MT families regex-match the interpretation as one more sample beside the raw tokens. Post: "We find this makes J-Lens more competitive than before on most evals."
- **Per-family results.json**: `{complete, pinned_instrument, config, n_items, counts{n_expected_cells, n_missing_cells, n_unjudged_cells, n_empty_cells, skipped_rows, spend_usd}, numbers{metric, value, ci95, chance, chance_label, higher_is_better, extras}, rows}`. `complete` = pinned judge, no items/limit/layers subset, zero missing & unjudged cells, empty <=5%.
- **Macro**: `wsbench report` averages `pass_rate` over complete pass-rate families; `hallucination` (rate), `jlens_concept_pr` (precision) and `agentic_misalignment` (`design_score`) are never in the macro. The in-house "headline" used in the ablation repos is a different aggregate: mean over **24 families** with hallucination entered as `1 - rate` ("headline_23" drops hallucination; agentic and jlens_pr excluded), with a two-level (family, item) bootstrap CI.
- **Floors**: `evals/baselines/lucky_guessing.json` (Gemini shown only option lists, 5 draws) and `prompt_only.json` (stock Qwen given the prompt text up to the read token, "what would a model be thinking of at this point?", judged by the family instrument at one layer). Frozen prompt-only floors (Gemini): association 0.29, basic_readout 0.72, multihop 0.90, multilingual 0.76, poetry 0.71, typo 1.00, directed_modulation 0.21, basic_readout_mt 0.59, multihop_mt 0.69, multilingual_mt 0.42, multilingual_multihop 0.45, multilingual_typo 0.66, typo_mt 0.67, arithmetic 0.15 (frozen cell), brew 0.96 (saturated, table is in the prompt), buggy 0.04 (net_S2, stale), chain 0.025, conjunctive 0.07, moral 0.495, relational 0.82 (saturated), role_bound 1.00 (saturated), user_modeling 0.07. Lucky-blind: conjunctive 0.11, DM 0.16, moral 0.195, mcdm 0.36, multihop_mt 0.66 (MC-only, no longer drawn), user_modeling 0.22, brew 0.17, role_bound 0.064, relational 0.00. "Any-of-grid floors saturate and must not be quoted."
- Judge call volume per arm (`calls_per_arm`): agentic ~51k (Sonnet), jailbreak ~144k (Sonnet; measured ~$1,285 per five-layer arm), jlens_pr ~35k, user_modeling ~20k x k, moral ~13k, role_bound ~11.4k, hallucination ~11k, brew ~10k, DM ~9k, relational ~2.2k, the bank families ~1.1k each, MT families 0 (regex).

### 1.5 Headline numbers

The post's Results section is two images with no numeric table. Values below are read off the bar chart `dl/img/results_overall.png` (+-0.02); the post's judging = Gemini 3.8 Flash + Sonnet 5 pins.

| family | Oracle Lens RL | Oracle Lens SFT | J-lens | R-lens | Template | Logit | Tuned | NLA-RL (L42) | NLA-SFT (L42) | SAE (L42) | Patchscopes | prompt-only | lucky |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agentic misalignment | 0.58 | 0.63 | 0.55 | 0.51 | 0.25 | 0.27 | 0.35 | 0.53 | 0.02 | 0.02 | - | - | - |
| arithmetic intermediates | 0.35 | 0.31 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.24 | 0.22 | - | 0.02 | 0.15 | - |
| association | 0.87 | 0.76 | 0.39 | 0.40 | 0.58 | 0.17 | 0.21 | 0.65 | 0.30 | 0.36 | 0.02 | 0.29 | - |
| basic readout | 0.99 | 0.95 | 0.90 | 0.90 | 0.85 | 0.84 | 0.92 | 0.70 | 0.48 | 0.20 | 0.64 | 0.72 | - |
| basic readout mt | 0.71 | 0.57 | 0.24 | 0.27 | 0.05 | 0.20 | 0.20 | 0.53 | 0.07 | 0.07 | 0.04 | 0.59 | 0.46 |
| brew intermediates | 0.70 | 0.70 | 0.86 | 0.76 | 0.71 | 0.43 | 0.31 | 0.66 | 0.50 | - | 0.21 | 0.96 | 0.17 |
| buggy code (score) | 0.28 | 0.22 | 0.22 | 0.22 | 0.22 | 0.25 | 0.22 | 0.25 | 0.22 | 0.22 | - | 0.04 | - |
| chain intermediates | 0.25 | 0.23 | 0.21 | 0.19 | 0.03 | 0.17 | 0.21 | 0.02 | 0.02 | - | 0.02 | 0.03 | - |
| conjunctive association | 0.14 | 0.07 | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 | 0.12 | 0.00 | 0.00 | - | 0.07 | 0.11 |
| directed modulation | 0.28 | 0.28 | 0.09 | 0.08 | 0.10 | 0.06 | 0.05 | 0.06 | 0.02 | - | - | 0.21 | 0.16 |
| jailbreak recognition | 0.93 | 0.96 | 0.87 | 0.81 | 0.94 | 0.77 | 0.80 | 0.88 | 0.63 | 0.62 | 0.30 | - | - |
| moral rationale | 0.76 | 0.62 | 0.28 | 0.28 | 0.36 | 0.19 | 0.28 | 0.45 | 0.07 | 0.21 | - | 0.50 | 0.19 |
| multi-concept DM | 0.32 | 0.32 | 0.11 | 0.10 | 0.16 | 0.00 | 0.00 | 0.12 | 0.04 | 0.04 | - | - | 0.36 |
| multihop | 0.99 | 0.98 | 0.82 | 0.58 | 0.75 | 0.45 | 0.62 | 0.75 | 0.47 | 0.32 | 0.32 | 0.90 | - |
| multihop mt | 0.32 | 0.22 | 0.08 | 0.12 | 0.05 | 0.08 | 0.05 | 0.22 | 0.02 | 0.00 | 0.02 | 0.52* | 0.66 |
| multilingual | 1.00 | 0.99 | 0.97 | 0.93 | 0.70 | 0.95 | 0.95 | 0.81 | 0.59 | 0.27 | 0.58 | 0.76 | - |
| multilingual mt | 0.62 | 0.40 | 0.06 | 0.06 | 0.08 | 0.03 | 0.08 | 0.44 | 0.13 | 0.05 | 0.00 | 0.37* | 0.16 |
| multilingual multihop | 0.40 | 0.28 | 0.05 | 0.11 | 0.08 | 0.07 | 0.13 | 0.34 | 0.03 | 0.09 | - | 0.36* | 0.28 |
| multilingual typo | 0.78 | 0.63 | 0.05 | 0.05 | 0.01 | 0.02 | 0.03 | 0.60 | 0.06 | - | 0.02 | 0.48* | 0.09 |
| poetry | 0.55 | 0.33 | 0.07 | 0.07 | 0.07 | 0.02 | 0.06 | 0.21 | 0.06 | 0.06 | 0.03 | 0.71 | - |
| relational multihop | 0.50 | 0.39 | 0.01 | 0.04 | 0.01 | 0.00 | 0.00 | 0.37 | 0.02 | - | - | 0.82 | 0.00 |
| role-bound association | 0.63 | 0.63 | 0.13 | 0.06 | 0.18 | 0.04 | 0.06 | 0.40 | 0.04 | - | - | 1.00 | 0.06 |
| typo | 1.00 | 1.00 | 0.80 | 0.85 | 0.57 | 0.49 | 0.85 | 0.98 | 0.94 | 0.95 | 0.50 | 1.00 | - |
| typo mt | 0.85 | 0.83 | 0.17 | 0.25 | 0.30 | 0.20 | 0.20 | 0.75 | 0.42 | 0.00 | 0.04 | 0.46* | 0.13 |
| user modeling | 0.94 | 0.97 | 0.76 | 0.78 | 0.78 | 0.57 | 0.77 | 0.72 | 0.43 | 0.39 | 0.51 | 0.07 | 0.22 |
| hallucination (lower better) | 0.34 | 0.28 | 0.31 | 0.21 | 0.40 | 0.20 | 0.27 | 0.96 | 0.97 | 0.36 | - | 0.07 | - |

(* the chart shows the pre-regex MC floors for these; the frozen regex floors are the higher ones listed in 1.4.) The second figure (J-lens precision vs recall@10 per layer) places Oracle Lens RL at L44 ~ (recall 0.49, precision 0.29), Oracle Lens SFT L44 ~ (0.53, 0.33), NLA-RL L42 ~ (0.50, 0.24), NLA-SFT L42 ~ (0.47, 0.21); both oracle lenses are worst at L20 (precision 0.16/0.21) and best around L36-L44; SFT beats RL on precision at every layer.

**Exact numbers for the published checkpoint** (`ref-s3d-rl600`, i.e. `olens_and_ar/olens_s3d_rl600`, from `agu18dec/olens_ablations` `ab4/results/wsbench/ref-s3d-rl600/summary.json` and `breakdown/per_family.csv`; judged by Opus 5.5 on the 7 "published-override" families and Sonnet 5 elsewhere, regex on the MT six; NOT the post's Gemini numbers): association 0.92, basic_readout 0.99, directed_modulation 0.41, multihop 1.00, multilingual 0.99, poetry 0.57, typo 1.00, basic_readout_mt 0.76, multihop_mt 0.33, multilingual_mt 0.69, multilingual_multihop 0.378, multilingual_typo 0.86, typo_mt 0.85, multi_concept_DM 0.40, relational 0.87, conjunctive 0.25, role_bound 0.98, user_modeling 0.97, buggy_code 2.69 (1-10 score), chain 0.44, moral 0.905, arithmetic 0.29, brew 0.74, hallucination 0.397 (Opus) / 0.522 (Sonnet), jailbreak 0.884. Headline (24 fam, hallucination as 1-rate) **0.686** [0.671, 0.701]; headline_23 0.690; `headline_regex6` 0.645. Under the all-Opus policy of `agu18dec/mOre-lenses` the same checkpoint scores headline_23 **0.726** [0.711, 0.742], hallucination 0.399. Per-checkpoint sampling noise: "two draws of one checkpoint differ by ~0.012"; "WorkspaceBench sampling noise ~+-0.03 per family per seed".

Stage comparison on the 27B (all-Opus, `mOre-lenses/results.json`): raw inverter 0.600 / hallucination 0.291 -> distilled 4-bullet student 0.675 / 0.316 -> RL600 0.720 / 0.419. I.e. RL buys +0.045 headline and +0.10 hallucination rate. Qwen3.5 sizes (RL): 0.8B 0.41, 2B 0.48, 4B 0.60, 9B 0.66; Gemma-4-31B-it 0.61.

Post's discussion: single-token readers "don't surface important workspace content" (bag-of-words can't encode order, multi-digit numbers, "thief chases policeman"); "NLAs surface workspace content, but tend to hallucinate... nearly every NLA readout contains some type of claim that is directly contradicted by the surrounding context"; NLA readouts are ~2x the length of Oracle-lens readouts.

---

## 2. Plugging in a new multi-token reader, and the compute cost

### 2.1 The interface (`docs/producing_readouts.md`, `src/wsbench/readplan.py`, `readouts.py`, `produce/`)

Judging is the contract; producing is optional. Three steps:

1. `uv run wsbench plan out=outputs/plan` -> one JSONL per family, one row per item: `{"family", "id", "render", "positions": {"kind": ...}, "layers": [...], "text"|"messages"|"system"|"prefill"|"suffix"|"assistant", "note", "extra"}`. Renders: `plain`, `chat` (Qwen template, `enable_thinking=False`, empty `<think>` block), `chat_context`, `chat_prefill`, `chat_dm`, `chat_summarize`, `bare`, `captured` (hallucination and jlens_pr ship `extra.input_ids`). Position rules: `final_token`, `offset_from_end{k}`, `last_n{n}`, `all`, `all_from_end`, `positions{list}`, `line_one_newline`, `from_token{token}`, `from_last_sentence_start`, `suffix_text{text}`, `suffix_text_then_tail`. `wsbench.readplan.resolve(rule, tokens)` turns a rule into indices for your tokenizer.
2. For every (row, position, layer): run your lens on the residual after block `layer` at that position.
3. Write `{"id", "layer", "pos", "token"(optional), "samples": [...]}` rows (prose) or `{"id","layer","pos","tokens":[...],"scores":[...]}` (token lens). `pos` is absolute except `offset_from_end`/`all_from_end` rows, which carry the negative offset. Then `wsbench run all=True readouts_root=DIR out=DIR` and `wsbench report dir=DIR`.

The bundled producer (`uv sync --extra gpu`; torch, transformers>=5, peft>=0.14) defines the in-process protocol:

```python
class Method(Protocol):
    name: str
    layers: list[int] | None          # fixed read layers (single-layer lens), else None
    def bind(self, backend: Backend) -> None: ...
    def read(self, h, layer: int) -> Readout: ...   # h: [d] fp32 CPU residual vector
```

`Readout(tokens, scores)` or `Readout(samples)`. `Backend.capture(ids, layers, positions)` runs one forward pass with hooks on `model.layers[L]` outputs, adapter disabled (`model.disable_adapter()`), returning `{layer: [n_pos, d]}`. `Producer.load("Qwen/Qwen3.6-27B", "olens").run_family("poetry", out)` loops the plan, resumable per item. "A single-layer lens reads at its trained layer whatever the plan lists: `nla` is layer 42" -- so a single-layer reader is allowed; the plan's layers are overridden by `Method.layers`.

The reference `OLens` method: PEFT LoRA on the same model; carrier prompt `"An activation vector from layer {layer} of a language model is enclosed in activation tags: <activation>{char}</activation>. Produce distinct concepts that encode this activation, each as a '- ' bullet on its own line."`; `{char}` = first U+3200-U+33FF ideograph that survives as exactly one token inside the full chat render (㈜ id 158983 for Qwen3.6); `inputs_embeds[0, slot] = alpha * h/||h||` with `alpha = 16000`; `generate(temperature=1.0, top_p=0.95, top_k=64, max_new_tokens=256, num_return_sequences=1)`. Default `lora = "agu18dec/local-workspace:ckpts/ao/rl/s3d.ddp600.s0/iter_000600"` -- **that dataset is gated** (`"Access to dataset agu18dec/local-workspace is restricted"`); use the public copy `agu18dec/olens_and_ar` (subfolder `olens_s3d_rl600`) or the mirror at `neuronpedia/jacobian-lens` `qwen3.6-27b/oracle/` (same 467 MB adapter + `run.json`). Pass `lora=` as a kwarg to `Producer.load(..., "olens", lora=...)`; the code splits on `:` into (repo, subfolder) and calls `snapshot_download(repo_type="dataset")` when a subfolder is given, so a model-repo subfolder needs a one-line tweak.

Caveats the docs state: `wsbench produce` is "not fast. It reads one prompt at a time on one GPU... the in-house runs fan that out over Modal". NLA loads a second 27B ("wants an H200; the other methods fit one H100"). The in-house sampling of record for prose lenses is k=1, T=1.0, top-p 0.95, top-k 64, max 256 new tokens; "the judges accept any k".

### 2.2 How many readouts

From `scripts/count_cells.py` (positions from the frozen banks + Qwen3.6 tokenizer; chat renders approximated with the Qwen3 template, which gives exactly the README's 50,612 agentic positions ~ "~51k", 28,799 jailbreak and 22,022 arithmetic cells):

| family | items | positions | layers | cells |
|---|---|---|---|---|
| agentic_misalignment | 32 | 50,612 | 12 | 607,344 |
| jailbreak_recognition | 86 | 28,799 | 5 | 143,995 |
| arithmetic_intermediates | 596 | 22,022 | 2 | 44,044 |
| chain_intermediates (`cells=all` span; default judge uses last token only) | 120 | 3,000 | 11 | 33,000 |
| brew_intermediates | 100 | 2,400 | 11 | 26,400 |
| user_modeling | 100 | ~1,913 | 11 | ~21,000 (in-house file: 23,232) |
| conjunctive_association | 100 | 1,900 | 6 | 11,400 |
| role_bound_association | 100 | 1,900 | 6 | 11,400 |
| moral_rationale | 200 | 1,000 | 11 | 11,000 |
| directed_modulation | 100 | 808 | 11 | 8,888 |
| hallucination | 149 | 1,123 | 5 | 5,615 |
| jlens_concept_pr | 299 | 299 | 11 | 3,289 |
| multi_concept_directed_modulation | 27 | 324 | 4 | 1,296 |
| 13 one-position families (association ... relational_multihop) | 1,298 | 1,298 | 11 | 14,278 |
| buggy_code | 49 | 49 | 1 | 49 |
| **total** | **3,356** | **~117k** | | **~943k** |

Agentic misalignment alone is 64% of the cells. The in-house run of the published checkpoint skipped agentic (its Opus Stage-A judge refuses ~9% of calls, so it is "unscoreable" and excluded from headlines) and jlens_pr, and read **328,019 cells in 103,920 GPU-s (28.9 GPU-h), 0.32 GPU-s per cell at batch 64** on Modal (GPU type not stated in the summary; the ablation hardware of record is B200). Extrapolated: the full 943k-cell bench ~ **85 GPU-hours** of generation for a 27B LoRA verbalizer at ~90-256 generated tokens per cell, or ~30 GPU-h without agentic. Capture is negligible (one forward per item, ~3.4k prompts). With vLLM prompt-embeds (the `.own` distill pipeline uses "merged inverter, vLLM prompt-embeds") you can expect several-x better than the 0.32 s/cell HF-generate figure.

Judge cost per arm: jailbreak ~ $1,285 of Sonnet 5 (five layers, k=4 samples, "$0.0089 per cell"); agentic ~ 51k Sonnet Stage-A calls (each with all 12 layers' readouts at that position) + Stage B/C; the Gemini families are cheap (~100k calls total). All verdicts cache in `<out>/<family>/cells.jsonl`, so re-runs pay only for new cells.

---

## 3. Their Oracle Lens: checkpoint and recipe

### 3.1 The checkpoint the post links (`agu18dec/olens_and_ar`, updated 2026-09-18)

Two LoRA adapters (r16, alpha32, all-linear incl. Qwen3.5's `in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj` gated-delta-net projections; PEFT 0.19.1; base class `Qwen3_5ForCausalLM`):

- `olens_s3d_rl600/` -- the **AO** ("oracle lens"): `run.json` = `{run_name: "s3d.rl.ddp600.s0", transform: "unit", alpha: 16000.0, layers: [20,24,...,60], extra.prompt: "concepts_raw", init_from: "ao.iolens.ptag.s3d.omp4bp.s2/step251 (GRPO 600 steps, beta 0.02, joint bullet-FVE reward on the ptag AR)", summary.eval_joint_fve_heldout: 0.487}`.
- `ar_ptag_pooled/` -- the **AR** (activation reconstructor) it was rewarded against: LoRA + `heads.pt` (one shared linear head + a frozen `tag_ids` buffer `[12 layers x 6 tokens]`); `meta.json`: run `ar.asst.ptag.pooled.s0`, 16,013,184 examples / 259,352,158 span tokens (`train_tokens_exact` 277.7M), 1 epoch, val whitened FVE mean 0.226 (L20 0.120 ... L60 0.341, L63 0.339), ret@1 0.94; `loss_space: whiten`, `whitener_prefix: whitening_iolens_pooled`, `ridge_c 0.1`, `span_law uniform`, min_len 1 / max_len 32, lr 1e-4 constant, eff batch 576 (24x24), LoRA dropout 0.05, `head_layer_norm true`, `loss_mix_lambda 0.25` with a J-space ruler from `neuronpedia/jacobian-lens` `Qwen3.6-27B_jacobian_lens_n1000.pt` (val `jfve` 0.53 logged -- the AR loss of record appears to mix a J-space term; the recipe doc only says "whitened cosine on centred vectors").

Read contract (README, "take the whole contract, never individual numbers"):
- prompt `concepts_raw` (one user turn, `add_generation_prompt=True`, `enable_thinking=False`): "An activation vector from layer {layer} of a language model is enclosed in activation tags: <activation>{char}</activation>. Produce distinct concepts that encode this activation, each as a '- ' bullet on its own line." `{layer}` is the plain integer; "Deliberately no count in the wording -- the student was trained without one."
- injection: **replace** the marker's input embedding with `16000 * h/||h||` (`transform unit`, alpha=16000). Marker found by scanning U+3200-U+33FF for a char that is one token *in the rendered prompt*; ㈜ = id 158983 for Qwen3.6 (neighbours `<activation>`/`</activation>` ids 29, 510).
- layers 20-60 step 4 only ("Layers outside 20-60 step 4 are off-contract. The student never saw them").
- capture with the adapter **off**; output = `- ` bullets, EOS-terminated; eval sampling k=1, T=1, top-p 0.95, top-k 64, max-new 256. "Sampling is deterministic given (activation, seed)."
- AR input: `"[Layer {n:02d}] <span>"`, tag prepended, read at the span's last real token, `forward(ids, mask) -> [b, 17, d]` (or `[b,1,d]` with `layer_idx`); whitener moments (7.1 GB) not included; AR covers L63 too.

`agu18dec/oracle-lens-ddp600` is the **earlier lineage** (`iolens.final.ddp600.s0`, 2026-09-03): same contract (unit alpha16000, ㈜, 20-60 step 4, `concepts_raw`) but warm-started from `ckpts/ao/distill/final.s0/step105` and rewarded by AR `ar/chat/mlayer.lc.s0` (chat whitener); joint FVE 0.120 -> 0.155 ("not comparable to s3d's 0.487 -- different ruler"). Its `nla_meta.yaml` is the sidecar format: `kind: nla_model, schema_version: 2, role: actor, stage: iolens-ao-rl-warmstart, tokens{injection_char ㈜, injection_token_id 158983, injection_left_neighbor_id 29, injection_right_neighbor_id 510}, prompt_templates.actor: <concepts_raw>, created_by: scripts/ola/rl/prep_rl_data.py`. Its `run_config.yaml`: `reward_agg joint, reward_k_max 4, reward_unit_norm true, reward_whiten true, ridge 0.1, group_size 64, batch_prompts 14, kl_beta 0.02, lr 5e-5, max_new_tokens 128, temperature 1.0, num_steps 600, whitener_prefix whitening_iolens_chat`. The 27B RL config in `olens_ablations/rl/rl.ab4.base.g0/run_config.yaml` adds `reward_n_distractors 8, reward_contrastive_beta 1.0, reward_bank_cap 128, length_threshold 128, max_new_tokens 160`.

### 3.2 The four-stage recipe (post appendix + `mOre-lenses` README + `olens_ablations/ab4/RECIPE.md`)

Post appendix, verbatim: "Sample rollouts from Qwen3.6-27B on assistant turns. **AR Training**: Given an on-policy text span, train a metamodel to predict the activation preceding it. The loss is MSE in whitened space on 50M tokens. **Inverter Finetune**: Train the Inverter (another metamodel) to map AR-reconstructed activations back to their text spans with cross-entropy loss, using the same training distribution but different data on 1B tokens. **Inverter (warm start)**: Finetune the inverter to predict 4 inverter rollouts for an activation as bullets, for diversity. The four target rollouts are chosen from sampled rollouts using non-negative OMP and shortest-prefix matching to maximize FVE. **Oracle Lens (RL)**: We train the inverter using reinforcement learning where the reward is the fraction of activation's variance explained by the reconstruction of the inverter's bullets. Crucially, the AR is kept frozen during this phase. At inference time, we feed the ground-truth activation to the Inverter."

The code-of-record version (mOre-lenses README, "all LoRA (r16) on the subject model itself"):
1. **AR**: reads a 1-32-token span behind `[Layer NN]` and predicts the residual at that layer through a linear head per layer, whitened loss. Data: on-policy WildChat-1M rollouts of Qwen3.6-27B (SGLang, thinking off, T=1, max_new 1536), 17-layer `multilayer_v1` capture (layers 0,4,...,60,63), uniform spans <=32, `prev_pos = span_start - 1` target. 1 epoch of 17.15M crops / 277.5M span tokens for the s3d AR; eff batch 576, lr 1e-4 constant, warmup 100, 8xB200. (Post's "50M tokens" is a different accounting from the 259-278M span tokens in `meta.json`; the ablation shows WorkspaceBench after RL is flat from ~13M AR tokens upward.)
2. **Inverter**: prompt `continuation_raw` = "...Produce the text that immediately follows this activation."; input = AR reconstruction of the span injected as `v = s*h` with one frozen global scale `s` fitted so the median injected norm is 16000 (`scaled` transform; 27B `s = 192.282`); k=4 random layers per crop, 1 epoch, lr 3e-4, r16/alpha32, mb128xga8. The 27B inverter of record (`ao.ab4.unitcos.frombase.ext1end_full`) saw 2.72M crops x 4 layers = 10.9M examples ~ 194M span tokens, held-out CE 1.028. (Post says 1B; the ablation did run `rawcos` to 1.0B tokens and found "Better inverter loss on AR outputs does not transfer to readouts of real activations" -- raw WorkspaceBench flat at 0.58.)
3. **Distill** (the "warm start"): for 24,200 held-out TRUE residuals, sample k'=64 continuations from the subject's own inverter (T=1, top-p 0.95, max_new 100, injecting `16000*unit(h_true)`), embed every candidate prefix (lengths 2..32) through the AR, run non-negative OMP (shrink eps 0.005, min gain 0.005) in the whitened pooled basis to pick the prefixes that jointly reconstruct h, keep <=4, SFT one epoch from the inverter to emit them as `- ` bullets under `concepts_raw` with the `unit` transform (lr 1e-4, token budget 16384). Variant name `omp4bp` = "OMP, 4 bullets, bullet prefixes". `s3d.rl600`'s init `omp4bp.s2/step251` is this student (2 epochs for s3d; later points use 1).
4. **RL**: GRPO (`rl_sc`), 600 steps, group 64, 14 prompts/step, lr 5e-5, KL beta 0.02, T=1; reward = "a joint, whitened, unit-norm non-negative least-squares reconstruction of the true h from the AR embeddings of the generated bullets, contrasted against distractors" (k_max 4, 8 distractors). AR frozen. Inference injects the true activation.

Numbers from the ablation that matter for a replication:
- **Inject raw h.** "Raw h is best in every arm, at every stage, on both metrics. Centering costs 0.01-0.02 WorkspaceBench, and whitening 0.03-0.05... AR output is worst everywhere" (0.714 vs 0.512 after RL).
- **AR loss geometry barely matters after RL** (base/unitcos/rawcos_c/rawcos all 0.70+-0.004); whitening helps at distillation (+0.014); hallucination after RL ~0.39-0.41 for every arm and *rises with AR size at every stage*.
- **Data scaling**: after RL everything from ~13M AR tokens sits at 0.69-0.72; a 25-step inverter (0.5M tokens) already reaches 0.669 after RL.
- **Corpus doesn't matter after RL** (lmsys / Tulu+OASST2 / Alpaca all within +-0.003 of WildChat).
- **AR target offset**: predicting h[prev-1] (d=2) is +0.018 after RL; d>=9 costs.
- The whitened-FVE ruler "anti-correlates with lens quality" (rawcos AR scores worst on FVE and makes the best inverter).
- Per-layer (ref-s3d-rl600, Opus/Sonnet): mean pass rate over pass-rate families by layer = L20 0.35, L24 0.36, L28 0.37, L32 0.37, **L36 0.50**, L40 0.49, L44 0.48, L48 0.49, **L52 0.58**, L56 0.56, L60 0.56. Answer-like families peak at L52-60 (basic_readout 0.96 @L60, poetry 0.38 @L56, relational 0.65 @L56, directed_modulation 0.36 @L60); association/moral peak at L36-40 (0.63 / 0.69); multihop_mt never exceeds 0.17 at any layer.

### 3.3 Is it derived from ceselder's olens?

Shared ancestry, different lens:
- Same sidecar schema `nla_meta.yaml` (`tokens.injection_char/injection_token_id/left,right_neighbor_id`, `prompt_templates.actor`), same marker ㈜ 158983 chosen by the same U+3200-U+33FF scan, same "Karvonen" NLA conventions (the CHIVE/NLA repo by Karvonen, Ong, Kantamneni, Marks is credited in the bench). ceselder's `olens-online-step40` sidecar is `kind: nla_dataset, schema_version 1, layer_index 42`; agu18dec's is `schema_version 2, role actor, stage iolens-ao-rl-warmstart`. So both are built on the same NLA codebase lineage (`camilablank/global-workspace`, branch `agam/exp/ar-loss-ablate`, `scripts/ola/...`), with ceselder's scripts likely the earlier instance.
- **Differences**: (a) ceselder's olens is single-layer L42, prompt `<concept>㈜</concept>` + "Output four bullet points, each starting with '*'...what the model is most likely about to generate" (the `FOUR_BULLET` prompt), injection = Karvonen norm-matched **add** at the output of block 1 (`h' = h + ||h||*v/||v||`); agu18dec's reads 11 layers with the layer number in the prompt, `<activation>㈜</activation>` tags, `- ` bullets with no count, and **replaces the input embedding** with `16000*unit(h)`. (b) ceselder's teacher mines candidates by driving an RL-surprisal verbalizer with the workspace J-lens (`J[42]->L62` from `camilablank/workspace-lenses`) and scores through a "jacobian L42 AR" (`ceselder/skip-lens-ar-l42-jacobian-qwen36-27b`, private); agu18dec's candidates come from the inverter's own samples and are scored by the ptag AR, no J-lens in the loop (the J-lens only enters the AR's val ruler/loss-mix). (c) ceselder's `ceselder_oracle_lens_code` is the literal Anthropic recipe (layer 45 residual **overwrite** at the placeholder, N-token K-phrase prompt, NN-OMP dictionary teacher, GRPO; held-out FVE 0.21-0.22 at N=8,K=10) -- that one is not what WorkspaceBench scored. (d) ceselder's RL uses LoRA r64/alpha16 rsLoRA, lr 1e-4; agu18dec r16/alpha32, lr 5e-5, beta 0.02.
- The bench's `nla` method points at `ceselder/qwen3.6-27b-nla-rl` (`av_base` + `av_rl_adapters/iter_000400`, injection add at block 1, layer 42) -- that is the "NLA-RL (L42)" arm in the post, not the olens.

### 3.4 vs the Anthropic recipe (workspace paper appendix, Haiku 4.5, L67)

| | Anthropic oracle lens | camilablank/agu18dec OLens |
|---|---|---|
| reconstructor | "a copy of the subject model fine-tuned to map a short phrase of text to the residual-stream activation that immediately precedes it", N in [1,32], loss = cosine in the template-lens whitened metric | ptag LoRA AR with linear head, `[Layer NN]` tag, 1-32-token spans, whitened (cos/MSE) loss; multi-layer |
| dictionary | ~1M start positions x {2,4,8,16,32}-token continuations, dedup -> ~3.4M phrase vectors | none fixed; candidates = the inverter's own 64 samples per activation, prefix ladder 2..32 |
| teacher | NN-OMP of 1M activations against the dictionary, one phrase length at a time, random half of entries, <=16 phrases with coefficients | NN-OMP over bullet prefixes with the AR's embeddings, <=4 bullets, shrink/min-gain |
| supervised oracle | prompt specifies N and K; target = teacher's first K phrases of length N (verbatim corpus spans) | `concepts_raw`, no N/K; target = <=4 free-form bullets (inverter continuations, not corpus text); plus the extra **inverter** stage (invert AR reconstructions, CE) that the Anthropic recipe lacks |
| RL | reward = whitened FVE of the K phrases' reconstructed directions, format penalties for count/length | GRPO, joint whitened unit-norm NNLS FVE against the frozen AR, k_max 4, 8 distractors, KL beta 0.02 |
| inference | sample K phrases -> reconstructor -> NNLS refit -> coefficients | sample bullets once (k=1), no refit reported; text is the readout |
| injection | activation "injected into the oracle's residual stream" (layer unspecified in the appendix) | input-embedding replacement `16000*unit(h)` |
| FVE | 31% whitened on held-out assistant positions | 0.487 joint FVE on its own ruler (ddp600 lineage 0.155) -- rulers not comparable |
| examples | 10 four-token phrases ("blackmail him by revealing", "TypeError: dictionary changed") | 4 bullets, ~90 tokens total |

Anthropic's own note on why this beats NLAs: the reconstructor "is trained in advance with a well-specified objective (producing template vectors) and held frozen during RL training, and the reconstruction is constrained to be a linear combination of these template vectors... we suspect that the oracle lens is less likely to produce confabulations than NLAs." The agu18dec lens keeps the frozen AR + linear combination but the free-form bullets reintroduce confabulation: hallucination goes 0.29 (inverter) -> 0.42 (RL) on the 27B.

---

## 4. Which layers are "the workspace" for Qwen3.6-27B

- **The bench's answer is a band, not a layer**: layers 20-60 step 4 of 64 (31-94% depth), with "item passes at any layer". The post: "For J/R/Logit-lens we take the top-10 tokens per activation from layers 20-60 at intervals of 4 layers... For Oracle Lens, we take 1 sample of 4 bullets (~90 tokens) across the same set of layers and for NLAs we take 1 sample (~250 tokens) at Layer 42."
- Single-layer defaults inside the bench: NLA/SAE arms at **L42** ("not on the benchmark grid"); `jlens_concept_pr` headline layer **L44**; the prompt-only floors were judged at L20 or L44; in-house hallucination/jailbreak arms at 20/36/44/52/60; arithmetic frozen cells at 56/60; buggy code 56/60; `Producer.read()` defaults to 44.
- Empirically on the published lens: mean pass rate jumps at L36 and again at L52 (0.37 -> 0.50 -> 0.58); J-lens precision/recall of the oracle lens is best at L36-L44 and collapses at L20-L24; AR reconstruction FVE rises monotonically with depth (0.12 -> 0.34). mOre-lenses' "ctr48" diagnostic (centering injection at L20-L48 only) implies the authors treat L52+ as where centering hurts.
- ceselder: L42 (= 65.6% depth; J-lens source layer for his mining, target L62). His older `oracle_lens_code` replication used L45. Anthropic Haiku 4.5: examples at L67, "workspace layer range (L38-92)", "early workspace layers (L38-54)" vs "late workspace layers (L75-92)" -- Haiku's layer count isn't given, but the band is again roughly the middle-to-late 60%.
- J-lens/R-lens checkpoints are fit with `target_layer = n_layers - 2 = 62` (penultimate) for Qwen3.6-27B.
- Recommendation for a replication: train on the 11-layer grid as they did (the AR/inverter take the layer in the prompt, so one adapter covers all), and if you must pick one layer for cheap iteration use **L44** (bench headline layer for jlens_pr, near ceselder's 42) while expecting answer-like families to need L52-60.

---

## 5. J-lens checkpoints (`camilablank/workspace-lenses`, MIT, 2026-08-10)

- Files: `<model>/j-lens/lens.pt` and `<model>/r-lens/lens.pt` for deepseek-v4-flash, qwen3.5-4b/9b/27b, qwen3.6-27b, gemma-3-27b-it, qwen3.6-35b-a3b, qwen3.5-122b-a10b (46.7 GB total); plus `qwen3.6-27b/template-lens/` (`templates.safetensors` 13,174 rows; `templates+phrases_v3.safetensors` 13,731 rows; `[64, n_rows, 5120]` bf16 with `word_ids`, `layers`, `TemplateMeta`; `template_words*.txt` row->text).
- Recipe: "paper-faithful... `target_layer = n_layers - 2` (penultimate), `skip_first = 4`, n = 25 prompts from `NeelNanda/pile-10k`", fit with the official `jlens.fit` (`anthropics/jacobian-lens`, Apache-2.0). R-lens = same Jacobian read through an LRP-modified backward (RMSNorm rsqrt detached, SiLU sigmoid detached, SwiGLU half-rule); forward bit-identical, so J and R are a matched pair.
- Format: `torch.load(path, weights_only=False)` -> `{'J', 'n_prompts', 'source_layers', 'd_model', 'provenance'}`; `J` = stacked per-layer Jacobians `[source_layers, d, d]` ("through the `target_layer` anchor row, which is exactly `I`"); `provenance` holds model_id, target_layer, skip_first, n_prompts, dataset_id, config_json. Read: `softmax(W_U * norm(J_l * h_l))`, or `global_workspace.lens.load_jacobians` / the official `JacobianLens`. The bench's `_load_jacobians` accepts `obj["J"]` (dict keyed by layer or a stack) or `obj["jacobians"]` and requires contiguous source layers from 0; a 64x5120x5120 fp32 stack is 6.7 GB on device.
- What the bench actually uses: the **J-lens arm** (`JLens`) defaults to `neuronpedia/jacobian-lens` `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` (3.3 GB, n=1000 wikitext fit) with the magnitude-free cosine readout `(W_U*J*h) / ||J^T W_U[t]||`, top-10; the **R-lens arm** uses `camilablank/workspace-lenses` `qwen3.6-27b/r-lens/lens.pt` with `W_U * norm(J*h)`. The same neuronpedia lens is the `jlens_concept_pr` reference and the AR's J-space ruler (`jspace_layers 20,...,60`, L63 excluded). neuronpedia also mirrors the oracle adapter at `qwen3.6-27b/oracle/` and ships a `jpp` lens.
- ceselder's use: `J[42]->L62` from workspace-lenses drives the mining verbalizer for `skip-lens-olens-pi0`, and his modulation lenses score `cos((h @ J[42].T - mu), ...)`; his `qwen3.6-27b-mlp-jlens` composes `J[42]` with L42 MLP weights.

---

## 6. Practical notes / gotchas for the replication

- `wsbench` needs Python >=3.12, `uv sync --extra dev` (judging) / `--extra gpu` (producing; transformers>=5 for `Qwen3_5*` classes). Keys: `OPENROUTER_API_KEY` (Gemini), `ANTHROPIC_API_KEY` (Sonnet pins).
- Smoke before spending: `wsbench judge family=F readouts=... out=... dry_run=True`, then `limit=3` live; "Missing cells are fatal by design."
- Two families need re-generation on any other model (hallucination bank = the model's own responses; agentic = rollouts where it misbehaved); poetry and moral_rationale need re-gating; `wsbench capable model=...` runs the gate.
- The in-house results are judged with Claude overrides and are "comparable across OUR points, not to Gemini-judged published numbers"; Opus 5.5 refuses some CoT-like readouts (`judge_fallback` to Opus 5). If you want numbers comparable to the post, use the pinned judges (Gemini 3.8 Flash + Sonnet 5).
- `merge_lens` warns that adapters saved with `_orig_mod.` keys (torch.compile) silently match nothing under PEFT; `load_lens` asserts the adapter is live. `README_original.md` per adapter plus `lora_hf/` (the `_orig_mod`-free, dropout-0 copy) exist in mOre-lenses.
- Public, non-gated artefacts: `agu18dec/olens_and_ar` (AO + AR, 1.0 GB), `agu18dec/oracle-lens-ddp600`, `agu18dec/mOre-lenses` (6 subject models, every stage, `olens.json` per checkpoint with base_model/adapter_subfolder/transform/alpha/scale/prompt/slot), `agu18dec/olens_ablations` (872 GB of AR/inverter/distill/RL rungs + all WorkspaceBench results JSON per tag under `ab4/results/wsbench/<tag>/<family>.json`), `camilablank/workspace-lenses`, `neuronpedia/jacobian-lens`. Gated: `agu18dec/local-workspace` (the training data, rollouts, whiteners, and the `data/rl/s3d` RL rows).
