# Leading-whitespace stripping ablation (Qwen3.6 pilot)

Question: does the WorkspaceBench / agu18dec oracle lens emit Chinese (5–7% of its tokens on CHIVE
transcripts, in ~40% of bullets) because of a training detail rather than because Qwen represents
Chinese? Their pipeline feeds each bullet to the AR as "the text after `- `, no leading space". Our
reconstructor was trained on phrases that keep their leading space (`" for your ..."`), and the
pilot lens emits 0 CJK characters. Hypothesis: stripping the space pushes English phrases off the
reconstructor's training distribution, while CJK phrases have no leading space and are unaffected,
so RL against a stripped reward drifts toward Chinese, even with an English-only corpus (ultrachat).

Setup (`sky/run_strip_ablation.sh`, 2×H100, 2026-10-21):

- Start from the pilot SFT checkpoint (`robbiethompson2018/oracle-lens-qwen3.6-27b-pilot-sft`) and
  the frozen pilot reconstructor; same config as the pilot RL (`configs/pilot.yaml`) except 20
  steps.
- Re-extract only the `rl`/`eval` activations from the local pilot transcripts (same seeds).
- Two legs in parallel, same seed: `strip` (`oracle.rl.strip_leading_ws=true`: each generated phrase
  is re-tokenised without leading whitespace before the reconstructor embeds it; format penalties
  still count the generated tokens) and `control` (no strip).
- Logged per step: `cjk_frac` = CJK ideographs / (ideographs + Latin letters) in the decoded
  completions. Every rollout's phrases go to `rollouts.jsonl`.
- Outputs: HF `robbiethompson2018/oracle-lens-qwen3.6-27b-pilot-strip-ablation` (adapters, logs,
  rollouts, re-extracted activations).

## Results (2026-10-21): null

Stripping hurts reward but RL does not move toward Chinese.

|                                                          | strip     | control   |
| -------------------------------------------------------- | --------- | --------- |
| step-1 rollout FVE (identical rollouts in both legs)     | 0.038     | 0.058     |
| held-out greedy FVE after 20 steps (N=8, K=10, 256 rows) | 0.134     | 0.160     |
| KL to SFT at step 20 (nats/token)                        | 0.65      | 0.59      |
| rollout phrases with a CJK ideograph, steps 1–10         | 1 / 5,419 | 1 / 5,418 |
| rollout phrases with a CJK ideograph, steps 11–20        | 1 / 5,437 | 3 / 5,428 |
| eval phrases with a CJK ideograph                        | 0 / 640   | 0 / 640   |

For comparison, the agu18dec lens on CHIVE transcripts: ~34% of bullets contain an ideograph,
~4.5–5% of its tokens are Chinese.

Caveats: 20 steps against agu18dec's 600, and our SFT start almost never samples Chinese (the few
hits are code-switches like `ID卡和`), so RL has little to amplify. The policy did move (KL ~0.6),
just not toward CJK. The agu18dec lens also differs in corpus (WildChat, multilingual), free-form
bullets, and its inverter stage; any of those is a better candidate than stripping.

Measurement gotcha: counting a token as CJK when it decodes to U+FFFD on its own (as a split
multi-byte CJK character does) also catches box-drawing characters (`─│├`) and emoji, which this
lens emits a lot. The first version of `cjk_frac` did that and read 0–2% per step in both legs;
measure on decoded text instead.
