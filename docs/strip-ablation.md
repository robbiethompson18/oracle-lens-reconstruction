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
- Logged per step: `cjk_frac` = share of completion tokens that are CJK (ideographs, CJK
  punctuation, full-width forms, or partial UTF-8 bytes). Every rollout's phrases go to
  `rollouts.jsonl`.
- Outputs: HF `robbiethompson2018/oracle-lens-qwen3.6-27b-pilot-strip-ablation` (adapters, logs,
  rollouts, re-extracted activations).

Results: pending.
