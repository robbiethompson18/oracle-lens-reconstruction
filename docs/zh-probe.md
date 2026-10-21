# Chinese-input probe (Qwen3.6 pilot)

Follow-up to the [strip ablation](strip-ablation.md), which found no drift toward Chinese when RL
ran on English-only (ultrachat) activations. Two questions:

1. Do our lenses emit Chinese when the activation comes from Chinese text?
2. Does the same 20-step RL drift toward Chinese when the RL activations come from WildChat (which
   includes Chinese conversations), with and without stripping each phrase's leading whitespace?

Setup (`sky/run_zh_probe.sh`, 2×H200, 2026-10-21):

- Qwen3.6-27B answers two prompt sets (vLLM, thinking off, T=1, ≤384 tokens): 584 first turns that
  WildChat-1M labels `Chinese` (`/work/zh`), and 1,976 natural-mix WildChat first turns (`/work/wc`;
  18% of the RL rows come from Chinese prompts). Toxic rows are dropped.
- Exp 1: four existing lenses read 256 Chinese eval positions (4 per transcript), greedy and T=1.
- Exp 2: the strip ablation's 20-step RL from the pilot SFT checkpoint again (same config, same
  frozen ultrachat reconstructor and whitening), but with `rl.npy`/`eval.npy` from the WildChat
  transcripts. Both legs, `strip` and `control`, then read 128 positions each of English ultrachat
  (the pilot eval split), the WildChat mix and Chinese WildChat.
- Metrics (`eval_fve.py`): FVE at N=8, K=10; `fve_true_next` = FVE of the ≤32 tokens that actually
  followed, cut into 8-token phrases; share of phrases containing a CJK ideograph; ideographs /
  (ideographs + Latin letters) over all phrases.
- Outputs: HF `robbiethompson2018/oracle-lens-qwen3.6-27b-pilot-zh-probe` (WildChat transcripts,
  activations, both exp-2 adapters, rollouts, every eval JSON with all readouts).

## Exp 1: our lenses mostly translate Chinese activations into English

Chinese eval positions; 57% of the true continuation's letters are ideographs; true-continuation FVE
0.159. On English inputs every one of these lenses had 0/640 phrases with an ideograph.

| lens (all RL on ultrachat)  | FVE (T=0) | phrases with an ideograph (T=0 / T=1) |
| --------------------------- | --------- | ------------------------------------- |
| SFT                         | 0.028     | 0.4% / 0.3%                           |
| RL 20 steps, control        | 0.119     | 9.5% / 8.3%                           |
| RL 20 steps, strip          | 0.111     | 13.6% / 12.6%                         |
| RL 40 steps (pilot release) | 0.204     | 12.5% / 10.9%                         |

The RL lenses get the content right and mostly say it in English (Chinese "strict mode" text reads
as `strict mode enabled by default in new modules`). The Chinese they do write is fluent and on
topic (`醫師指出，導致此現象的主因`, FVE 0.40). So 20–40 steps of English-only RL already moves the
lens from 0.4% to 10–14% Chinese phrases when the activation is Chinese. The pilot RL lens beats the
true continuation's FVE (0.204 vs 0.159).

## Exp 2: WildChat RL with stripping makes the lens write Chinese about English text

| eval set (true ideograph share) | metric                    | control | strip |
| ------------------------------- | ------------------------- | ------- | ----- |
| English ultrachat (0%)          | FVE (true next 0.143)     | 0.180   | 0.136 |
|                                 | phrases with an ideograph | 1.4%    | 74.0% |
|                                 | ideograph share           | 0.6%    | 51.3% |
| WildChat mix (18%)              | FVE (true next 0.136)     | 0.150   | 0.144 |
|                                 | phrases with an ideograph | 32.8%   | 62.8% |
| Chinese WildChat (54%)          | FVE (true next 0.158)     | 0.170   | 0.183 |
|                                 | phrases with an ideograph | 71.4%   | 80.8% |

Greedy, 128 rows each; T=1 is within ~3 points everywhere.

- **Stripping plus some Chinese in the RL data reproduces the agu18dec behaviour, and more.** On
  plain English text the strip lens writes on-topic Chinese (`项目紧张时期，我会调整工作模式` for a
  thank-you-for-the-project email). 74% of its phrases contain an ideograph, against ~34% of bullets
  for the agu18dec lens on CHIVE. Its English FVE drops (0.136 vs the control's 0.180), but under
  the stripped reward Chinese phrases still score better than stripped English ones.
- **The corpus alone does little.** The control leg goes 0% → 1.4% on English inputs and mostly
  matches the source language elsewhere.
- **Stripping alone does nothing** (the strip ablation: 0/640 on ultrachat). It needs Chinese rows
  in RL to seed Chinese phrases, which it then rewards on English rows too. In rollouts, the share
  of phrases with an ideograph on non-Chinese rows climbs from 0% (steps 1–5) to 70% (steps 16–20)
  for strip, and to 30% for control. Few rows per bucket, so read the eval table, not this.
- Mechanism, consistent with the above: the reconstructor was trained on phrases that keep their
  leading space. Stripping pushes English phrases off its distribution, while CJK phrases have no
  leading space and are untouched. Once RL samples Chinese, it is the cheaper way to score.

Caveats: one seed per leg, 20 steps, 8 prompts per step. The reconstructor is ours (ultrachat, keeps
leading spaces), not agu18dec's AR, and this says nothing about the inverter stage or free-form
bullets. Whether the agu18dec AR has the same leading-space asymmetry is the next thing to check:
feed it the same English phrases with and without the leading space and compare FVE.
