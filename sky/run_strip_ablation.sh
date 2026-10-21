#!/usr/bin/env bash
# Leading-whitespace stripping ablation on the Qwen3.6 pilot (docs/strip-ablation.md). Needs 2 GPUs and, under
# /work/pilot: ckpt_oracle_sft, ckpt_recon (from HF), whitening, corpus/transcripts.jsonl, dictionary/len8_phrases.jsonl.
# Re-extracts only the rl/eval activations, then runs both 20-step RL legs from the SFT checkpoint in parallel
# (GPU 0 strips each phrase's leading whitespace before the reconstructor, GPU 1 is the control), then evals both.
set -uo pipefail
export HF_HOME=/hf PYTHONPATH=~/olens/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FLA_TILELANG=0
PY=~/venv/bin/python; CFG=~/olens/configs/pilot.yaml; ROOT=/work/pilot
mkdir -p $ROOT/logs

CUDA_VISIBLE_DEVICES=0 $PY -m olens.extract_acts --config $CFG -o "extract.target_positions={rl: 10000}" > $ROOT/logs/extract_rl.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -m olens.extract_acts --config $CFG -o "extract.target_positions={eval: 4000}" > $ROOT/logs/extract_eval.log 2>&1 &
wait
tail -2 $ROOT/logs/extract_*.log

leg() {  # gpu name strip
  local o=(-o oracle.rl.steps=20 -o oracle.rl.eval_every=10 -o oracle.rl.save_every=1000
           -o paths.ckpt_oracle_rl=ckpt_rl_$2 -o oracle.rl.strip_leading_ws=$3)
  CUDA_VISIBLE_DEVICES=$1 $PY -m olens.rl_oracle --config $CFG "${o[@]}" > $ROOT/logs/rl_$2.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=$1 $PY -m olens.eval_fve --config $CFG --ckpt $ROOT/ckpt_rl_$2 --rows 256 "${o[@]}" > $ROOT/logs/eval_$2.log 2>&1
  echo "leg $2 rc=$?"
}
leg 0 strip true &
leg 1 control false &
wait
# volumes don't outlive the pod: push adapters, logs, rollouts and the re-extracted activations to HF
~/venv/bin/hf upload robbiethompson2018/oracle-lens-qwen3.6-27b-pilot-strip-ablation $ROOT . \
  --include "ckpt_rl_*/**" "logs/**" "acts/rl.*" "acts/eval.*" > $ROOT/logs/hf_upload.log 2>&1 && echo HF_UPLOADED
echo ABLATION_DONE
