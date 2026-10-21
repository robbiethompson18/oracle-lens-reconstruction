#!/usr/bin/env bash
# Chinese-input probe on the Qwen3.6 pilot (docs/zh-probe.md). Needs 2 GPUs and, under /work/pilot: ckpt_recon,
# ckpt_oracle_sft, ckpt_oracle_rl, ckpt_rl_{strip,control} (strip ablation), acts/eval.* (English), whitening,
# dictionary/len8_phrases.jsonl.
#   exp1: Qwen answers Chinese WildChat prompts (/work/zh) and natural-mix WildChat prompts (/work/wc, for exp2);
#         the four existing lenses read the Chinese and the English eval activations.
#   exp2: the 20-step strip/control RL from the SFT checkpoint again, on natural-mix WildChat activations.
set -uo pipefail
export PATH=~/venv/bin:$PATH HF_HOME=/hf PYTHONPATH=~/olens/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FLA_TILELANG=0
PY=~/venv/bin/python; CFG=~/olens/configs/pilot.yaml; P=/work/pilot
HF_REPO=robbiethompson2018/oracle-lens-qwen3.6-27b-pilot-zh-probe
mkdir -p $P/logs
WC=(-o corpus.prompt_dataset=allenai/WildChat-1M -o corpus.prompt_split=train)
ZH=("${WC[@]}" -o paths.corpus=/work/zh/corpus -o paths.acts=/work/zh/acts -o corpus.language=Chinese
    -o corpus.n_transcripts=600 -o "corpus.splits={eval: 1.0}" -o extract.positions_per_transcript=4 -o "extract.target_positions={eval: 2000}")
MIX=("${WC[@]}" -o paths.corpus=/work/wc/corpus -o paths.acts=/work/wc/acts -o corpus.n_transcripts=2000
     -o "corpus.splits={rl: 0.85, eval: 0.15}" -o extract.positions_per_transcript=8 -o "extract.target_positions={rl: 10000, eval: 1000}")
EVAL=(--rows 256 --temperature 0 --temperature 1 --acts /work/zh/acts)

data() {  # gpu name overrides...
  local g=$1 n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $PY -m olens.gen_corpus --config $CFG "$@" > $P/logs/corpus_$n.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=$g $PY -m olens.extract_acts --config $CFG "$@" > $P/logs/extract_$n.log 2>&1
  echo "data $n rc=$?"
}

evals() {  # gpu ckpt [overrides...]
  local g=$1 c=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $PY -m olens.eval_fve --config $CFG --ckpt $c "${EVAL[@]}" "$@" > $P/logs/eval_$(basename $c).log 2>&1
  echo "eval $(basename $c) rc=$?"
}

upload() {  # volumes don't outlive the pod
  ~/venv/bin/hf upload $HF_REPO /work . --include "pilot/logs/**" --include "pilot/ckpt_*/eval_*.json" --include "zh/**" \
    --include "wc/**" > $P/logs/hf_upload_$1.log 2>&1 && echo "HF_UPLOADED $1"
}

if [ "${1:-}" = exp1 ]; then
  [ -f /work/wc/acts/eval.npy ] || { data 0 zh "${ZH[@]}" & data 1 wc "${MIX[@]}" & wait; }
  (evals 0 $P/ckpt_oracle_sft; evals 0 $P/ckpt_oracle_rl) &
  (evals 1 $P/ckpt_rl_control; evals 1 $P/ckpt_rl_strip -o oracle.rl.strip_leading_ws=true) &
  wait
  upload exp1
  echo EXP1_DONE
elif [ "${1:-}" = exp2 ]; then
  leg() {  # gpu name strip
    local o=(-o paths.acts=/work/wc/acts -o oracle.rl.steps=20 -o oracle.rl.eval_every=10 -o oracle.rl.save_every=1000
             -o paths.ckpt_oracle_rl=/work/wc/ckpt_rl_$2 -o oracle.rl.strip_leading_ws=$3)
    CUDA_VISIBLE_DEVICES=$1 $PY -m olens.rl_oracle --config $CFG "${o[@]}" > $P/logs/rl_wc_$2.log 2>&1 &&
      evals $1 /work/wc/ckpt_rl_$2 "${o[@]}" --acts $P/acts --acts /work/wc/acts --rows 128
    echo "leg $2 rc=$?"
  }
  leg 0 strip true &
  leg 1 control false &
  wait
  upload exp2
  echo EXP2_DONE
fi
