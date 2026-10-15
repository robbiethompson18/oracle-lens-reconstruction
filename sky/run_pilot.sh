#!/usr/bin/env bash
# Run the whole pilot on the pod, one stage after another, logging to /work/pilot/logs/<stage>.log.
# Usage on the pod: bash ~/olens/sky/run_pilot.sh [first_stage]
set -uo pipefail
export HF_HOME=/hf PYTHONPATH=~/olens/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0 FLA_TILELANG=0  # tilelang backend needs nvcc; pod has none
PY=~/venv/bin/python; CFG=~/olens/configs/pilot.yaml
mkdir -p /work/pilot/logs
stages=(smoke gen_corpus extract_acts train_reconstructor build_dictionary teacher_omp train_oracle_sft rl_oracle)
start=${1:-smoke}; go=0
for s in "${stages[@]}"; do
  [ "$s" = "$start" ] && go=1
  [ $go = 1 ] || continue
  echo "=== $s $(date -u +%H:%M:%S)"
  $PY -m olens.$s --config $CFG > /work/pilot/logs/$s.log 2>&1
  rc=$?
  tail -3 /work/pilot/logs/$s.log
  [ $rc = 0 ] || { echo "STAGE $s FAILED rc=$rc"; exit $rc; }
done
$PY -m olens.eval_fve --config $CFG --ckpt /work/pilot/ckpt_oracle_sft --rows 256 > /work/pilot/logs/eval_sft.log 2>&1
$PY -m olens.eval_fve --config $CFG --ckpt /work/pilot/ckpt_oracle_rl --rows 256 > /work/pilot/logs/eval_rl.log 2>&1
echo PILOT_DONE
