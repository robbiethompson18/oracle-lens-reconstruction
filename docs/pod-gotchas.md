# Pod gotchas (RunPod + SkyPilot + Qwen3.6-27B)

Read before launching or debugging a pod run.

- **RunPod credentials on the Mac**: SkyPilot wants `~/.runpod/config.toml` in the `runpod config`
  format (`[default]` header, `api_key = ...`, `api_url = ...`). The older `apikey=` format is
  silently ignored and `sky check` reports RunPod disabled.
- **GPU availability**: US H200/B200 were unavailable on 2026-10-15; `sky/pod.yaml` falls back to
  H100 SXM. Launch with `--infra runpod/US --retry-until-up`.
- **Driver**: pod had driver 595 (CUDA 13.2); vLLM 0.31 pins torch 2.13 cu130, which needs ≥ 580.
- **No nvcc on the image**: see `CODE_SMELL.md` (causal_conv1d, FLA_TILELANG=0).
- **vLLM on 80 GB**: Qwen3.6's Mamba cache allows ~326 concurrent sequences at
  `gpu_memory_utilization 0.9`; `max_num_seqs` must be at most that or engine init fails.
- **Memory**: reconstructor stage with rank-64 LoRA on all linears (493M params) needs gradient
  checkpointing and batch ≤ 32 on 80 GB. Peak ~66 GB with those.
- **Model class**: `AutoModelForCausalLM` loads the checkpoint as `Qwen3_5ForCausalLM` (53.8 GB,
  zero missing keys under transformers 5.17). `common.load_model` asserts this.
- **Shipping code**: `sky/ship.sh` (tar over ssh). SkyPilot workdir uploads fail through the local
  API server.
- **Logs**: `/work/pilot/logs/<stage>.log` on the pod; `run_pilot.log` is the stage runner.
- **Setup log path**: `sky launch --detach-run` returns before setup finishes; the setup log is
  `~/sky_logs/<job-id>-<task>/tasks/setup.log` on the pod (not `~/sky_logs/*/setup-*.log`). It ends
  with `SETUP_DONE`. A waiter on the wrong glob idled a pod for 8 h ($32) on 2026-10-16.
- **Runner output**: `run_pilot.sh` prints stage headers to stdout; when started in tmux, mirror
  the pane with `tmux pipe-pane -t <session> -o "cat >> $ROOT/logs/run_pilot.log"`.
- **Qwen3.8-27B**: same `qwen3_5` architecture, same marker/`<|im_end|>` ids, same chat-template
  thinking switch; `configs/qwen38_pilot.yaml` differs from `pilot.yaml` only in model name and root.
