#!/usr/bin/env bash
# Ship src/ + configs/ + sky/run_*.sh to the pod (no SkyPilot workdir: uploads via the API server fail).
set -euo pipefail
cd "$(dirname "$0")/.."
COPYFILE_DISABLE=1 tar cf - src configs sky/run_*.sh pyproject.toml | ssh -T "${1:-olens}" 'mkdir -p ~/olens && tar xf - -C ~/olens && echo shipped'
