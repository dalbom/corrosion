#!/bin/bash
set -euo pipefail

RUN_DIR="${1:?usage: scripts/evaluate_run.sh runs/domain/method/experiment/run_id}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

micromamba run -n py310 python -m sensor_image_recon.cli evaluate --run "$RUN_DIR"
