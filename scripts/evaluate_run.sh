#!/bin/bash
set -euo pipefail

RUN_DIR="${1:?usage: scripts/evaluate_run.sh runs/domain/method/study/sensor_set/recipe/seed/run_id}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/recon.sh" evaluate --run "$RUN_DIR"
