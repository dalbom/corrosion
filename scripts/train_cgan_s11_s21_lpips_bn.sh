#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

micromamba run -n py310 python -m sensor_image_recon.cli train \
  --config configs/corrosion/cgan_s11_s21_lpips_bn.yaml
