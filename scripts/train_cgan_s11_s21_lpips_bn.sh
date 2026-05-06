#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/recon.sh" train --config configs/corrosion/cgan_s11_s21_lpips_bn.yaml
