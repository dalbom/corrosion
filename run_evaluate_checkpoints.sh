#!/bin/bash
# Evaluate all checkpoints in a WGAN-GP experiment directory using MACE
# Uses VALIDATION set (not test) for proper model selection
#
# Usage: ./run_evaluate_checkpoints.sh <experiment_dir>
# Example: ./run_evaluate_checkpoints.sh logs_ext/cGAN/20251215-210123_S11_wgangp

set -e

MICROMAMBA="/home/dalbom/.local/bin/micromamba"
ENV_NAME="py310"
VAL_CSV="datasets/Corrosion_cGAN_validation.csv"
IMG_ROOT="datasets/corrosion_img"

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_dir>"
    echo "Example: $0 logs_ext/cGAN/20251215-210123_S11_wgangp"
    echo ""
    echo "Available experiment directories:"
    ls -d logs_ext/cGAN/*_wgangp 2>/dev/null || echo "  (none found)"
    exit 1
fi

EXP_DIR="$1"

if [ ! -d "$EXP_DIR" ]; then
    echo "Error: Directory not found: $EXP_DIR"
    exit 1
fi

echo "========================================"
echo "Evaluating checkpoints in: $EXP_DIR"
echo "Validation CSV: $VAL_CSV"
echo "Image root: $IMG_ROOT"
echo "========================================"

$MICROMAMBA run -n $ENV_NAME python evaluate_checkpoints.py \
    --exp_dir "$EXP_DIR" \
    --val_csv "$VAL_CSV" \
    --img_root "$IMG_ROOT"

echo ""
echo "Evaluation complete!"
