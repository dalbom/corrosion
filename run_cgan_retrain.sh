#!/bin/bash
# cGAN: Train trials 2-5 on 5090 (trial1 already done)
# Each trial trains all 15 sensor combos for 100 epochs, no early stopping.
# After each trial, only best_model.pt is kept; epoch checkpoints are deleted.
#
# Usage:
#   bash run_cgan_retrain.sh 0   # GPU 0: trials 2,3
#   bash run_cgan_retrain.sh 1   # GPU 1: trials 4,5
set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

if [ "$GPU_ID" = "0" ]; then
    TRIALS=(2 3)
else
    TRIALS=(4 5)
fi

echo "GPU: $GPU_ID | Trials: ${TRIALS[@]}"

ENV_NAME="corrosion"
if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate $ENV_NAME
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate $ENV_NAME
else
    source $ENV_NAME/bin/activate
fi

cd corrosion

TRAIN_CSV="datasets/Corrosion_cGAN_train.csv"
VAL_CSV="datasets/Corrosion_cGAN_validation.csv"
IMG_ROOT="datasets"
EPOCHS=100
BATCH_SIZE=32
N_CRITIC=2
MACE_SAMPLE_SIZE=64

ALL_SENSORS=(
    "S11" "S21" "Phase11" "Phase21"
    "S11 S21" "S11 Phase11" "S11 Phase21" "S21 Phase11" "S21 Phase21" "Phase11 Phase21"
    "S11 S21 Phase11" "S11 S21 Phase21" "S11 Phase11 Phase21" "S21 Phase11 Phase21"
    "S11 S21 Phase11 Phase21"
)

train_model() {
    local trial=$1; shift
    local channels="$@"
    local channel_str=$(echo $channels | tr ' ' '_')
    local trial_dir="checkpoints/baseline/cgan/trial${trial}"
    echo ""
    echo "[cGAN] trial$trial $channel_str"
    PYTHONUNBUFFERED=1 python train_cgan.py \
        --csv "$TRAIN_CSV" \
        --val_csv "$VAL_CSV" \
        --img_root "$IMG_ROOT" \
        --use_channels $channels \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --n_critic $N_CRITIC \
        --mace_sample_size $MACE_SAMPLE_SIZE \
        --log_dir "$trial_dir"

    # Cleanup: remove epoch checkpoints, keep only best_model.pt
    latest_exp=$(ls -dt "$trial_dir"/*_${channel_str}_wgangp 2>/dev/null | head -1)
    if [ -n "$latest_exp" ]; then
        find "$latest_exp" -name "*.pt" ! -name "best_model.pt" -delete 2>/dev/null
        echo "  Cleaned up epoch checkpoints in $(basename $latest_exp)"
    fi
}

echo "=========================================="
echo "cGAN Training: trials 2-5 (100 epochs each)"
echo "=========================================="

for trial in "${TRIALS[@]}"; do
    echo ""
    echo "############################################"
    echo "# Trial $trial"
    echo "############################################"

    TRIAL_DIR="checkpoints/baseline/cgan/trial${trial}"
    mkdir -p "$TRIAL_DIR"

    # Remove old failed checkpoints if any
    if [ -d "$TRIAL_DIR" ] && [ "$(ls -A $TRIAL_DIR 2>/dev/null)" ]; then
        echo "Clearing old trial$trial checkpoints..."
        rm -rf "$TRIAL_DIR"/*
    fi

    for sensor_args in "${ALL_SENSORS[@]}"; do
        train_model $trial $sensor_args
    done

    echo "Trial $trial complete."
done

echo ""
echo "All cGAN training complete!"
echo "Trials saved to checkpoints/baseline/cgan/trial{2,3,4,5}/"
