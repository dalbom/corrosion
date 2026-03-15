#!/bin/bash
# Improved DiT training — GPU 0 (7 of 14 remaining channel combos)
# S11_Phase21 already trained; these are the rest
set -e
export CUDA_VISIBLE_DEVICES=0

ENV_NAME="corrosion"
TRAIN_CSV="./datasets/Corrosion_cGAN_train.csv"
VAL_CSV="./datasets/Corrosion_cGAN_validation.csv"
IMG_ROOT="./datasets/corrosion_img"

# Activate environment
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

COMMON_ARGS="--model_type dit --csv $TRAIN_CSV --val_csv $VAL_CSV --img_root $IMG_ROOT \
    --batch_size 32 --lr 2e-4 --num_steps 250000 \
    --p_uncond 0.1 --lambda_mace 10.0 --guidance_scale 2.0 --objective pred_x0 \
    --log_every 25000 --save_every 100000"

echo "=========================================="
echo "Improved DiT Training — GPU 0 (7 combos)"
echo "=========================================="

echo "[1/7] S11"
python improved/train_improved.py $COMMON_ARGS --use_channels S11

echo "[2/7] S21"
python improved/train_improved.py $COMMON_ARGS --use_channels S21

echo "[3/7] Phase11"
python improved/train_improved.py $COMMON_ARGS --use_channels Phase11

echo "[4/7] Phase21"
python improved/train_improved.py $COMMON_ARGS --use_channels Phase21

echo "[5/7] S11 S21"
python improved/train_improved.py $COMMON_ARGS --use_channels S11 S21

echo "[6/7] S11 Phase11"
python improved/train_improved.py $COMMON_ARGS --use_channels S11 Phase11

echo "[7/7] S21 Phase11"
python improved/train_improved.py $COMMON_ARGS --use_channels S21 Phase11

echo ""
echo "GPU 0 — All 7 combos done!"
