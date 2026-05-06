#!/bin/bash
# GPU 1: Retrain missing DiT baselines (7 combos)
# DiT: S11, S21, Phase11, Phase21, S11_Phase11, S11_Phase21, S11_S21
set -e
export CUDA_VISIBLE_DEVICES=1

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

TRAIN_CSV="./datasets/Corrosion_cGAN_train.csv"
VAL_CSV="./datasets/Corrosion_cGAN_validation.csv"
IMG_ROOT="./datasets/corrosion_img"

echo "=========================================="
echo "GPU 1: Retrain missing DiT baselines (7 combos)"
echo "=========================================="

train_dit() {
    local channels="$@"
    local channel_str=$(echo $channels | tr ' ' '_')
    echo ""
    echo "[DiT] $channel_str"
    PYTHONUNBUFFERED=1 python train_dit.py \
        --csv "$TRAIN_CSV" \
        --val_csv "$VAL_CSV" \
        --img_root "$IMG_ROOT" \
        --use_channels $channels \
        --image_size 128 \
        --timesteps 1000 \
        --sampling_timesteps 250 \
        --batch_size 8 \
        --lr 1e-4 \
        --num_steps 1000000 \
        --log_every 100000 \
        --save_every 1000000 \
        --log_dir "checkpoints/baseline/dit/$channel_str" \
        --patch_size 4 \
        --hidden_size 384 \
        --depth 12 \
        --num_heads 6 \
        --compile "reduce-overhead"
}

train_dit S11
train_dit S21
train_dit Phase11
train_dit Phase21
train_dit S11 Phase11
train_dit S11 Phase21
train_dit S11 S21

echo ""
echo "GPU 1: All DiT retraining complete!"
