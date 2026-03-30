#!/bin/bash
# GPU 0: Retrain missing DDPM baselines (3 combos)
# DDPM: Phase11_Phase21, S11_S21_Phase11, S11_S21_Phase21
set -e
export CUDA_VISIBLE_DEVICES=0

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
echo "GPU 0: Retrain missing DDPM baselines"
echo "=========================================="

# Settings: batch=8, lr=8e-5, 1M steps, pred_noise, compile=reduce-overhead

train_ddpm() {
    local channels="$@"
    local channel_str=$(echo $channels | tr ' ' '_')
    echo ""
    echo "[DDPM] $channel_str"
    PYTHONUNBUFFERED=1 python train.py \
        --csv "$TRAIN_CSV" \
        --val_csv "$VAL_CSV" \
        --img_root "$IMG_ROOT" \
        --use_channels $channels \
        --image_size 128 \
        --timesteps 1000 \
        --sampling_timesteps 250 \
        --batch_size 8 \
        --lr 8e-5 \
        --num_steps 1000000 \
        --log_every 100000 \
        --save_every 1000000 \
        --log_dir "checkpoints/baseline/ddpm/$channel_str" \
        --compile "reduce-overhead"
}

train_ddpm Phase11 Phase21
train_ddpm S11 S21 Phase11
train_ddpm S11 S21 Phase21

echo ""
echo "GPU 0: All DDPM retraining complete!"
