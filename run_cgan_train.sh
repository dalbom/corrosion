#!/bin/bash
# Train WGAN-GP models for all 15 sensor combinations
# Uses Wasserstein distance with gradient penalty for stable training

set -e

MICROMAMBA="/home/dalbom/.local/bin/micromamba"
ENV_NAME="py310"
TRAIN_CSV="datasets/Corrosion_cGAN_train.csv"
VAL_CSV="datasets/Corrosion_cGAN_validation.csv"
IMG_ROOT="datasets"
EPOCHS=100
PATIENCE=50
BATCH_SIZE=32
N_CRITIC=2

# Function to train a single model
train_model() {
    local channels="$@"
    echo "========================================"
    echo "Training WGAN-GP with channels: $channels"
    echo "========================================"
    
    $MICROMAMBA run -n $ENV_NAME python train_wgan_gp.py \
        --csv $TRAIN_CSV \
        --val_csv $VAL_CSV \
        --img_root $IMG_ROOT \
        --use_channels $channels \
        --epochs $EPOCHS \
        --patience $PATIENCE \
        --batch_size $BATCH_SIZE \
        --n_critic $N_CRITIC
    
    echo ""
}

# Ensure data split exists
if [ ! -f "$TRAIN_CSV" ] || [ ! -f "$VAL_CSV" ]; then
    echo "Creating train/validation split..."
    $MICROMAMBA run -n $ENV_NAME python split_cgan_dataset.py
fi

echo "Training all 15 sensor combinations..."
echo ""

# 1-sensor combinations (4 models)
train_model S11
train_model S21
train_model Phase11
train_model Phase21

# # 2-sensor combinations (6 models)
train_model S11 S21
train_model S11 Phase11
train_model S11 Phase21
train_model S21 Phase11
train_model S21 Phase21
train_model Phase11 Phase21

# # 3-sensor combinations (4 models)
train_model S11 S21 Phase11
train_model S11 S21 Phase21
train_model S11 Phase11 Phase21
train_model S21 Phase11 Phase21

# # 4-sensor combination (1 model)
train_model S11 S21 Phase11 Phase21

echo "========================================"
echo "All 15 models trained successfully!"
echo "Checkpoints saved to: logs_ext/cGAN/"
echo "========================================"
