#!/bin/bash
# cGAN pilot 6: 2ch LPIPS + BatchNorm cond input.
# Test if BN with learnable affine matches/exceeds 2ch LPIPS no-norm
# (5.33 raw MACE) by recovering absolute magnitude info that dataset z-score lost.
set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

MICROMAMBA="/home/dalbom/.local/bin/micromamba"
ENV="py310"

TRAIN_CSV="datasets/Corrosion_cGAN_train.csv"
VAL_CSV="datasets/Corrosion_cGAN_validation.csv"
TEST_CSV="datasets/Corrosion_test.csv"
TRAIN_IMG_ROOT="datasets"
EVAL_IMG_ROOT="datasets/corrosion_img"

TRIAL=1
CHANNELS="S11 S21"
CHANNEL_STR="S11_S21"
TRIAL_DIR="checkpoints/baseline/cgan/trial${TRIAL}"

EPOCHS=100
BATCH_SIZE=32
N_CRITIC=2
MACE_SAMPLE_SIZE=64
LAMBDA_L1=100.0
LAMBDA_SSIM=50.0
LAMBDA_PERC=10.0

RAW_NEW="results/baseline/cgan/trial1_lpips_bn/${CHANNEL_STR}"
MATCHED_NEW="results/corrected/cgan/trial1_lpips_bn/${CHANNEL_STR}"
RAW_LPIPS_NONORM="results/baseline/cgan/trial1_lpips/${CHANNEL_STR}"
MATCHED_LPIPS_NONORM="results/corrected/cgan/trial1_lpips/${CHANNEL_STR}"
RAW_LPIPS_ZSCORE="results/baseline/cgan/trial1_lpips_norm/${CHANNEL_STR}"
MATCHED_LPIPS_ZSCORE="results/corrected/cgan/trial1_lpips_norm/${CHANNEL_STR}"
RAW_BASE="results/baseline/cgan/trial${TRIAL}/${CHANNEL_STR}"
MATCHED_BASE="results/corrected/cgan/trial${TRIAL}/${CHANNEL_STR}"

echo "=========================================="
echo "cGAN pilot 6: 2ch LPIPS + BatchNorm cond"
echo "GPU: $GPU_ID, channels: $CHANNELS, cond_norm_type=batchnorm"
echo "=========================================="

echo ""
echo "[1/5] Training..."
PYTHONUNBUFFERED=1 $MICROMAMBA run -n $ENV python train_cgan.py \
    --csv "$TRAIN_CSV" \
    --val_csv "$VAL_CSV" \
    --img_root "$TRAIN_IMG_ROOT" \
    --use_channels $CHANNELS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --n_critic $N_CRITIC \
    --mace_sample_size $MACE_SAMPLE_SIZE \
    --lambda_l1 $LAMBDA_L1 \
    --lambda_ssim $LAMBDA_SSIM \
    --lambda_perceptual $LAMBDA_PERC \
    --cond_norm_type batchnorm \
    --log_dir "$TRIAL_DIR"

NEW_CKPT_DIR=$(ls -dt ${TRIAL_DIR}/*_${CHANNEL_STR}_wgangp_l1ssim_perc_bn 2>/dev/null | head -1)
NEW_CKPT="${NEW_CKPT_DIR}/best_model.pt"
echo "  New checkpoint: $NEW_CKPT"
find "$NEW_CKPT_DIR" -name "checkpoint_epoch_*.pt" -delete 2>/dev/null || true

echo ""
echo "[2/5] Inference..."
rm -rf "$RAW_NEW" "$MATCHED_NEW"
PYTHONUNBUFFERED=1 $MICROMAMBA run -n $ENV python inference_cgan.py \
    --checkpoint "$NEW_CKPT" \
    --test_csv "$TEST_CSV" \
    --output_dir "$RAW_NEW" \
    --img_root "$TRAIN_IMG_ROOT" \
    --use_channels $CHANNELS \
    --batch_size 16

echo ""
echo "[3/5] Histogram matching..."
$MICROMAMBA run -n $ENV python run_histogram_matching.py \
    --gen_dir "$RAW_NEW" \
    --corrected_dir "$MATCHED_NEW" \
    --train_csv "$TRAIN_CSV" \
    --img_root "$EVAL_IMG_ROOT"

echo ""
echo "[4/5] Evaluating..."

echo ""
echo "[5/5] Comparison table"
echo "=========================================="
printf "Variant\t\t\tMAE\tMSE\tPSNR\tSSIM\tMACE\n"

eval_one() {
    local label="$1"
    local gen_root="$2"
    if [ -d "$gen_root" ]; then
        local result=$($MICROMAMBA run -n $ENV python evaluate_metrics.py \
            --csv "$TEST_CSV" --img_root "$EVAL_IMG_ROOT" \
            --gen_root "$gen_root" --sensor "$label" --table 2>/dev/null)
        echo "$result"
    else
        echo "$label	(missing dir)"
    fi
}

eval_one "base_2ch_raw    " "$RAW_BASE"
eval_one "base_2ch_matched" "$MATCHED_BASE"
eval_one "lpips_2ch_raw   " "$RAW_LPIPS_NONORM"
eval_one "lpips_2ch_match " "$MATCHED_LPIPS_NONORM"
eval_one "lpips_2ch_zsc_r " "$RAW_LPIPS_ZSCORE"
eval_one "lpips_2ch_zsc_m " "$MATCHED_LPIPS_ZSCORE"
eval_one "lpips_2ch_bn_r  " "$RAW_NEW"
eval_one "lpips_2ch_bn_m  " "$MATCHED_NEW"

echo "=========================================="
echo ""
echo "Acceptance: lpips_2ch_bn should match or improve over lpips_2ch_raw (5.33 MACE)."
echo "New checkpoint: $NEW_CKPT"
