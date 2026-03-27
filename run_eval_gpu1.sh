#!/bin/bash
# GPU 1: DiT — 15 sensors × 5 trials (skips missing checkpoints)
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

TEST_CSV="datasets/Corrosion_test.csv"
IMG_ROOT="datasets/corrosion_img"
TRAIN_CSV="datasets/Corrosion_cGAN_train.csv"
SEEDS=(42 123 456 789 1024)

ALL_SENSORS=(
    "S11" "S21" "Phase11" "Phase21"
    "S11_S21" "S11_Phase11" "S11_Phase21" "S21_Phase11" "S21_Phase21" "Phase11_Phase21"
    "S11_S21_Phase11" "S11_S21_Phase21" "S11_Phase11_Phase21" "S21_Phase11_Phase21"
    "S11_S21_Phase11_Phase21"
)

echo "=========================================="
echo "GPU 1: DiT evaluation pipeline"
echo "=========================================="

for trial in 1 2 3 4 5; do
    seed=${SEEDS[$((trial-1))]}
    echo ""
    echo "--- DiT Trial $trial (seed=$seed) ---"

    for sensor in "${ALL_SENSORS[@]}"; do
        channels=$(echo "$sensor" | tr '_' ' ')
        ckpt=$(find "checkpoints/baseline/dit/$sensor" -name "model_step_1000000.pt" -type f 2>/dev/null | head -1)
        outdir="results/baseline/dit/trial${trial}/${sensor}"
        corrdir="results/corrected/dit/trial${trial}/${sensor}"

        # Skip if no checkpoint
        if [ -z "$ckpt" ]; then
            echo "  SKIP $sensor: no checkpoint"
            continue
        fi

        # Skip if already generated
        if [ -d "$outdir" ] && [ "$(find "$outdir" -name '*.png' 2>/dev/null | wc -l)" -ge 1000 ]; then
            echo "  SKIP $sensor: already generated"
        else
            echo "  [Inference] $sensor"
            PYTHONUNBUFFERED=1 python -c "
import torch; torch.manual_seed($seed); torch.cuda.manual_seed_all($seed)
import sys; sys.argv = ['inference_dit.py',
    '--checkpoint', '$ckpt',
    '--csv', '$TEST_CSV',
    '--output', '$outdir',
    '--img_root', '$IMG_ROOT',
    '--use_channels'] + '$channels'.split() + [
    '--image_size', '128',
    '--sampling_timesteps', '250',
    '--batch_size', '16']
from inference_dit import parse_args, main
main(parse_args())
"
        fi

        # Histogram matching
        if [ -d "$corrdir" ] && [ "$(find "$corrdir" -name '*.png' 2>/dev/null | wc -l)" -ge 1000 ]; then
            echo "  SKIP $sensor: already corrected"
        else
            echo "  [Histogram] $sensor"
            python run_histogram_matching.py \
                --gen_dir "$outdir" \
                --corrected_dir "$corrdir" \
                --train_csv "$TRAIN_CSV" \
                --img_root "$IMG_ROOT"
        fi
    done

    # Evaluation for this trial
    out_csv="output/notion/DiT_${trial}.csv"
    mkdir -p output/notion
    echo "Index,Sensor Type,MAE,MSE,PSNR,SSIM,MACE,비고" > "$out_csv"

    for sensor_dir in results/corrected/dit/trial${trial}/*/; do
        [ -d "$sensor_dir" ] || continue
        sensor=$(basename "$sensor_dir")
        sensor_display=$(echo "$sensor" | sed 's/_/, /g')

        result=$(python evaluate_metrics.py \
            --csv "$TEST_CSV" --img_root "$IMG_ROOT" \
            --gen_root "results/corrected/dit/trial${trial}/$sensor" \
            --sensor "$sensor" --table 2>/dev/null)
        mae=$(echo "$result" | awk -F'\t' '{print $2}')
        mse=$(echo "$result" | awk -F'\t' '{print $3}')
        psnr=$(echo "$result" | awk -F'\t' '{print $4}')
        ssim=$(echo "$result" | awk -F'\t' '{print $5}')
        mace=$(echo "$result" | awk -F'\t' '{print $6}')

        echo "${sensor},\"${sensor_display}\",${mae},${mse},${psnr},${ssim},${mace}," >> "$out_csv"
    done
    echo "  Saved: $out_csv"
done

echo ""
echo "GPU 1: DiT pipeline complete!"
