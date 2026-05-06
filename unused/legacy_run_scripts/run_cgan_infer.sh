#!/bin/bash
# Run inference for all 15 cGAN models
# Generates images to generated_cGAN/<sensor_combination>/<specimen_idx>/

set -e

MICROMAMBA="/home/dalbom/.local/bin/micromamba"
ENV_NAME="py310"
CHECKPOINT_DIR="logs_ext/cGAN"
TEST_CSV="datasets/Corrosion_test.csv"
OUTPUT_BASE="generated_cGAN"

# Function to run inference for a single model
run_inference() {
    local checkpoint_subdir="$1"
    local sensor_combo="$2"
    
    local checkpoint_path="${CHECKPOINT_DIR}/${checkpoint_subdir}/best_model.pt"
    local output_dir="${OUTPUT_BASE}/${sensor_combo}"
    
    if [ ! -f "$checkpoint_path" ]; then
        echo "WARNING: Checkpoint not found: $checkpoint_path"
        return
    fi
    
    echo "========================================"
    echo "Inference: $sensor_combo"
    echo "Checkpoint: $checkpoint_path"
    echo "Output: $output_dir"
    echo "========================================"
    
    $MICROMAMBA run -n $ENV_NAME python inference_cgan.py \
        --checkpoint "$checkpoint_path" \
        --test_csv "$TEST_CSV" \
        --output_dir "$output_dir"
    
    echo ""
}

echo "Running inference for all 15 WGAN-GP models..."
echo ""

# 1-sensor combinations (4 models)
run_inference "20260117-081620_S11_wgangp" "S11"
run_inference "20260117-085712_S21_wgangp" "S21"
run_inference "20260117-094204_Phase11_wgangp" "Ph11"
run_inference "20260117-103916_Phase21_wgangp" "Ph21"

# 2-sensor combinations (6 models)
run_inference "20260117-105255_S11_S21_wgangp" "S11_S21"
run_inference "20260117-113957_S11_Phase11_wgangp" "S11_Ph11"
run_inference "20260117-123056_S11_Phase21_wgangp" "S11_Ph21"
run_inference "20260117-132036_S21_Phase11_wgangp" "S21_Ph11"
run_inference "20260117-135758_S21_Phase21_wgangp" "S21_Ph21"
run_inference "20260117-142522_Phase11_Phase21_wgangp" "Ph11_Ph21"

# 3-sensor combinations (4 models)
run_inference "20260117-151935_S11_S21_Phase11_wgangp" "S11_S21_Ph11"
run_inference "20260117-161246_S11_S21_Phase21_wgangp" "S11_S21_Ph21"
run_inference "20260117-170836_S11_Phase11_Phase21_wgangp" "S11_Ph11_Ph21"
run_inference "20260117-172224_S21_Phase11_Phase21_wgangp" "S21_Ph11_Ph21"

# 4-sensor combination (1 model)
run_inference "20260117-173651_S11_S21_Phase11_Phase21_wgangp" "S11_S21_Ph11_Ph21"

echo "========================================"
echo "All inference complete!"
echo "Generated images saved to: ${OUTPUT_BASE}/"
echo "========================================"

