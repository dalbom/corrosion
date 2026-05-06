#!/bin/bash
# initiate_debug.sh - Benchmark torch.compile modes on 5090.
# Runs train_debug.py for 2 epochs (848 steps) with three compile options.
# Compare the SECOND epoch timing (first epoch includes compilation overhead).

set -e

# --- Parameters ---
ENV_NAME="corrosion"
PYTHON_VERSION="3.12"
BATCH_SIZE=${BATCH_SIZE:-8}
LOG_DIR_BASE=${LOG_DIR_BASE:-"logs_debug"}
TRAIN_CSV=${TRAIN_CSV:-"./datasets/Corrosion_cGAN_train.csv"}
REPO_URL="https://github.com/dalbom/corrosion"
GDRIVE_FILE_ID="1Cn3QAFoJm8n8NsdOtSFRhBa3qGJjKn-A"

# 3392 samples / batch_size=8 / drop_last = 424 steps/epoch → 2 epochs = 848 steps
NUM_STEPS=848
# Set log_every > NUM_STEPS to skip validation sampling during benchmark
LOG_EVERY=10000
CHANNELS="S11"
# ------------------

run_benchmark() {
    local compile_mode="$1"

    echo ""
    echo "############################################################"
    echo "  BENCHMARK: --compile $compile_mode"
    echo "############################################################"
    echo ""

    python train_debug.py \
        --csv "$TRAIN_CSV" \
        --img_root "./datasets/corrosion_img" \
        --use_channels $CHANNELS \
        --image_size 128 \
        --timesteps 1000 \
        --sampling_timesteps 250 \
        --batch_size $BATCH_SIZE \
        --lr 8e-5 \
        --num_steps $NUM_STEPS \
        --log_every $LOG_EVERY \
        --save_every 0 \
        --log_dir "$LOG_DIR_BASE/compile_${compile_mode}" \
        --compile "$compile_mode"

    echo ""
    echo ">>> Finished: --compile $compile_mode"
    echo ""
}

# 1. Detect and setup environment
if command -v micromamba >/dev/null 2>&1; then
    echo "Using micromamba..."
    micromamba create -n $ENV_NAME python=$PYTHON_VERSION -y
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate $ENV_NAME
elif command -v conda >/dev/null 2>&1; then
    echo "Using conda..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate $ENV_NAME
else
    echo "Neither micromamba nor conda found. Using venv..."
    python3.12 -m venv $ENV_NAME || python3 -m venv $ENV_NAME
    source $ENV_NAME/bin/activate
fi

# 2. git clone this repo
if [ ! -d "corrosion" ]; then
    git clone $REPO_URL
    cd corrosion
else
    cd corrosion
    git pull
fi

# 2.5 Download and extract dataset from Google Drive
if [ ! -d "datasets" ]; then
    echo "Dataset (datasets folder) not found. Downloading from Google Drive..."
    pip install gdown
    gdown --id $GDRIVE_FILE_ID -O datasets.tar

    if [ -f "datasets.tar" ]; then
        echo "Extracting dataset..."
        tar -xvf datasets.tar
        rm datasets.tar
        echo "Dataset setup complete."
    else
        echo "ERROR: Failed to download dataset from Google Drive."
        echo "Please download manually from: https://drive.google.com/file/d/$GDRIVE_FILE_ID/view"
        echo "Extract and place the 'datasets' folder in the 'corrosion' directory."
        exit 1
    fi
fi

# 3. install dependencies
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt
fi
pip install -e .

# 4. Check GPU
echo "Checking GPU availability..."
GPU_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$GPU_AVAILABLE" != "True" ]; then
    echo "CRITICAL ERROR: CUDA is not available in the current environment!"
    exit 1
fi

echo ""
echo "========================================================="
echo "  5090 Compile Benchmark: 2 epochs x 3 compile modes"
echo "  Channels: $CHANNELS | Batch: $BATCH_SIZE | Steps: $NUM_STEPS"
echo "  Compare the SECOND epoch timing across runs."
echo "========================================================="
echo ""

# 5. Run benchmarks
run_benchmark none
run_benchmark default
run_benchmark reduce-overhead

echo "========================================================="
echo "All benchmarks completed!"
echo "Logs saved to: $LOG_DIR_BASE/"
echo "Please send the terminal output to Haebom."
echo "========================================================="
