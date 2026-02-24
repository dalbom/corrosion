#!/bin/bash
# initiate_exp.sh - Script to distribute experiments to other machines.
# First run: creates env, clones repo, downloads dataset, installs deps.
# Subsequent runs: activates env, pulls latest code, runs experiments.

set -e

# --- Parameters (Modify these as needed) ---
ENV_NAME="corrosion"
PYTHON_VERSION="3.12"
BATCH_SIZE=${BATCH_SIZE:-32}
LOG_DIR_BASE=${LOG_DIR_BASE:-"logs/"}
NUM_STEPS=${NUM_STEPS:-1000}
TRAIN_CSV=${TRAIN_CSV:-"./datasets/Corrosion_cGAN_train.csv"}
REPO_URL="https://github.com/dalbom/corrosion"
GDRIVE_FILE_ID="1Cn3QAFoJm8n8NsdOtSFRhBa3qGJjKn-A"
COMPILE_MODE=${COMPILE_MODE:-"reduce-overhead"}
# -------------------------------------------

# Helper function embedded to avoid external file dependency
train_model() {
    local channels="$@"
    local channel_str=$(echo $channels | tr ' ' '_')

    echo "========================================"
    echo "Training Diffusion with channels: $channels"
    echo "========================================"

    python train.py \
        --csv "$TRAIN_CSV" \
        --img_root "./datasets/corrosion_img" \
        --use_channels $channels \
        --image_size 128 \
        --timesteps 1000 \
        --sampling_timesteps 250 \
        --batch_size $BATCH_SIZE \
        --lr 8e-5 \
        --num_steps $NUM_STEPS \
        --log_every 100000 \
        --save_every $NUM_STEPS \
        --log_dir "$LOG_DIR_BASE/$channel_str" \
        --compile "$COMPILE_MODE"

    echo ""
}

# --- Setup (skipped if env and repo already exist) ---

# 1. Detect and setup environment
setup_env() {
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
}

activate_env() {
    if command -v micromamba >/dev/null 2>&1; then
        eval "$(micromamba shell hook --shell bash)"
        micromamba activate $ENV_NAME
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $ENV_NAME
    else
        source $ENV_NAME/bin/activate
    fi
}

# Check if this is a fresh setup or a re-run
NEEDS_SETUP=false

# Check if env exists
if command -v micromamba >/dev/null 2>&1; then
    if ! micromamba env list 2>/dev/null | grep -q "$ENV_NAME"; then
        NEEDS_SETUP=true
    fi
elif command -v conda >/dev/null 2>&1; then
    if ! conda env list 2>/dev/null | grep -q "$ENV_NAME"; then
        NEEDS_SETUP=true
    fi
else
    if [ ! -d "$ENV_NAME" ]; then
        NEEDS_SETUP=true
    fi
fi

if [ "$NEEDS_SETUP" = true ]; then
    echo "First run detected. Setting up environment..."

    # 1. Create environment
    setup_env

    # 2. Clone repo
    if [ ! -d "corrosion" ]; then
        git clone $REPO_URL
        cd corrosion
    else
        cd corrosion
        git pull
    fi

    # 2.5 Download dataset
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

    # 3. Install dependencies
    pip install --upgrade pip
    if [ -f "requirements.txt" ]; then
        pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
        pip install -r requirements.txt
    fi
    pip install -e .
else
    echo "Environment '$ENV_NAME' already exists. Skipping setup."
    activate_env
    cd corrosion
    git pull
fi

# 4. Run experiments
echo "Checking GPU availability..."
GPU_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$GPU_AVAILABLE" != "True" ]; then
    echo "CRITICAL ERROR: CUDA is not available in the current environment!"
    exit 1
fi

echo "GPU is available. Starting specified sensor combinations..."

# Machine #1
# train_model Phase11
# train_model Phase21
# train_model S11 S21
# train_model S11 Phase11

# # Machine #2
# train_model S11 Phase21
# train_model S21 Phase11
# train_model S21 Phase21

# # Machine #3
# train_model Phase11 Phase21
# train_model S11 S21 Phase11
# train_model S11 S21 Phase21

# # Machine #4
# train_model S11 Phase11 Phase21
# train_model S21 Phase11 Phase21
# train_model S11 S21 Phase11 Phase21

echo "========================================================="
echo "All experiments initiated successfully!"
echo "Please send the contents of the logs directory to Haebom."
echo "========================================================="
