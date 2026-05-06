#!/bin/bash
# initiate_dit_exp.sh - Distribute DiT experiments to multiple machines.
# Assumes the corrosion repo and dataset already exist from prior DDPM/cGAN runs.
# Environment setup is kept for safety but skipped if env already exists.

set -e

# --- Parameters (Modify these as needed) ---
ENV_NAME="corrosion"
PYTHON_VERSION="3.12"
BATCH_SIZE=${BATCH_SIZE:-8}
LOG_DIR_BASE=${LOG_DIR_BASE:-"logs/dit"}
NUM_STEPS=${NUM_STEPS:-1000000}
TRAIN_CSV=${TRAIN_CSV:-"./datasets/Corrosion_cGAN_train.csv"}
VAL_CSV=${VAL_CSV:-"./datasets/Corrosion_cGAN_validation.csv"}
REPO_URL="https://github.com/dalbom/corrosion"
COMPILE_MODE=${COMPILE_MODE:-"reduce-overhead"}

# DiT architecture
PATCH_SIZE=${PATCH_SIZE:-4}
HIDDEN_SIZE=${HIDDEN_SIZE:-384}
DEPTH=${DEPTH:-12}
NUM_HEADS=${NUM_HEADS:-6}
LR=${LR:-1e-4}
# -------------------------------------------

train_model() {
    local channels="$@"
    local channel_str=$(echo $channels | tr ' ' '_')

    echo "========================================"
    echo "Training DiT with channels: $channels"
    echo "========================================"

    python train_dit.py \
        --csv "$TRAIN_CSV" \
        --val_csv "$VAL_CSV" \
        --img_root "./datasets/corrosion_img" \
        --use_channels $channels \
        --image_size 128 \
        --timesteps 1000 \
        --sampling_timesteps 250 \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --num_steps $NUM_STEPS \
        --log_every 100000 \
        --save_every $NUM_STEPS \
        --log_dir "$LOG_DIR_BASE/$channel_str" \
        --patch_size $PATCH_SIZE \
        --hidden_size $HIDDEN_SIZE \
        --depth $DEPTH \
        --num_heads $NUM_HEADS \
        --compile "$COMPILE_MODE"

    echo ""
}

# --- Setup (skipped if env and repo already exist) ---

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

# Check if env exists
NEEDS_SETUP=false
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
    setup_env

    if [ ! -d "corrosion" ]; then
        git clone $REPO_URL
    fi
    cd corrosion

    # Pull latest code (includes dit/ and train_dit.py)
    git pull

    # Install dependencies
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

    # Pull latest to get DiT code
    git pull
fi

# --- Run experiments ---
echo "Checking GPU availability..."
GPU_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")

if [ "$GPU_AVAILABLE" != "True" ]; then
    echo "CRITICAL ERROR: CUDA is not available in the current environment!"
    exit 1
fi

echo "GPU is available. Starting DiT training..."

# Machine #1
# train_model S11
# train_model S21
# train_model Phase11
# train_model Phase21

# Machine #2
# train_model S11 S21
# train_model S11 Phase11
# train_model S11 Phase21

# Machine #3
# train_model S21 Phase11
# train_model S21 Phase21
# train_model Phase11 Phase21
# train_model S11 S21 Phase11

# Machine #4
# train_model S11 S21 Phase21
# train_model S11 Phase11 Phase21
# train_model S21 Phase11 Phase21
# train_model S11 S21 Phase11 Phase21

echo "========================================================="
echo "All DiT experiments initiated successfully!"
echo "Please send the contents of the logs directory to Haebom."
echo "========================================================="
