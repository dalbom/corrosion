#!/bin/bash
# fix_cuda.sh - Upgrade PyTorch to CUDA 12.8 for Blackwell (5090) support.
# Run this on machines that already have the 'corrosion' env set up via initiate_exp.sh.
# Safe to run on 3090/4090 as well (backward compatible).

set -e

ENV_NAME="corrosion"

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

echo "=== Before ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Arch: {torch.cuda.get_device_capability(0)}')"

echo ""
echo "Upgrading PyTorch to CUDA 12.8..."
pip install --upgrade torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128

echo ""
echo "=== After ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Arch: {torch.cuda.get_device_capability(0)}')"

echo ""
echo "Done! Re-run your training script."
