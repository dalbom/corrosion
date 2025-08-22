## Corrosion Diffusion: Conditional Image Generation from RF Measurements

This repository trains and evaluates a conditional diffusion model that synthesizes corrosion images from RF measurement vectors. It adapts `denoising_diffusion_pytorch` to condition a UNet on continuous features such as `S11`, `S21`, `Phase11`, and `Phase21`.

- Single‑channel target images are used (red channel of RGB).
- Conditioning is a concatenated float vector (length 201 per selected channel).
- Training, inference, and evaluation scripts are included.

### Installation

```bash
# Python 3.9+ recommended
python -m venv .venv && source .venv/bin/activate

# Install PyTorch (choose the right command for your CUDA version)
# See https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install denoising-diffusion-pytorch einops tqdm pillow pandas tensorboard scikit-image
```

### Dataset

See `DATASET.md` for details.

- CSV files: `datasets/Corrosion_train.csv` and `datasets/Corrosion_test.csv`
- Columns: `filename`, `S11`, `S21`, `Phase11`, `Phase21`
- Each measurement column is a single string with 201 space‑separated floats
- Ground‑truth images live under `datasets/corrosion_img/<SAMPLE_INDEX>/<filename>.png`
  - The `<SAMPLE_INDEX>` is the second token of `filename` split by `_`

Example filename → image path mapping:
- `0525_61_30.89263840450541_augmented` → `datasets/corrosion_img/61/0525_61_30.89263840450541_augmented.png`

### Quick Start

- Training: `train.py`
- Inference (image generation): `inference.py`
- Evaluation (MAE/MSE/PSNR/SSIM on red channel): `compare.py`

You can also see `run.sh` for example commands.

### Training

```bash
python train.py \
  --csv        ./datasets/Corrosion_train.csv \
  --val_csv    ./datasets/Corrosion_test.csv \
  --img_root   ./datasets/corrosion_img \
  --use_channels S11 S21 Phase11 \
  --image_size 64 \
  --timesteps 1000 \
  --sampling_timesteps 250 \
  --batch_size 16 \
  --lr 8e-5 \
  --num_steps 1000000 \
  --log_every 10000 \
  --save_every 100000 \
  --log_dir ./logs_ext
```

Key notes:
- Images are read as RGB but only the red channel is used and normalized to `[-1, 1]`.
- Conditioning dimension is `201 * len(--use_channels)` (e.g., `402` for `S11 S21`).
- Optional: `--use_mlp <hidden_dim>` enables a projection MLP applied to the conditioning vector.
- Checkpoints and TensorBoard logs are written under a timestamped subdirectory in `--log_dir` (default `./logs_ext/<YYYYMMDD-HHMMSS>`).

### Inference (Generate Images)

```bash
python inference.py \
  --checkpoint ./logs_ext/20250816-213838/model_step_400000.pt \
  --csv ./datasets/Corrosion_test.csv \
  --output ./output/S11_S21_400k \
  --img_root ./datasets/corrosion_img \
  --use_channels S11 S21 \
  --image_size 64 \
  --timesteps 1000 \
  --sampling_timesteps 250 \
  --batch_size 16
```

Behavior:
- Generates single‑channel images; saved as red‑only RGB PNGs.
- If the original target image exists, the generated image is resized back to the original resolution before saving.
- If you trained with `--use_mlp`, pass the same flag and value during inference.

### Evaluation

Compare generated images against ground truth (on the red channel):

```bash
python compare.py \
  --csv ./datasets/Corrosion_test.csv \
  --img_root ./datasets/corrosion_img \
  --gen_root ./output/S11_S21_400k \
  --out_csv ./output/S11_S21_400k_metrics.csv
```

- Outputs per‑image metrics CSV and prints averages for MAE, MSE, PSNR, SSIM.
- `scikit-image` is optional; if not installed, SSIM will be reported as NaN.

### Logging

- TensorBoard: `tensorboard --logdir ./logs_ext`
- Checkpoints: `./logs_ext/<run_id>/model_step_*.pt`

### Tips

- Ensure the `filename` convention and directory structure match `DATASET.md`.
- For multi‑GPU, consider using `torchrun` or `accelerate` to launch `train.py`.
- Adjust `--image_size` to balance speed and fidelity.

### Acknowledgements

- Built on top of `denoising_diffusion_pytorch` by Phil Wang.

### License

Retains the license obligations of upstream components. Ensure you keep attribution and comply with dataset licenses.


