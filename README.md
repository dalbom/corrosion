## Sensor Image Reconstruction From Measurement Vectors

This repository trains and evaluates sensor-conditioned image reconstruction models. The active corrosion domain synthesizes red-channel corrosion images from RF measurement vectors such as `S11`, `S21`, `Phase11`, and `Phase21`; the package layout is domain-neutral so thermal S-parameter reconstruction can be added later through a domain adapter.

- Single‑channel target images are used (red channel of RGB).
- Conditioning is a concatenated float vector (length 201 per selected channel).
- Active methods: DDPM, DiT, and cGAN.
- New pipeline metrics are computed on raw generated images only. Histogram matching is archived under `scripts/legacy/` for historical comparisons.

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

- CSV split files: `datasets/corrosion/splits/train.csv`, `val.csv`, and `test.csv`
- Columns: `filename`, `S11`, `S21`, `Phase11`, `Phase21`
- Each measurement column is a single string with 201 space‑separated floats
- Ground‑truth images live under `datasets/corrosion/images/<SAMPLE_INDEX>/<filename>.png`
  - The `<SAMPLE_INDEX>` is the second token of `filename` split by `_`
- Raw S-parameter files, when needed for provenance, live under `datasets/corrosion/sparameters/<SAMPLE_INDEX>/`

Example filename → image path mapping:
- `0525_61_30.89263840450541_augmented` → `datasets/corrosion/images/61/0525_61_30.89263840450541_augmented.png`

### Quick Start

New config-driven entrypoints:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -m sensor_image_recon.cli train \
  --config configs/corrosion/cgan_s11_s21.yaml

python -m sensor_image_recon.cli infer \
  --run runs/corrosion/cgan/corrosion_default_verification/s11_s21/seed_001/<run_id>

python -m sensor_image_recon.cli evaluate \
  --run runs/corrosion/cgan/corrosion_default_verification/s11_s21/seed_001/<run_id>
```

Run outputs are immutable and self-contained:

```text
runs/{domain}/{method}/{study}/{sensor_set}/{seed}/{run_id}/
  config.yaml
  metadata.json
  checkpoints/
  tensorboard/
  samples/
  inference/
  metrics/
```

For overview across many sensor/method combinations, generate a symlink catalog:

```bash
python -m sensor_image_recon.cli catalog --runs-root runs
# or:
python scripts/generate_catalog.py --runs-root runs --domain corrosion --method cgan
```

The catalog keeps indexes and symlinks only; checkpoint and image files remain in their run folders.

```text
runs/catalog/{domain}/{method}/{study}/
  registered_runs.json
  leaderboard.csv
  checkpoints/{sensor_set}/{seed}/best_model.pt -> selected run
  inference/{sensor_set}/{seed} -> selected run inference directory
  samples/{sensor_set}/{seed} -> selected run samples directory
```

If multiple runs have the same config identity, the catalog registers the most recent run and lists older skipped runs in `registered_runs.json` and in the command output.

### Sweeps

Domain-wide inventory lives in `configs/corrosion/domain.yaml`. It defines datasets, supported sensor sets, and methods. The L1 + SSIM + LPIPS reconstruction loss with BatchNorm conditioning is the default for every active method. You can generate an interactive sweep config:

```bash
python scripts/generate_sweep_config.py \
  --domain-config configs/corrosion/domain.yaml
```

Or run a prepared sweep:

```bash
python -m sensor_image_recon.cli sweep \
  --config configs/sweeps/corrosion/cgan_all_sensors.yaml
```

Sweep selections support one method with all sensor sets, one sensor set with all methods, or explicit method/sensor subsets:

```yaml
selection:
  methods: [cgan, ddpm]
  sensor_sets: [s11, s11_s21]
  seeds: [1, 2, 3]
stages: [train, infer, evaluate, catalog]
```

Old one-off entrypoints and run scripts have been moved under `unused/`. They are retained for reference only; new experiments should use `python -m sensor_image_recon.cli` or `scripts/recon.sh`.

### Tips

- Ensure the `filename` convention and directory structure match `DATASET.md`.
- Adjust `--image_size` to balance speed and fidelity.

### Legacy Post Image Processing

Histogram matching and other correction utilities are retained only as legacy references. They are not part of the new raw-only evaluation process.

**Available Methods:**

| Method | Description | Best For |
|--------|-------------|----------|
| **Histogram Matching** | Matches the full histogram distribution of generated images to real images | General use, good baseline |
| **Offset Correction** | Adds/subtracts a constant value to shift mean intensity | Simple linear shift |
| **Scaling Correction** | Multiplies by a factor to match mean intensity | Proportional adjustment |
| **Linear Regression** | Applies a learned linear transformation (y = mx + b) | When relationship is linear |
| **Soft Histogram Matching** | Blends histogram matching with original (α=0.5) | Preserving some original characteristics |
| **Nonlinear Curve Fitting** | Gamma correction + polynomial LUT | Complex nonlinear relationships |
| **Optimal Transport (EMD)** | Minimizes Earth Mover's Distance between distributions | Theoretically optimal mapping |

**Quick Usage:**

```bash
# Apply correction to generated images for historical comparisons
python scripts/legacy/correct_generated_images.py

# Visualize before/after distributions
python scripts/legacy/create_histogram_figure.py
```

For detailed explanations of each method, see **[IMGPROC.md](IMGPROC.md)**.

### Acknowledgements

- Built on top of `denoising_diffusion_pytorch` by Phil Wang.

### License

Retains the license obligations of upstream components. Ensure you keep attribution and comply with dataset licenses.
