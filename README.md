## Sensor Image Reconstruction From Measurement Vectors

This repository trains and evaluates sensor-conditioned image reconstruction models. The active corrosion domain synthesizes red-channel corrosion images from RF measurement vectors such as `S11`, `S21`, `Phase11`, and `Phase21`; the package layout is domain-neutral so thermal S-parameter reconstruction can be added later through a domain adapter.

- Single‑channel target images are used (red channel of RGB).
- Conditioning is a concatenated float vector (length 201 per selected channel).
- Active methods: DDPM, DiT, and cGAN.
- New pipeline metrics are computed on raw generated images only. Histogram matching is archived under `scripts/legacy/` for historical comparisons.

### Installation

Choose one environment manager and expand its setup commands.

<details>
<summary>pip + venv</summary>

```bash
# Python 3.9+ recommended
python -m venv .venv && source .venv/bin/activate

# Install PyTorch (choose the right command for your CUDA version)
# See https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install -r requirements.txt
```

</details>

<details>
<summary>conda</summary>

```bash
conda create -n sensor-recon python=3.10 -y
conda activate sensor-recon

# Install PyTorch (choose the right command for your CUDA version)
# See https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install -r requirements.txt
```

</details>

<details>
<summary>micromamba</summary>

```bash
micromamba create -n py310 python=3.10 -y
micromamba activate py310

# Install PyTorch (choose the right command for your CUDA version)
# See https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install -r requirements.txt
```

</details>

### Dataset

Expected active layout:

- CSV split files: `datasets/corrosion/splits/train.csv`, `val.csv`, and `test.csv`
- Columns: `filename`, `S11`, `S21`, `Phase11`, `Phase21`
- Each measurement column is a single string with 201 space‑separated floats
- Ground‑truth images live under `datasets/corrosion/images/<SAMPLE_INDEX>/<filename>.png`
  - The `<SAMPLE_INDEX>` is the second token of `filename` split by `_`
- Raw S-parameter files, when needed for provenance, live under `datasets/corrosion/sparameters/<SAMPLE_INDEX>/`

Example filename → image path mapping:
- `0525_61_30.89263840450541_augmented` → `datasets/corrosion/images/61/0525_61_30.89263840450541_augmented.png`

### Quick Start

For normal use, create a run plan through the interview script and execute it:

```bash
python scripts/generate_sweep_config.py \
  --domain-config configs/corrosion/domain.yaml

python -m sensor_image_recon.cli sweep \
  --config configs/sweeps/corrosion/<study_name>.yaml
```

The interview asks for the study name, methods, sensor combinations, seeds, and stages. It writes a YAML file so the same run can be repeated later.

<details>
<summary>Manual train, infer, and evaluate commands</summary>

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -m sensor_image_recon.cli train \
  --config configs/corrosion/cgan_s11_s21.yaml

python -m sensor_image_recon.cli infer \
  --run runs/corrosion/cgan/corrosion_default_verification/s11_s21/seed_001/<run_id>

python -m sensor_image_recon.cli evaluate \
  --run runs/corrosion/cgan/corrosion_default_verification/s11_s21/seed_001/<run_id>
```

</details>

<details>
<summary>Run directory layout</summary>

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

</details>

<details>
<summary>Catalog overview for checkpoints and generated images</summary>

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

</details>

### Sweeps

Generate a sweep config through the interview script:

```bash
python scripts/generate_sweep_config.py \
  --domain-config configs/corrosion/domain.yaml
```

Then run the generated file:

```bash
python -m sensor_image_recon.cli sweep \
  --config configs/sweeps/corrosion/<study_name>.yaml
```

<details>
<summary>What the domain inventory defines</summary>

Domain-wide inventory lives in `configs/corrosion/domain.yaml`. It defines datasets, supported sensor sets, and methods. The L1 + SSIM + LPIPS reconstruction loss with BatchNorm conditioning is the default for every active method.

</details>

<details>
<summary>Run a prepared sweep config</summary>

Or run a prepared sweep:

```bash
python -m sensor_image_recon.cli sweep \
  --config configs/sweeps/corrosion/cgan_all_sensors.yaml
```

</details>

<details>
<summary>Edit sweep selections manually</summary>

Sweep selections support one method with all sensor sets, one sensor set with all methods, or explicit method/sensor subsets:

```yaml
selection:
  methods: [cgan, ddpm]
  sensor_sets: [s11, s11_s21]
  seeds: [1, 2, 3]
stages: [train, infer, evaluate, catalog]
```

</details>

Old one-off entrypoints and run scripts were removed from the active repository. New experiments should use `python -m sensor_image_recon.cli` or `scripts/recon.sh`.

### Tips

- Ensure the `filename` convention and directory structure match the Dataset section above.
- Adjust `--image_size` to balance speed and fidelity.

### Acknowledgements

- Built on top of `denoising_diffusion_pytorch` by Phil Wang.

### License

Retains the license obligations of upstream components. Ensure you keep attribution and comply with dataset licenses.
