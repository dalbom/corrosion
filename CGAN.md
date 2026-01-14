# cGAN Baseline for Corrosion Image Generation

A conditional GAN (cGAN) implementation for generating synthetic corrosion images conditioned on sensor measurements.

## Overview

This cGAN baseline uses:
- **Generator**: Transposed convolutions with batch normalization
- **Discriminator**: Spectral normalization with projection-based conditioning
- **Training**: 128×128 images, resized to 300×110 for output
- **Conditioning**: 4 sensor types (S11, S21, Phase11, Phase21), 201 values each

## Files

| File | Description |
|------|-------------|
| `split_cgan_dataset.py` | Splits training CSV into 80% train / 20% validation |
| `cgan/models.py` | Generator & Discriminator architectures |
| `cgan/dataset.py` | Dataset class with configurable sensor channels |
| `train_cgan.py` | Training script with TensorBoard and early stopping |
| `inference_cgan.py` | Generate images for test set |
| `run_cgan_train.sh` | Train all 15 sensor combinations |

## Data Preparation

```bash
micromamba run -n py310 python split_cgan_dataset.py
```

Creates:
- `datasets/Corrosion_cGAN_train.csv` (3414 samples, 80%)
- `datasets/Corrosion_cGAN_validation.csv` (854 samples, 20%)

## Training

### Single Model
```bash
micromamba run -n py310 python train_cgan.py \
  --csv datasets/Corrosion_cGAN_train.csv \
  --val_csv datasets/Corrosion_cGAN_validation.csv \
  --use_channels S11 S21 \
  --epochs 300 --patience 10
```

### All 15 Combinations
```bash
./run_cgan_train.sh
```

### Key Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--use_channels` | S11 | Sensor channels for conditioning |
| `--epochs` | 300 | Maximum training epochs |
| `--patience` | 10 | Early stopping patience |
| `--batch_size` | 32 | Batch size |
| `--lr_g`, `--lr_d` | 2e-4 | Learning rates |
| `--image_size` | 128 | Training image size (square) |

### Checkpoints
Saved to `logs_ext/cGAN/YYYYMMDD-HHMMSS_<channels>/`:
- `best_model.pt` - Best model checkpoint
- `checkpoint_epoch_XXXX.pt` - Periodic checkpoints
- `tensorboard/` - TensorBoard logs

## Inference

```bash
micromamba run -n py310 python inference_cgan.py \
  --checkpoint logs_ext/cGAN/<experiment>/best_model.pt \
  --test_csv datasets/Corrosion_test.csv \
  --output_dir generated_cGAN/<experiment>
```

Output images are 300×110 RGB PNG (red channel only, matching original format).

## Sensor Combinations (15 total)

| Sensors | Conditioning Dim |
|---------|------------------|
| 1 sensor (4 models) | 201 |
| 2 sensors (6 models) | 402 |
| 3 sensors (4 models) | 603 |
| 4 sensors (1 model) | 804 |

## TensorBoard

```bash
tensorboard --logdir logs_ext/cGAN
```

Logs include:
- Generator/Discriminator losses (train + val)
- Sample images from train and validation sets (resized to 300×110)
