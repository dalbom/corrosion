from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from sensor_image_recon.core.identity import attach_config_identity


DEFAULTS: dict[str, Any] = {
    "domain": "corrosion",
    "experiment": "default",
    "run": {"root": "runs"},
    "dataset": {
        "image_size": 128,
        "channels": ["S11", "S21"],
        "num_workers": 4,
    },
    "training": {
        "seed": None,
        "epochs": 100,
        "num_steps": 1000,
        "batch_size": 32,
        "num_workers": 4,
        "max_batches_per_epoch": 0,
        "val_sample_size": 64,
        "lr": 1e-4,
        "lr_g": 1e-4,
        "lr_c": 1e-4,
        "n_critic": 2,
        "lambda_gp": 10.0,
        "save_every": 5,
    },
    "loss": {
        "lambda_l1": 100.0,
        "lambda_ssim": 50.0,
        "lambda_perceptual": 10.0,
        "denoising_weight": 1.0,
    },
    "conditioning": {
        "norm_type": "batchnorm",
    },
    "architecture": {
        "latent_dim": 128,
        "ngf": 128,
        "ndf": 128,
        "image_size": 128,
        "timesteps": 1000,
        "sampling_timesteps": 250,
        "objective": "pred_noise",
        "dim": 64,
        "dim_max": 256,
        "num_downsamples": 3,
        "num_blocks_per_stage": 2,
        "patch_size": 4,
        "hidden_size": 384,
        "depth": 12,
        "num_heads": 6,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    config = _deep_merge(DEFAULTS, loaded)
    config["_config_path"] = str(path)
    if config["training"].get("seed") is None:
        config["training"]["seed"] = config.get("seed", 0)
    config["seed"] = config["training"]["seed"]
    if "domain" in config and "method" in config:
        attach_config_identity(config)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(serializable, fh, sort_keys=False)
