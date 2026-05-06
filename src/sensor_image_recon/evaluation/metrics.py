from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from sensor_image_recon.domains.corrosion.metrics import mean_absolute_corrosion_error


def load_red01(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.uint8)
    return arr[..., 0].astype(np.float32) / 255.0


def resize_to(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    pil = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    pil = pil.resize((w, h), resample=Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8).astype(np.float32) / 255.0


def image_metrics(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    if generated.shape != real.shape:
        generated = resize_to(generated, real.shape)
    diff = real - generated
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    psnr = float(10.0 * math.log10(1.0 / mse)) if mse > 0 else float("inf")
    try:
        from skimage.metrics import structural_similarity

        ssim = float(structural_similarity(real, generated, data_range=1.0))
    except Exception:
        ssim = float("nan")
    return {
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "mace": mean_absolute_corrosion_error(real, generated),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    return {
        "mae": float(frame["mae"].mean()),
        "mse": float(frame["mse"].mean()),
        "psnr": float(frame["psnr"].replace([float("inf")], np.nan).mean()),
        "ssim": float(frame["ssim"].mean()),
        "mace": float(frame["mace"].mean()),
    }
