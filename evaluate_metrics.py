"""
Evaluate Generated Images Quality (evaluate_metrics.py)
========================================================

This script compares generated images against their corresponding real (ground truth)
images and computes quality metrics for evaluation.

Metrics computed:
    - MAE (Mean Absolute Error): Average pixel-wise absolute difference
    - MSE (Mean Squared Error): Average pixel-wise squared difference  
    - PSNR (Peak Signal-to-Noise Ratio): Logarithmic quality measure in dB
    - SSIM (Structural Similarity Index): Perceptual quality metric
    - MACE (Mean Absolute Corrosion Error): Difference in corrosion percentage

Usage:
    python evaluate_metrics.py \\
        --csv datasets/Corrosion_test.csv \\
        --img_root ./datasets/corrosion_img \\
        --gen_root ./generated/S11 \\
        --out_csv metrics_S11.csv

Output:
    - Per-image metrics saved to CSV
    - Aggregate statistics printed to console

Note:
    - Only the RED channel is used for comparison (corrosion intensity)
    - Images are resized to match if dimensions differ
    - Requires scikit-image for SSIM computation

Author: Corrosion Diffusion Project
"""

import argparse
from pathlib import Path
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def load_red_channel(path: Path) -> np.ndarray:
    """Load image and return red channel as float32 array in [0,1]."""
    with Image.open(path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
    red = arr[..., 0].astype(np.float32) / 255.0
    return red


def resize_to(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    pil = Image.fromarray((img * 255.0).clip(0, 255).astype(np.uint8))
    pil = pil.resize((w, h), resample=Image.BILINEAR)
    out = np.asarray(pil, dtype=np.uint8).astype(np.float32) / 255.0
    return out


def compute_metrics(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Return (mae, mse, psnr, ssim_val, mace) for two float images in [0,1]."""
    diff = x - y
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    psnr = float(10.0 * math.log10(1.0 / mse)) if mse > 0 else float("inf")
    if _HAS_SKIMAGE:
        ssim_val = float(ssim(x, y, data_range=1.0))
    else:
        ssim_val = float("nan")
    
    # MACE: Mean Absolute Corrosion Error
    # Corrosion score is mean pixel value * 100 (0-100 scale)
    real_corrosion = float(x.mean()) * 100.0
    gen_corrosion = float(y.mean()) * 100.0
    mace = abs(real_corrosion - gen_corrosion)
    
    return mae, mse, psnr, ssim_val, mace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare generated images to originals")
    p.add_argument("--csv", type=str, required=True, help="CSV listing filenames and columns")
    p.add_argument("--img_root", type=str, default="./datasets/corrosion_img", help="Root dir of original images")
    p.add_argument("--gen_root", type=str, required=True, help="Root dir of generated images")
    p.add_argument("--out_csv", type=str, default="metrics.csv", help="Where to save per-image metrics")
    p.add_argument("--max_items", type=int, default=0, help="Optional cap on number of rows to evaluate (0 = all)")
    p.add_argument("--table", action="store_true", help="Print result as a single tab-separated line")
    p.add_argument("--sensor", type=str, default="", help="Sensor name for table output")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.csv)
    img_root = Path(args.img_root)
    gen_root = Path(args.gen_root)

    rows: List[dict] = []
    num_missing = 0

    iterable = df.itertuples(index=False)
    if args.max_items > 0:
        iterable = list(iterable)[: args.max_items]

    for row in iterable:
        filename = getattr(row, "filename")
        try:
            sample_index = filename.split("_")[1]
        except Exception:
            # skip malformed names
            continue

        orig_path = img_root / sample_index / f"{filename}.png"
        gen_path = gen_root / sample_index / f"{filename}.png"

        if not orig_path.exists() or not gen_path.exists():
            num_missing += 1
            continue

        orig = load_red_channel(orig_path)
        gen = load_red_channel(gen_path)

        if gen.shape != orig.shape:
            gen = resize_to(gen, orig.shape)

        mae, mse, psnr, ssim_val, mace = compute_metrics(orig, gen)

        rows.append(
            dict(
                filename=filename,
                sample_index=sample_index,
                mae=mae,
                mse=mse,
                psnr=psnr,
                ssim=ssim_val,
                mace=mace,
            )
        )

    if not rows:
        print("No comparable pairs found (missing files?).")
        return

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    avg_mae = out_df.mae.mean()
    avg_mse = out_df.mse.mean()
    avg_psnr = out_df.psnr[out_df.psnr != float("inf")].mean()
    avg_ssim = out_df.ssim.mean() if _HAS_SKIMAGE else float("nan")
    avg_mace = out_df.mace.mean()

    if args.table:
        # Format: Sensor MAE MSE PSNR SSIM MACE
        ssim_str = f"{avg_ssim:.4f}" if _HAS_SKIMAGE else "N/A"
        print(f"{args.sensor}\t{avg_mae:.6f}\t{avg_mse:.6f}\t{avg_psnr:.2f}\t{ssim_str}\t{avg_mace:.4f}")
    else:
        print(f"Saved per-image metrics to {out_path}")
        print(
            "Averages -> MAE: {:.6f}, MSE: {:.6f}, PSNR: {:.2f} dB, SSIM: {}, MACE: {:.4f}".format(
                avg_mae, avg_mse, avg_psnr,
                ("{:.4f}".format(avg_ssim) if _HAS_SKIMAGE else "skimage not installed"),
                avg_mace
            )
        )
        if num_missing:
            print(f"Warning: {num_missing} pairs skipped due to missing files.")


if __name__ == "__main__":
    main(parse_args())





