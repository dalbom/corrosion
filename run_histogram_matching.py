"""
Non-interactive histogram matching for generated corrosion images.
Extracted from corrosion/correct_generated_images.py for pipeline automation.

Usage:
    python run_histogram_matching.py \
        --gen_dir results/baseline/ddpm/trial1/S11 \
        --corrected_dir results/corrected/ddpm/trial1/S11 \
        --train_csv datasets/Corrosion_cGAN_train.csv \
        --img_root datasets/corrosion_img
"""
import argparse
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd


def get_r_channel(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    return img[:, :, 2]  # BGR format, R is index 2


def get_histogram_mapping(source_values, template_values):
    source_hist, _ = np.histogram(source_values.flatten(), bins=256, range=(0, 256))
    template_hist, _ = np.histogram(template_values.flatten(), bins=256, range=(0, 256))
    source_cdf = source_hist.cumsum() / source_hist.sum()
    template_cdf = template_hist.cumsum() / template_hist.sum()
    mapping = np.interp(source_cdf, template_cdf, np.arange(256))
    return mapping


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir", required=True, help="Dir with generated images (sensor subfolder)")
    p.add_argument("--corrected_dir", required=True, help="Output dir for corrected images")
    p.add_argument("--train_csv", default="datasets/Corrosion_cGAN_train.csv")
    p.add_argument("--img_root", default="datasets/corrosion_img")
    args = p.parse_args()

    # Load training reference distribution
    train_df = pd.read_csv(args.train_csv)
    train_r_values = []
    for fname in train_df["filename"].tolist():
        try:
            subdir = fname.split("_")[1]
            path = os.path.join(args.img_root, subdir, f"{fname}.png")
            r = get_r_channel(path)
            if r is not None:
                train_r_values.append(r.flatten())
        except (IndexError, FileNotFoundError):
            pass
    all_train_r = np.concatenate(train_r_values)

    # Find all generated images
    gen_paths = sorted(glob(os.path.join(args.gen_dir, "**", "*.png"), recursive=True))
    if not gen_paths:
        print(f"No images found in {args.gen_dir}")
        return

    # Collect generated R channel values for histogram mapping
    gen_r_values = []
    for p_path in gen_paths:
        r = get_r_channel(p_path)
        if r is not None:
            gen_r_values.append(r.flatten())
    all_gen_r = np.concatenate(gen_r_values)

    # Compute histogram mapping
    mapping = get_histogram_mapping(all_gen_r, all_train_r)

    # Apply mapping and save
    os.makedirs(args.corrected_dir, exist_ok=True)
    for gen_path in tqdm(gen_paths, desc="Correcting"):
        img = cv2.imread(gen_path)
        if img is None:
            continue
        b, g, r = cv2.split(img)

        # Apply histogram mapping
        bin_indices = np.clip(r.flatten().astype(int), 0, 255)
        corrected_r = mapping[bin_indices].reshape(r.shape).astype(np.uint8)
        corrected_img = cv2.merge((b, g, corrected_r))

        # Preserve relative path structure
        rel_path = os.path.relpath(gen_path, args.gen_dir)
        out_path = os.path.join(args.corrected_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, corrected_img)


if __name__ == "__main__":
    main()
