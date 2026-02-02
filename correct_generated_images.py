"""
Color Correction for Generated Images (correct_generated_images.py)
=====================================================================

This script applies various color/intensity correction methods to generated
corrosion images to better match the distribution of real images.

Problem:
    Generated images from the diffusion model often have systematic offsets
    in their intensity distribution compared to real images. This script
    corrects these discrepancies using various statistical methods.

Available Correction Methods:
    1. Histogram Matching - Match the full histogram distribution
    2. Offset Correction - Add/subtract a constant value
    3. Scaling Correction - Multiply by a constant factor
    4. Linear Regression - Apply learned linear transformation (y = mx + b)
    5. Soft Histogram Matching - Blended histogram matching with original
    6. Nonlinear Curve Fitting - Gamma correction + polynomial LUT
    7. Optimal Transport (EMD) - Minimize Earth Mover's Distance

Usage:
    python correct_generated_images.py
    
    Then select a correction method from the interactive menu.

Input:
    - generated/: Directory containing generated images by sensor type
    - datasets/Corrosion_train.csv: Reference distribution from training data
    - datasets/test/: Real test images for error calculation

Output:
    - corrected/: Corrected images saved with same directory structure
    - Console output with before/after error statistics

Note:
    - Only the RED channel is processed (contains corrosion intensity)
    - Correction parameters are computed per generated set

Author: Corrosion Diffusion Project
"""

import os
from glob import glob

import cv2
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit
from scipy.stats import linregress
from tqdm import tqdm
import pandas as pd
import ot


def get_image_paths_from_csv(filenames, base_image_dir, desc):
    """
    Reads a list of filenames and constructs the full image paths.
    """
    image_paths = []
    for fname in tqdm(filenames, desc=desc):
        try:
            subdir = fname.split("_")[1]
            # Construct the full path and check if it exists
            path = os.path.join(base_image_dir, subdir, f"{fname}.png")
            if os.path.exists(path):
                image_paths.append(path)
        except IndexError:
            print(
                f"Warning: Could not parse filename '{fname}' to determine subdirectory."
            )
    return image_paths


def get_r_channel(image_path):
    """Loads an image and returns its R channel."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return img[:, :, 2]


def get_histogram_mapping(source_values, template_values):
    """Computes a mapping to transform the source distribution to the template distribution."""
    source_hist, _ = np.histogram(source_values.flatten(), bins=256, range=(0, 256))
    template_hist, _ = np.histogram(template_values.flatten(), bins=256, range=(0, 256))

    source_cdf = source_hist.cumsum() / source_hist.sum()
    template_cdf = template_hist.cumsum() / template_hist.sum()

    mapping = np.interp(source_cdf, template_cdf, np.arange(256))
    return mapping





def build_set_lut(source_pixels: np.ndarray, target_pixels: np.ndarray) -> np.ndarray:
    src_hist, _ = np.histogram(source_pixels, bins=256, range=(0, 255), density=True)
    tgt_hist, _ = np.histogram(target_pixels, bins=256, range=(0, 255), density=True)
    src_cdf = np.cumsum(src_hist)
    src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1
    tgt_cdf = np.cumsum(tgt_hist)
    tgt_cdf /= tgt_cdf[-1] if tgt_cdf[-1] > 0 else 1
    lut = (
        np.interp(src_cdf, tgt_cdf, np.arange(256))
        .clip(0, 255)
        .round()
        .astype(np.uint8)
    )
    return lut


def gamma_func(x, gamma, scale):
    return np.clip(scale * ((x / 255.0) ** gamma) * 255.0, 0, 255)


def fit_gamma_curve(source_pixels: np.ndarray, target_pixels: np.ndarray):
    # source_pixels, target_pixels: 1D uint8 arrays
    src = source_pixels.astype(np.float32)
    tgt = target_pixels.astype(np.float32)
    popt, _ = curve_fit(gamma_func, src, tgt, p0=[1.0, 1.0], maxfev=2000)
    gamma, scale = popt
    return gamma, scale


def apply_gamma_curve(image: np.ndarray, gamma: float, scale: float):
    corrected = gamma_func(image.astype(np.float32), gamma, scale)
    return corrected.astype(np.uint8)


def fit_polynomial_lut(
    source_pixels: np.ndarray, target_pixels: np.ndarray, degree: int = 3
):
    src_hist, _ = np.histogram(source_pixels, bins=256, range=(0, 255), density=True)
    tgt_hist, _ = np.histogram(target_pixels, bins=256, range=(0, 255), density=True)
    src_cdf = np.cumsum(src_hist)
    src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1
    tgt_cdf = np.cumsum(tgt_hist)
    tgt_cdf /= tgt_cdf[-1] if tgt_cdf[-1] > 0 else 1
    # Fit polynomial mapping from src_cdf to tgt_cdf
    # Note: this is an unconventional way to use polyfit on CDFs for a LUT
    # A direct mapping (like in histogram matching) is usually better.
    # This implementation follows the user's provided logic.
    x_vals = np.arange(256)
    mapping = np.interp(src_cdf, tgt_cdf, x_vals)
    coefs = poly.Polynomial.fit(x_vals, mapping, degree).convert().coef
    lut = np.clip(poly.polyval(x_vals, coefs), 0, 255).round().astype(np.uint8)
    return lut


def get_optimal_transport_lut(source_pixels: np.ndarray, target_pixels: np.ndarray) -> np.ndarray:
    """
    Computes an optimal transport-based LUT to map source distribution to target distribution.
    Uses the Earth Mover's Distance (Wasserstein distance) to find the optimal mapping.
    """
    # Compute histograms (as probability distributions)
    source_hist, _ = np.histogram(source_pixels, bins=256, range=(0, 256), density=True)
    target_hist, _ = np.histogram(target_pixels, bins=256, range=(0, 256), density=True)
    
    # Normalize to ensure they sum to 1 (probability masses)
    source_hist = source_hist / source_hist.sum()
    target_hist = target_hist / target_hist.sum()
    
    # Create the cost matrix (squared Euclidean distance between bin centers)
    bin_centers = np.arange(256)
    M = ot.dist(bin_centers.reshape(-1, 1), bin_centers.reshape(-1, 1), metric='sqeuclidean')
    M = M / M.max()  # Normalize cost matrix
    
    # Compute optimal transport plan using EMD (Earth Mover's Distance)
    # This gives us the optimal coupling between source and target distributions
    transport_plan = ot.emd(source_hist, target_hist, M)
    
    # Build LUT from transport plan
    # For each source bin, find the weighted average target bin
    lut = np.zeros(256, dtype=np.float64)
    for i in range(256):
        if transport_plan[i].sum() > 0:
            # Weighted average of target bins based on transport plan
            lut[i] = np.average(bin_centers, weights=transport_plan[i])
        else:
            lut[i] = i  # No transport, keep original
    
    return lut.clip(0, 255).round().astype(np.uint8)


def apply_correction_and_save(
    gen_path, real_corrosion_map, method, params, source_base, target_base
):
    """Applies a correction to a generated image and saves it."""
    img = cv2.imread(gen_path)
    if img is None:
        return None, None, None

    b, g, r = cv2.split(img)
    original_corrosion = r.mean() / 255.0 * 100

    if method == "histogram":
        mapping = params["mapping"]
        bin_indices = np.digitize(r.flatten(), np.arange(256)) - 1
        bin_indices[bin_indices < 0] = 0
        bin_indices[bin_indices >= len(mapping)] = len(mapping) - 1
        corrected_r_flat = mapping[bin_indices]
        corrected_r = corrected_r_flat.reshape(r.shape).astype(np.uint8)
    elif method == "offset":
        corrected_r = np.clip(r + params["offset"], 0, 255).astype(np.uint8)
    elif method == "scaling":
        corrected_r = np.clip(r * params["scale"], 0, 255).astype(np.uint8)
    elif method == "linear_regression":
        corrected_r = np.clip(r * params["slope"] + params["intercept"], 0, 255).astype(
            np.uint8
        )
    elif method == "soft_hist":
        lut = params["lut"]
        alpha = params.get("alpha", 0.5)
        matched_r = lut[r.astype(np.uint8)]
        blended = alpha * matched_r.astype(np.float32) + (1 - alpha) * r.astype(
            np.float32
        )
        corrected_r = np.clip(blended, 0, 255).astype(np.uint8)
    elif method == "nonlinear":
        # Apply gamma curve first
        gamma = params["gamma"]
        scale = params["scale"]
        corrected_r = apply_gamma_curve(r, gamma, scale)
        # Optionally refine with polynomial LUT
        lut = params["poly_lut"]
        corrected_r = lut[corrected_r]
    elif method == "optimal_transport":
        lut = params["ot_lut"]
        corrected_r = lut[r.astype(np.uint8)]

    corrected_corrosion = corrected_r.mean() / 255.0 * 100
    corrected_img = cv2.merge((b, g, corrected_r))

    # Fix Save Path: use relpath from the directory of the set, not the whole base dir
    relative_path = os.path.relpath(
        gen_path, os.path.join(source_base, params["gen_dir"])
    )
    target_path = os.path.join(target_base, params["gen_dir"], relative_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    cv2.imwrite(target_path, corrected_img)

    real_corrosion = real_corrosion_map.get(os.path.basename(gen_path))
    original_error = (
        abs(real_corrosion - original_corrosion)
        if real_corrosion is not None
        else None
    )
    corrected_error = (
        abs(real_corrosion - corrected_corrosion)
        if real_corrosion is not None
        else None
    )

    return original_corrosion, corrected_corrosion, original_error, corrected_error


def main():
    # --- 1. User Input ---
    print("Please select a correction method:")
    print("1. Histogram Matching")
    print("2. Offset Correction")
    print("3. Scaling Correction")
    print("4. Linear Regression")
    print("5. Soft Histogram Matching (interpolated)")
    print("6. Nonlinear Curve Fitting (Gamma/Polynomial LUT)")
    print("7. Optimal Transport (Earth Mover's Distance)")
    choice = input("Enter the number of your choice: ")

    method_map = {
        "1": "histogram",
        "2": "offset",
        "3": "scaling",
        "4": "linear_regression",
        "5": "soft_hist",
        "6": "nonlinear",
        "7": "optimal_transport",
    }
    method = method_map.get(choice)
    if not method:
        print("Invalid choice. Exiting.")
        return

    # --- 2. Setup Paths and Data ---
    #train_csv_path = "datasets/Corrosion_train.csv"
    train_csv_path = "datasets/Corrosion_cGAN_train.csv"
    test_dir = "datasets/test"
    base_image_dir = "datasets/corrosion_img"
    generated_images_base_dir = "generated_cGAN"
    corrected_images_base_dir = "corrected_cGAN"

    # --- Load Training Data for Correction Reference ---
    print("Loading training data for correction reference...")
    train_df = pd.read_csv(train_csv_path)
    train_image_paths = get_image_paths_from_csv(
        train_df["filename"].tolist(), base_image_dir, "Loading train images"
    )
    train_r_channels = {
        os.path.basename(p): get_r_channel(p) for p in train_image_paths
    }
    train_r_channels = {k: v for k, v in train_r_channels.items() if v is not None}
    all_train_r_values = np.concatenate(
        [v.flatten() for v in train_r_channels.values()]
    )

    # --- Load Test Data for Final Error Calculation ---
    print("Loading test data for error calculation...")
    test_image_paths = glob(os.path.join(test_dir, "*.png"))
    test_corrosion_map = {
        os.path.basename(p): get_r_channel(p).mean() / 255.0 * 100
        for p in test_image_paths
        if get_r_channel(p) is not None
    }

    # Calculate and print the mean corrosion for the real test set as a reference
    mean_real_corrosion_normalized = np.mean(list(test_corrosion_map.values()))
    print(f"\nMean Corrosion of Real Test Set: {mean_real_corrosion_normalized:.2f}\n")

    generated_dirs = sorted(
        [
            d
            for d in os.listdir(generated_images_base_dir)
            if os.path.isdir(os.path.join(generated_images_base_dir, d))
        ]
    )

    total_original_errors = []
    total_corrected_errors = []

    # --- 3. Process each generated set ---
    for gen_dir in tqdm(generated_dirs, desc="Processing sets"):
        gen_image_paths = glob(
            os.path.join(generated_images_base_dir, gen_dir, "**", "*.png"),
            recursive=True,
        )
        if not gen_image_paths:
            continue

        # --- 4. Calculate correction parameters based on aligned data ---
        params = {"gen_dir": gen_dir}
        # Align generated images with the test set to know which ones to process
        gen_r_channels = {
            os.path.basename(p): get_r_channel(p) for p in gen_image_paths
        }
        gen_r_channels = {k: v for k, v in gen_r_channels.items() if v is not None}
        common_keys = test_corrosion_map.keys() & gen_r_channels.keys()

        aligned_gen_paths = [
            p for p in gen_image_paths if os.path.basename(p) in common_keys
        ]

        if not aligned_gen_paths:
            continue

        aligned_gen_pixels = np.concatenate(
            [gen_r_channels[k].flatten() for k in common_keys]
        )

        if method == "soft_hist":
            params["lut"] = get_histogram_mapping(
                aligned_gen_pixels, all_train_r_values
            )
            params["alpha"] = 0.5  # default, make it configurable later if needed
        elif method == "nonlinear":
            # For curve fitting, we need aligned pixel pairs between train and gen
            min_len = min(len(aligned_gen_pixels), len(all_train_r_values))
            gen_sample = np.random.choice(aligned_gen_pixels, min_len, replace=False)
            train_sample = np.random.choice(all_train_r_values, min_len, replace=False)
            gamma, scale = fit_gamma_curve(gen_sample, train_sample)
            params["gamma"] = gamma
            params["scale"] = scale
            params["poly_lut"] = fit_polynomial_lut(gen_sample, train_sample)
        elif method == "histogram":
            params["mapping"] = get_histogram_mapping(
                aligned_gen_pixels, all_train_r_values
            )
        elif method == "optimal_transport":
            print("  Computing Optimal Transport LUT (this may take a moment)...")
            params["ot_lut"] = get_optimal_transport_lut(
                aligned_gen_pixels, all_train_r_values
            )
        elif method == "linear_regression":
            min_len = min(len(aligned_gen_pixels), len(all_train_r_values))
            gen_sample = np.random.choice(aligned_gen_pixels, min_len, replace=False)
            train_sample = np.random.choice(all_train_r_values, min_len, replace=False)
            res = linregress(x=gen_sample, y=train_sample)
            params["slope"] = res.slope
            params["intercept"] = res.intercept
        else:  # Offset and Scaling
            mean_gen_corrosion = aligned_gen_pixels.mean()
            mean_train_corrosion = all_train_r_values.mean()
            if method == "offset":
                params["offset"] = mean_train_corrosion - mean_gen_corrosion
            elif method == "scaling":
                params["scale"] = mean_train_corrosion / mean_gen_corrosion

        # --- 5. Apply correction and calculate metrics ---
        results = [
            apply_correction_and_save(
                p,
                test_corrosion_map,
                method,
                params,
                generated_images_base_dir,
                corrected_images_base_dir,
            )
            for p in aligned_gen_paths
        ]

        original_corrosion, corrected_corrosion, original_errors, corrected_errors = zip(*results)
        valid_original_errors = [e for e in original_errors if e is not None]
        valid_corrected_errors = [e for e in corrected_errors if e is not None]
        total_original_errors.extend(valid_original_errors)
        total_corrected_errors.extend(valid_corrected_errors)

        print(f"\n--- Results for {gen_dir} (Method: {method}) ---")
        print(f"  - Mean Original Corrosion: {np.mean(original_corrosion):.2f}")
        print(f"  - Mean Corrected Corrosion: {np.mean(corrected_corrosion):.2f}")
        if valid_original_errors:
            print(f"  - Mean Original Corrosion Error: {np.mean(valid_original_errors):.2f}")
        if valid_corrected_errors:
            print(f"  - Mean Corrected Corrosion Error: {np.mean(valid_corrected_errors):.2f}")

    print("\n--- Overall Results ---")
    if total_original_errors:
        print(f"  - Overall Mean Original Corrosion Error: {np.mean(total_original_errors):.2f}")
    if total_corrected_errors:
        print(f"  - Overall Mean Corrected Corrosion Error: {np.mean(total_corrected_errors):.2f}")


if __name__ == "__main__":
    main()
