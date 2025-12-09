"""
Create Histogram Figure for Correction Methods (create_histogram_figure.py)
=============================================================================

This script visualizes the effect of various color correction methods on the
R channel distribution of generated images. It creates histogram/KDE plots
comparing:
    - Real image distribution (ground truth)
    - Original generated image distribution
    - Corrected generated image distribution

Available Correction Methods (same as correct_generated_images.py):
    1. Histogram Matching - Match the full histogram distribution
    2. Offset Correction - Add/subtract a constant value
    3. Scaling Correction - Multiply by a constant factor
    4. Linear Regression - Apply learned linear transformation (y = mx + b)
    5. Soft Histogram Matching - Blended histogram matching with original
    6. Nonlinear Curve Fitting - Gamma correction + polynomial LUT
    7. Optimal Transport (EMD) - Minimize Earth Mover's Distance

Usage:
    python create_histogram_figure.py
    
    Then select a correction method from the interactive menu.

Output:
    - r_value_kde_<method>.png: KDE plots showing before/after distributions

Author: Corrosion Diffusion Project
"""

import os
from glob import glob
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import correction functions from correct_generated_images.py
from correct_generated_images import (
    get_histogram_mapping,
    get_optimal_transport_lut,
    fit_gamma_curve,
    fit_polynomial_lut,
    apply_gamma_curve,
)
from scipy.stats import linregress


def get_r_channel_values(image_paths, desc="Loading images"):
    """
    Extract R channel values from a list of image paths.
    Images are loaded, and the R channel is extracted.
    """
    r_values = []
    for img_path in tqdm(image_paths, desc=desc, leave=False):
        img = cv2.imread(img_path)
        if img is not None:
            if len(img.shape) == 3:
                # Use the red channel for 3-channel images (OpenCV is BGR)
                r_channel = img[:, :, 2]
            else:
                # Assume grayscale for single-channel images
                r_channel = img
            r_values.append(r_channel)

    if not r_values:
        return np.array([])

    return np.concatenate([r.flatten() for r in r_values])


def apply_correction_to_values(r_values, method, params):
    """
    Apply a correction method to R channel values (in-memory, no saving).
    Returns the corrected values.
    """
    r = r_values.astype(np.float32)
    
    if method == "histogram":
        mapping = params["mapping"]
        bin_indices = np.digitize(r_values, np.arange(256)) - 1
        bin_indices[bin_indices < 0] = 0
        bin_indices[bin_indices >= len(mapping)] = len(mapping) - 1
        corrected = mapping[bin_indices]
    elif method == "offset":
        corrected = np.clip(r + params["offset"], 0, 255)
    elif method == "scaling":
        corrected = np.clip(r * params["scale"], 0, 255)
    elif method == "linear_regression":
        corrected = np.clip(r * params["slope"] + params["intercept"], 0, 255)
    elif method == "soft_hist":
        lut = params["lut"]
        alpha = params.get("alpha", 0.5)
        bin_indices = np.digitize(r_values, np.arange(256)) - 1
        bin_indices[bin_indices < 0] = 0
        bin_indices[bin_indices >= len(lut)] = len(lut) - 1
        matched = lut[bin_indices]
        corrected = alpha * matched + (1 - alpha) * r
        corrected = np.clip(corrected, 0, 255)
    elif method == "nonlinear":
        gamma = params["gamma"]
        scale = params["scale"]
        lut = params["poly_lut"]
        # Apply gamma correction
        corrected = np.clip(scale * ((r / 255.0) ** gamma) * 255.0, 0, 255)
        # Apply polynomial LUT
        corrected_int = corrected.astype(np.uint8)
        corrected = lut[corrected_int].astype(np.float32)
    elif method == "optimal_transport":
        lut = params["ot_lut"]
        bin_indices = r_values.astype(np.uint8)
        corrected = lut[bin_indices].astype(np.float32)
    else:
        corrected = r  # No correction
        
    return corrected.astype(np.uint8)


def compute_correction_params(gen_pixels, real_pixels, method):
    """
    Compute correction parameters based on the selected method.
    """
    params = {}
    
    if method == "histogram":
        params["mapping"] = get_histogram_mapping(gen_pixels, real_pixels)
    elif method == "offset":
        params["offset"] = real_pixels.mean() - gen_pixels.mean()
    elif method == "scaling":
        params["scale"] = real_pixels.mean() / gen_pixels.mean()
    elif method == "linear_regression":
        min_len = min(len(gen_pixels), len(real_pixels))
        gen_sample = np.random.choice(gen_pixels, min_len, replace=False)
        real_sample = np.random.choice(real_pixels, min_len, replace=False)
        res = linregress(x=gen_sample, y=real_sample)
        params["slope"] = res.slope
        params["intercept"] = res.intercept
    elif method == "soft_hist":
        params["lut"] = get_histogram_mapping(gen_pixels, real_pixels)
        params["alpha"] = 0.5
    elif method == "nonlinear":
        min_len = min(len(gen_pixels), len(real_pixels))
        gen_sample = np.random.choice(gen_pixels, min_len, replace=False)
        real_sample = np.random.choice(real_pixels, min_len, replace=False)
        gamma, scale = fit_gamma_curve(gen_sample, real_sample)
        params["gamma"] = gamma
        params["scale"] = scale
        params["poly_lut"] = fit_polynomial_lut(gen_sample, real_sample)
    elif method == "optimal_transport":
        print("  Computing Optimal Transport LUT (this may take a moment)...")
        params["ot_lut"] = get_optimal_transport_lut(gen_pixels, real_pixels)
    
    return params


def get_method_display_name(method):
    """Get human-readable name for the method."""
    names = {
        "histogram": "Histogram Matching",
        "offset": "Offset Correction",
        "scaling": "Scaling Correction",
        "linear_regression": "Linear Regression",
        "soft_hist": "Soft Histogram Matching",
        "nonlinear": "Nonlinear Curve Fitting",
        "optimal_transport": "Optimal Transport (EMD)",
    }
    return names.get(method, method)


def main():
    # --- 1. User Input (same as correct_generated_images.py) ---
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

    method_display = get_method_display_name(method)
    print(f"\nSelected method: {method_display}\n")

    # --- 2. Setup paths ---
    real_images_dir = "datasets/test"
    generated_images_base_dir = "generated"

    # --- 3. Load real images for reference ---
    print("Loading real images for reference...")
    real_image_paths = glob(os.path.join(real_images_dir, "*.png"))
    real_r_values = get_r_channel_values(real_image_paths, desc="Loading real images")
    
    if len(real_r_values) == 0:
        print("Error: No real images found. Run prepare_test_dataset.py first.")
        return

    # --- 4. Process each generated set ---
    generated_dirs = sorted(
        [
            d
            for d in os.listdir(generated_images_base_dir)
            if os.path.isdir(os.path.join(generated_images_base_dir, d))
        ]
    )

    if not generated_dirs:
        print("Error: No generated image directories found.")
        return

    results = {}
    for gen_dir in tqdm(generated_dirs, desc="Processing generated sets"):
        gen_image_paths = glob(
            os.path.join(generated_images_base_dir, gen_dir, "**", "*.png"),
            recursive=True,
        )
        if gen_image_paths:
            original_r = get_r_channel_values(
                gen_image_paths, desc=f"Loading {gen_dir}"
            )

            # Compute correction parameters for this set
            params = compute_correction_params(original_r, real_r_values, method)

            # Apply correction to get corrected R values
            corrected_r = apply_correction_to_values(original_r, method, params)

            results[gen_dir] = {"original": original_r, "corrected": corrected_r}

    # --- 5. Print all mean values ---
    print("\n--- Mean R Values ---")
    print(f"Mean R (Real): {np.mean(real_r_values):.2f}")
    for gen_dir, data in results.items():
        mean_orig = np.mean(data["original"])
        mean_corr = np.mean(data["corrected"])
        print(
            f"{gen_dir}: Original Mean R = {mean_orig:.2f}, "
            f"Corrected Mean R = {mean_corr:.2f}"
        )

    # --- 6. Generate and save histogram plots ---
    print("\nGenerating histograms...")
    num_sets = len(results)
    
    # Fixed 3x5 grid layout to match the old figure
    rows = 3
    cols = 5
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (gen_dir, data) in enumerate(results.items()):
        if i >= len(axes):
            break
        ax = axes[i]

        # Plot distributions using Kernel Density Estimates for a smooth curve
        sns.kdeplot(real_r_values, ax=ax, color="red", label="Real", fill=True, alpha=0.3)
        sns.kdeplot(data["original"], ax=ax, color="blue", label="Original", linestyle="--")
        sns.kdeplot(data["corrected"], ax=ax, color="green", label="Corrected", linewidth=2)

        ax.set_title(gen_dir, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_yticklabels([])  # Hide y-axis labels for clarity

    # Hide any unused subplots
    for i in range(num_sets, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    
    # Save with method name in filename
    save_path = f"r_value_kde_{method}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nHistogram figure saved to: {save_path}")


if __name__ == "__main__":
    main()
