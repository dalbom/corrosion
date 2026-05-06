"""
Create Comparison Figure for Manuscript (create_comparison_figure.py)
======================================================================

This script creates a 3×5 grid figure comparing:
    - Row 1: Real (ground truth) corrosion images
    - Row 2: Generated images from the diffusion model
    - Row 3: Color-corrected generated images

The figure is designed for inclusion in research papers/manuscripts to
visually demonstrate the quality of the generation and correction pipeline.

Usage:
    python create_comparison_figure.py

Configuration (hardcoded in script):
    - sensor_type: Which sensor configuration to visualize (e.g., "S11_Ph21_1M")
    - num_images: Number of columns (random samples) to display
    - output_filename: Where to save the resulting figure

Output:
    - PNG figure saved to current directory
    - Randomly selected samples for fair representation

Note:
    - Requires generated and corrected images to exist
    - Images are displayed without axes for clean presentation

Author: Corrosion Diffusion Project
"""

import os
import random
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def main():
    """
    Creates a 3x5 figure comparing real, generated (S11), and corrected (S11) images.
    """

    sensor_type = "S11_Ph21_1M"

    real_images_dir = "datasets/test"
    generated_s11_dir = f"generated/{sensor_type}"
    corrected_s11_dir = f"corrected/{sensor_type}"
    output_filename = f"manuscript_figure_{sensor_type}.png"
    num_images = 5

    # 1. Get all real test image paths
    real_image_paths = glob(os.path.join(real_images_dir, "*.png"))
    if not real_image_paths:
        print(f"Error: No real test images found in {real_images_dir}")
        return

    # 2. Randomly select 5 images
    if len(real_image_paths) < num_images:
        print(
            f"Warning: Fewer than {num_images} real images found. Using {len(real_image_paths)} images."
        )
        num_images = len(real_image_paths)

    selected_real_paths = random.sample(real_image_paths, num_images)

    # 3. Find corresponding generated and corrected images
    image_sets = []
    for real_path in selected_real_paths:
        basename = os.path.basename(real_path)

        # Search for the corresponding file in generated and corrected subdirectories
        gen_path_list = glob(
            os.path.join(generated_s11_dir, "**", basename), recursive=True
        )
        corr_path_list = glob(
            os.path.join(corrected_s11_dir, "**", basename), recursive=True
        )

        if gen_path_list and corr_path_list:
            image_sets.append(
                {
                    "real": real_path,
                    "generated": gen_path_list[0],
                    "corrected": corr_path_list[0],
                }
            )
        else:
            print(
                f"Warning: Could not find corresponding images for {basename}. Skipping."
            )

    if not image_sets:
        print("Error: Could not find any complete image sets. Exiting.")
        return

    # 4. Create the 3x5 plot
    fig, axes = plt.subplots(3, num_images, figsize=(15, 9))

    for col_idx, img_set in enumerate(image_sets):
        # Row 1: Real
        real_img = cv2.imread(img_set["real"])
        real_img_rgb = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        axes[0, col_idx].imshow(real_img_rgb)
        axes[0, col_idx].axis("off")

        # Row 2: Generated
        gen_img = cv2.imread(img_set["generated"])
        gen_img_rgb = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
        axes[1, col_idx].imshow(gen_img_rgb)
        axes[1, col_idx].axis("off")

        # Row 3: Corrected
        corr_img = cv2.imread(img_set["corrected"])
        corr_img_rgb = cv2.cvtColor(corr_img, cv2.COLOR_BGR2RGB)
        axes[2, col_idx].imshow(corr_img_rgb)
        axes[2, col_idx].axis("off")

    # Add row titles
    axes[0, 0].set_title("Real", fontsize=14)
    axes[1, 0].set_title("Generated (S11)", fontsize=14)
    axes[2, 0].set_title("Corrected (S11)", fontsize=14)

    # Adjust titles to be row labels
    for i, row_label in enumerate(["Real", "Generated (S11)", "Corrected (S11)"]):
        axes[i, 0].set_ylabel(row_label, fontsize=14, rotation=90, labelpad=20)
        # remove the titles from the top of the columns now
        for j in range(num_images):
            axes[i, j].set_title("")

    plt.subplots_adjust(wspace=0, hspace=0)

    # 5. Save the figure
    print(f"Saving figure to {output_filename}...")
    plt.savefig(output_filename, bbox_inches="tight", pad_inches=0)
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
