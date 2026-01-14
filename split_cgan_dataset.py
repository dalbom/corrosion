"""
Split Corrosion_train.csv into train/validation sets for cGAN training.

This script performs a SPECIMEN-LEVEL split to prevent data leakage:
- Extracts unique specimen indices from filenames
- Splits specimen indices into 80% train, 20% validation
- All files from a specimen go to the same split

This ensures the model is validated on completely unseen specimens.
"""
import pandas as pd
from pathlib import Path
import argparse
import numpy as np


def extract_specimen_index(filename):
    """Extract specimen index from filename (e.g., '0525_61_30.89_augmented' -> '61')."""
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Split dataset for cGAN training (specimen-level)")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="datasets/Corrosion_train.csv",
        help="Path to original training CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets",
        help="Output directory for split CSVs",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of specimens for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Read the original CSV
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples from {args.input_csv}")

    # Extract specimen indices from filenames
    df['specimen_idx'] = df['filename'].apply(extract_specimen_index)
    
    # Get unique specimen indices
    unique_specimens = df['specimen_idx'].dropna().unique()
    print(f"Found {len(unique_specimens)} unique specimen indices: {sorted(unique_specimens)}")

    # Shuffle and split specimen indices
    np.random.seed(args.seed)
    shuffled_specimens = np.random.permutation(unique_specimens)
    
    split_idx = int(len(shuffled_specimens) * args.train_ratio)
    train_specimens = set(shuffled_specimens[:split_idx])
    val_specimens = set(shuffled_specimens[split_idx:])

    print(f"\nTrain specimens ({len(train_specimens)}): {sorted(train_specimens)}")
    print(f"Validation specimens ({len(val_specimens)}): {sorted(val_specimens)}")

    # Split dataframe based on specimen indices
    df_train = df[df['specimen_idx'].isin(train_specimens)].drop(columns=['specimen_idx'])
    df_val = df[df['specimen_idx'].isin(val_specimens)].drop(columns=['specimen_idx'])

    # Save splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "Corrosion_cGAN_train.csv"
    val_path = output_dir / "Corrosion_cGAN_validation.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)

    print(f"\nTraining set: {len(df_train)} samples ({len(train_specimens)} specimens) -> {train_path}")
    print(f"Validation set: {len(df_val)} samples ({len(val_specimens)} specimens) -> {val_path}")


if __name__ == "__main__":
    main()
