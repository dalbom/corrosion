"""
Evaluate WGAN-GP checkpoints using MACE (Mean Absolute Corrosion Error).

This script:
1. Loads all checkpoints from a specified experiment directory
2. Generates images for each checkpoint using validation CSV conditioning
3. Computes MACE against real validation images (no post-processing)
4. Reports MACE for each checkpoint to find the best epoch

Usage:
    python evaluate_checkpoints.py --exp_dir logs_ext/cGAN/YYYYMMDD-HHMMSS_S11_wgangp
"""
import argparse
import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cgan.dataset import CorrosionCGANDataset
from cgan.models import Generator


def get_r_channel(image_path):
    """Loads an image and returns its R channel."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return img[:, :, 2]


def compute_corrosion_score(r_channel):
    """Compute corrosion score from R channel (0-100 scale)."""
    return r_channel.mean() / 255.0 * 100


def load_real_corrosion_from_csv(csv_path, img_root="datasets/corrosion_img"):
    """
    Load real images based on CSV filenames and compute their corrosion scores.
    
    Args:
        csv_path: Path to CSV file with 'filename' column
        img_root: Root directory containing subdirectories with images
    
    Returns:
        Dict mapping filename.png -> corrosion score
    """
    df = pd.read_csv(csv_path)
    corrosion_map = {}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading real images"):
        fname = row['filename']
        if not fname.endswith('.png'):
            fname_with_ext = fname + '.png'
        else:
            fname_with_ext = fname
            fname = fname.replace('.png', '')
        
        # Parse specimen ID from filename (format: MMDD_specimen_xxx)
        parts = fname.split('_')
        if len(parts) >= 2:
            specimen_id = parts[1]
            img_path = os.path.join(img_root, specimen_id, fname_with_ext)
            
            if os.path.exists(img_path):
                r_channel = get_r_channel(img_path)
                if r_channel is not None:
                    corrosion_map[fname_with_ext] = compute_corrosion_score(r_channel)
    
    return corrosion_map


def generate_and_evaluate(generator, data_loader, real_corrosion_map, device, latent_dim):
    """
    Generate images and compute MACE against real images.
    
    Returns:
        mace: Mean Absolute Corrosion Error
        errors: List of individual errors
    """
    generator.eval()
    errors = []
    
    with torch.no_grad():
        for conds, filenames in data_loader:
            batch_size = conds.size(0)
            conds = conds.to(device)
            
            # Generate images
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise, conds)
            
            # Convert to numpy and compute corrosion scores
            fake_images_np = fake_images.cpu().numpy()
            
            for i, fname in enumerate(filenames):
                # Denormalize from [-1, 1] to [0, 255]
                img = fake_images_np[i, 0]  # Single channel
                img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                
                # Compute generated corrosion score
                gen_corrosion = compute_corrosion_score(img)
                
                # Get real corrosion score
                # Filename format: 0524_18_xxx -> need to match with .png
                if not fname.endswith('.png'):
                    fname = fname + '.png'
                
                real_corrosion = real_corrosion_map.get(fname)
                if real_corrosion is not None:
                    error = abs(real_corrosion - gen_corrosion)
                    errors.append(error)
    
    if errors:
        mace = np.mean(errors)
    else:
        mace = float('inf')
    
    return mace, errors


class ConditioningDataset(torch.utils.data.Dataset):
    """Simple dataset for conditioning - returns conditions and filenames."""
    
    def __init__(self, csv_path, use_channels, img_root="datasets"):
        self.df = pd.read_csv(csv_path)
        self.use_channels = use_channels
        self.img_root = img_root
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        
        # Build conditioning vector
        cond_parts = []
        for ch in self.use_channels:
            values = np.array([float(x) for x in row[ch].split()], dtype=np.float32)
            cond_parts.append(values)
        cond = np.concatenate(cond_parts)
        
        return torch.from_numpy(cond), filename


def evaluate_checkpoint(checkpoint_path, val_csv, real_corrosion_map, device):
    """Evaluate a single checkpoint and return MACE."""
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    cond_dim = ckpt.get('cond_dim', 201)
    use_channels = ckpt.get('use_channels', ['S11'])
    image_size = ckpt.get('image_size', 128)
    latent_dim = ckpt.get('latent_dim', 128)
    ngf = ckpt.get('ngf', 128)
    
    # Create generator and load weights
    generator = Generator(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        image_size=image_size,
        ngf=ngf,
    ).to(device)
    generator.load_state_dict(ckpt['generator_state_dict'])
    generator.eval()
    
    # Create dataset from validation CSV
    val_dataset = ConditioningDataset(val_csv, use_channels)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    # Evaluate
    mace, errors = generate_and_evaluate(
        generator, val_loader, real_corrosion_map, device, latent_dim
    )
    
    return mace, len(errors)


def main():
    parser = argparse.ArgumentParser(description="Evaluate WGAN-GP checkpoints using MACE")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory containing checkpoints")
    parser.add_argument("--val_csv", type=str, default="datasets/Corrosion_cGAN_validation.csv",
                        help="Path to validation CSV")
    parser.add_argument("--img_root", type=str, default="datasets/corrosion_img",
                        help="Directory containing real images")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load real validation corrosion scores
    print("Loading real validation images...")
    real_corrosion_map = load_real_corrosion_from_csv(args.val_csv, args.img_root)
    print(f"Loaded {len(real_corrosion_map)} real validation images")
    print(f"Mean real corrosion: {np.mean(list(real_corrosion_map.values())):.2f}")
    
    # Find all checkpoints
    exp_dir = Path(args.exp_dir)
    checkpoints = sorted(glob(str(exp_dir / "checkpoint_epoch_*.pt")))
    best_model_path = exp_dir / "best_model.pt"
    
    if best_model_path.exists():
        checkpoints.append(str(best_model_path))
    
    if not checkpoints:
        print(f"No checkpoints found in {exp_dir}")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoints to evaluate")
    print("=" * 60)
    
    results = []
    
    for ckpt_path in tqdm(checkpoints, desc="Evaluating checkpoints"):
        ckpt_name = os.path.basename(ckpt_path)
        
        try:
            mace, n_samples = evaluate_checkpoint(
                ckpt_path, args.val_csv, real_corrosion_map, device
            )
            results.append({
                'checkpoint': ckpt_name,
                'mace': mace,
                'n_samples': n_samples
            })
            print(f"  {ckpt_name}: MACE = {mace:.4f} ({n_samples} samples)")
        except Exception as e:
            print(f"  {ckpt_name}: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Sorted by MACE (best first)")
    print("=" * 60)
    
    results_sorted = sorted(results, key=lambda x: x['mace'])
    for r in results_sorted:
        print(f"  {r['checkpoint']}: MACE = {r['mace']:.4f}")
    
    if results_sorted:
        best = results_sorted[0]
        print(f"\nBest checkpoint: {best['checkpoint']} (MACE = {best['mace']:.4f})")
        
        # Rename best checkpoint to best_model.pt
        if best['checkpoint'] != 'best_model.pt':
            best_ckpt_path = exp_dir / best['checkpoint']
            new_best_path = exp_dir / "best_model.pt"
            
            # Remove old best_model.pt if it exists
            if new_best_path.exists():
                print(f"\nRemoving old best_model.pt...")
                new_best_path.unlink()
            
            # Copy (not rename) so we keep the original checkpoint too
            import shutil
            shutil.copy(best_ckpt_path, new_best_path)
            print(f"Copied {best['checkpoint']} -> best_model.pt")
        else:
            print("best_model.pt is already the best checkpoint.")


if __name__ == "__main__":
    main()
