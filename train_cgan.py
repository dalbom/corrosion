"""
Training script for WGAN-GP (Wasserstein GAN with Gradient Penalty) on corrosion images.
Uses MACE (Mean Absolute Corrosion Error) for early stopping and model selection.
"""
import argparse
import os
from datetime import datetime
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from cgan.dataset import CorrosionCGANDataset
from cgan.models import Generator, Critic, compute_gradient_penalty, weights_init


def resize_for_display(images: torch.Tensor, target_size: tuple = (110, 300)) -> torch.Tensor:
    """Resize images from 128x128 to 300x110 for display."""
    resized = torch.nn.functional.interpolate(
        images, size=target_size, mode='bilinear', align_corners=False
    )
    return resized


def compute_corrosion_score(image: np.ndarray) -> float:
    """Compute corrosion score from image (0-100 scale).
    
    Args:
        image: Grayscale image array in [0, 255] range
    
    Returns:
        Corrosion score in [0, 100] range
    """
    return image.mean() / 255.0 * 100


def load_real_corrosion_scores(csv_path: str, img_root: str = "datasets/corrosion_img") -> dict:
    """
    Precompute corrosion scores for real validation images.
    
    Args:
        csv_path: Path to validation CSV file
        img_root: Root directory containing image subdirectories
    
    Returns:
        Dict mapping filename -> corrosion score
    """
    df = pd.read_csv(csv_path)
    corrosion_map = {}
    
    for _, row in df.iterrows():
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
                img = cv2.imread(img_path)
                if img is not None:
                    r_channel = img[:, :, 2]  # Red channel
                    corrosion_map[fname_with_ext] = compute_corrosion_score(r_channel)
    
    return corrosion_map


def compute_validation_mace(
    generator: torch.nn.Module,
    val_dataset: CorrosionCGANDataset,
    real_corrosion_map: dict,
    device: torch.device,
    latent_dim: int,
    sample_size: int = 64,
) -> float:
    """
    Compute MACE on a subset of validation samples.
    
    Args:
        generator: Generator model in eval mode
        val_dataset: Validation dataset
        real_corrosion_map: Dict mapping filename -> real corrosion score
        device: Torch device
        latent_dim: Latent dimension for generator
        sample_size: Number of samples to evaluate
    
    Returns:
        MACE value (lower is better)
    """
    generator.eval()
    
    # Sample random indices
    n_samples = min(sample_size, len(val_dataset))
    indices = random.sample(range(len(val_dataset)), n_samples)
    
    errors = []
    
    with torch.no_grad():
        for idx in indices:
            _, cond, filename = val_dataset[idx]
            cond = cond.unsqueeze(0).to(device)
            
            # Generate image
            noise = torch.randn(1, latent_dim, device=device)
            fake_image = generator(noise, cond)
            
            # Convert to numpy and compute corrosion score
            img_np = fake_image.cpu().numpy()[0, 0]  # [H, W]
            img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            gen_corrosion = compute_corrosion_score(img_np)
            
            # Get real corrosion score
            if not filename.endswith('.png'):
                filename = filename + '.png'
            
            real_corrosion = real_corrosion_map.get(filename)
            if real_corrosion is not None:
                errors.append(abs(real_corrosion - gen_corrosion))
    
    if errors:
        return np.mean(errors)
    return float('inf')


def train_wgan_gp(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    channels_str = "_".join(args.use_channels)
    exp_dir = Path(args.log_dir) / f"{timestamp}_{channels_str}_wgangp"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    
    # TensorBoard writer
    writer = SummaryWriter(exp_dir / "tensorboard")
    
    # Precompute real corrosion scores for validation
    print("Precomputing real validation corrosion scores...")
    real_corrosion_map = load_real_corrosion_scores(
        args.val_csv, 
        os.path.join(args.img_root, "corrosion_img")
    )
    print(f"Loaded {len(real_corrosion_map)} real corrosion scores")
    
    # Create datasets
    train_dataset = CorrosionCGANDataset(
        csv_path=args.csv,
        img_root=args.img_root,
        use_channels=args.use_channels,
        image_size=args.image_size,
    )
    
    val_dataset = CorrosionCGANDataset(
        csv_path=args.val_csv,
        img_root=args.img_root,
        use_channels=args.use_channels,
        image_size=args.image_size,
    )
    
    cond_dim = train_dataset.get_cond_dim()
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Create models
    generator = Generator(
        latent_dim=args.latent_dim,
        cond_dim=cond_dim,
        image_size=args.image_size,
        ngf=args.ngf,
    ).to(device)
    
    critic = Critic(
        cond_dim=cond_dim,
        image_size=args.image_size,
        ndf=args.ndf,
    ).to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    critic.apply(weights_init)
    
    # Optimizers - Adam with lower beta1 for WGAN
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=args.lr_c, betas=(0.0, 0.9))
    
    # Fixed noise for visualization
    num_vis = min(8, args.batch_size)
    fixed_noise = torch.randn(num_vis, args.latent_dim, device=device)
    
    # Get fixed conditioning from train and val sets
    fixed_train_cond = None
    fixed_val_cond = None
    fixed_train_real = None
    fixed_val_real = None
    
    for images, conds, _ in train_loader:
        fixed_train_cond = conds[:num_vis].to(device)
        fixed_train_real = images[:num_vis].to(device)
        break
    
    for images, conds, _ in val_loader:
        fixed_val_cond = conds[:num_vis].to(device)
        fixed_val_real = images[:num_vis].to(device)
        break
    
    # Early stopping based on MACE (lower = better)
    best_mace = float('inf')
    patience_counter = 0
    
    # Training loop
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        generator.train()
        critic.train()
        
        epoch_c_loss = 0.0
        epoch_g_loss = 0.0
        epoch_gp = 0.0
        epoch_w_distance = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for real_images, conds, _ in pbar:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            conds = conds.to(device)
            
            # ---------------------
            # Train Critic (n_critic times per G update)
            # ---------------------
            for _ in range(args.n_critic):
                critic.zero_grad()
                
                # Real images score
                score_real = critic(real_images, conds)
                
                # Fake images score
                noise = torch.randn(batch_size, args.latent_dim, device=device)
                fake_images = generator(noise, conds).detach()
                score_fake = critic(fake_images, conds)
                
                # Gradient penalty
                gp = compute_gradient_penalty(critic, real_images, fake_images, conds, device)
                
                # Wasserstein loss: maximize E[C(real)] - E[C(fake)]
                # We minimize -W_distance + lambda * GP
                w_distance = score_real.mean() - score_fake.mean()
                loss_c = -w_distance + args.lambda_gp * gp
                
                loss_c.backward()
                optimizer_C.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            generator.zero_grad()
            
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = generator(noise, conds)
            score_fake = critic(fake_images, conds)
            
            # Generator wants to maximize C(fake) -> minimize -C(fake)
            loss_g = -score_fake.mean()
            loss_g.backward()
            optimizer_G.step()
            
            # Logging
            epoch_c_loss += loss_c.item()
            epoch_g_loss += loss_g.item()
            epoch_gp += gp.item()
            epoch_w_distance += w_distance.item()
            num_batches += 1
            
            pbar.set_postfix({
                'W_dist': f'{w_distance.item():.3f}',
                'GP': f'{gp.item():.3f}',
                'G_loss': f'{loss_g.item():.3f}'
            })
            
            # TensorBoard step logging
            if global_step % 100 == 0:
                writer.add_scalar('Train/W_distance_step', w_distance.item(), global_step)
                writer.add_scalar('Train/GP_step', gp.item(), global_step)
                writer.add_scalar('Train/G_loss_step', loss_g.item(), global_step)
            
            global_step += 1
        
        # Epoch averages
        avg_c_loss = epoch_c_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        avg_gp = epoch_gp / num_batches
        avg_w_distance = epoch_w_distance / num_batches
        
        writer.add_scalar('Train/C_loss_epoch', avg_c_loss, epoch)
        writer.add_scalar('Train/G_loss_epoch', avg_g_loss, epoch)
        writer.add_scalar('Train/GP_epoch', avg_gp, epoch)
        writer.add_scalar('Train/W_distance_epoch', avg_w_distance, epoch)
        
        # ---------------------
        # Validation
        # ---------------------
        generator.eval()
        critic.eval()
        
        val_w_distance = 0.0
        val_g_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for real_images, conds, _ in val_loader:
                batch_size = real_images.size(0)
                real_images = real_images.to(device)
                conds = conds.to(device)
                
                # Score real
                score_real = critic(real_images, conds)
                
                # Score fake
                noise = torch.randn(batch_size, args.latent_dim, device=device)
                fake_images = generator(noise, conds)
                score_fake = critic(fake_images, conds)
                
                w_dist = score_real.mean() - score_fake.mean()
                g_loss = -score_fake.mean()
                
                val_w_distance += w_dist.item()
                val_g_loss += g_loss.item()
                val_batches += 1
        
        avg_val_w_distance = val_w_distance / val_batches
        avg_val_g_loss = val_g_loss / val_batches
        
        # Compute validation MACE
        val_mace = compute_validation_mace(
            generator, val_dataset, real_corrosion_map, 
            device, args.latent_dim, args.mace_sample_size
        )
        
        writer.add_scalar('Val/W_distance', avg_val_w_distance, epoch)
        writer.add_scalar('Val/G_loss', avg_val_g_loss, epoch)
        writer.add_scalar('Val/MACE', val_mace, epoch)
        
        print(f"Epoch {epoch}: Train W={avg_w_distance:.4f} GP={avg_gp:.4f} G={avg_g_loss:.4f} | "
              f"Val W={avg_val_w_distance:.4f} G={avg_val_g_loss:.4f} MACE={val_mace:.2f}")
        
        # ---------------------
        # Generate sample images
        # ---------------------
        with torch.no_grad():
            # Training set samples
            fake_train = generator(fixed_noise, fixed_train_cond)
            fake_train_display = resize_for_display(fake_train)
            real_train_display = resize_for_display(fixed_train_real)
            
            # Validation set samples
            fake_val = generator(fixed_noise, fixed_val_cond)
            fake_val_display = resize_for_display(fake_val)
            real_val_display = resize_for_display(fixed_val_real)
            
            # Normalize from [-1, 1] to [0, 1] for display
            fake_train_display = (fake_train_display + 1) / 2
            real_train_display = (real_train_display + 1) / 2
            fake_val_display = (fake_val_display + 1) / 2
            real_val_display = (real_val_display + 1) / 2
            
            # Create grids
            grid_fake_train = make_grid(fake_train_display, nrow=4, normalize=False)
            grid_real_train = make_grid(real_train_display, nrow=4, normalize=False)
            grid_fake_val = make_grid(fake_val_display, nrow=4, normalize=False)
            grid_real_val = make_grid(real_val_display, nrow=4, normalize=False)
            
            writer.add_image('Train/Generated', grid_fake_train, epoch)
            writer.add_image('Train/Real', grid_real_train, epoch)
            writer.add_image('Val/Generated', grid_fake_val, epoch)
            writer.add_image('Val/Real', grid_real_val, epoch)
        
        # ---------------------
        # Checkpointing
        # ---------------------
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_C_state_dict': optimizer_C.state_dict(),
                # Include model config for inference compatibility
                'cond_dim': cond_dim,
                'use_channels': args.use_channels,
                'image_size': args.image_size,
                'latent_dim': args.latent_dim,
                'ngf': args.ngf,
                'ndf': args.ndf,
                'val_mace': val_mace,
            }, exp_dir / f"checkpoint_epoch_{epoch:04d}.pt")
        
        # ---------------------
        # Early stopping (based on validation MACE - lower is better)
        # ---------------------
        if val_mace < best_mace:
            best_mace = val_mace
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'cond_dim': cond_dim,
                'use_channels': args.use_channels,
                'image_size': args.image_size,
                'latent_dim': args.latent_dim,
                'ngf': args.ngf,
                'ndf': args.ndf,
                'val_mace': val_mace,
            }, exp_dir / "best_model.pt")
            print(f"  -> New best model saved (MACE={val_mace:.2f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{args.patience})")
        
        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    writer.close()
    print(f"Training complete. Best MACE: {best_mace:.2f}")
    print(f"Checkpoints saved to {exp_dir}")
    
    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="Train WGAN-GP for corrosion images with MACE-based early stopping")
    
    # Data arguments
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to training CSV")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="Path to validation CSV")
    parser.add_argument("--img_root", type=str, default="datasets",
                        help="Root directory containing corrosion_img")
    parser.add_argument("--use_channels", type=str, nargs='+',
                        default=['S11'],
                        help="Sensor channels to use for conditioning")
    
    # Model arguments
    parser.add_argument("--image_size", type=int, default=128,
                        help="Training image size (square)")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Latent vector dimension")
    parser.add_argument("--ngf", type=int, default=128,
                        help="Generator feature map size")
    parser.add_argument("--ndf", type=int, default=128,
                        help="Critic feature map size")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr_g", type=float, default=1e-4,
                        help="Generator learning rate")
    parser.add_argument("--lr_c", type=float, default=1e-4,
                        help="Critic learning rate")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="Number of critic updates per generator update")
    parser.add_argument("--lambda_gp", type=float, default=10.0,
                        help="Gradient penalty coefficient")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (based on MACE)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    
    # MACE evaluation arguments
    parser.add_argument("--mace_sample_size", type=int, default=64,
                        help="Number of validation samples to use for MACE computation")
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default="logs_ext/cGAN",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("WGAN-GP Training Configuration (MACE-based early stopping)")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)
    
    train_wgan_gp(args)


if __name__ == "__main__":
    main()
