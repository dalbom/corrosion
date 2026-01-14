"""
Training script for conditional GAN on corrosion images.
Supports multiple sensor channel combinations, early stopping, and TensorBoard logging.
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from cgan.dataset import CorrosionCGANDataset
from cgan.models import Generator, Discriminator, weights_init


def resize_for_display(images: torch.Tensor, target_size: tuple = (110, 300)) -> torch.Tensor:
    """Resize images from 128x128 to 300x110 (W x H) for display."""
    # images: [B, 1, 128, 128]
    resized = torch.nn.functional.interpolate(
        images, size=target_size, mode='bilinear', align_corners=False
    )
    return resized


def train_cgan(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    channels_str = "_".join(args.use_channels)
    exp_dir = Path(args.log_dir) / f"{timestamp}_{channels_str}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    
    # TensorBoard writer
    writer = SummaryWriter(exp_dir / "tensorboard")
    
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
    
    discriminator = Discriminator(
        cond_dim=cond_dim,
        image_size=args.image_size,
        ndf=args.ndf,
    ).to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
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
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for real_images, conds, _ in pbar:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            conds = conds.to(device)
            
            # Labels with label smoothing for real (0.9 instead of 1.0)
            real_labels = torch.full((batch_size, 1), args.label_smooth, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # ---------------------
            # Train Discriminator (every d_steps)
            # ---------------------
            train_d = (global_step % args.d_steps == 0)
            discriminator.zero_grad()
            
            # Real images
            output_real = discriminator(real_images, conds)
            loss_d_real = criterion(output_real, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = generator(noise, conds)
            output_fake = discriminator(fake_images.detach(), conds)
            loss_d_fake = criterion(output_fake, fake_labels)
            
            loss_d = (loss_d_real + loss_d_fake) / 2
            
            if train_d:
                loss_d.backward()
                optimizer_D.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            generator.zero_grad()
            
            output_fake = discriminator(fake_images, conds)
            loss_g = criterion(output_fake, real_labels)
            loss_g.backward()
            optimizer_G.step()
            
            # Logging
            epoch_d_loss += loss_d.item()
            epoch_g_loss += loss_g.item()
            num_batches += 1
            
            pbar.set_postfix({'D_loss': loss_d.item(), 'G_loss': loss_g.item()})
            
            # TensorBoard step logging
            if global_step % 100 == 0:
                writer.add_scalar('Train/D_loss_step', loss_d.item(), global_step)
                writer.add_scalar('Train/G_loss_step', loss_g.item(), global_step)
            
            global_step += 1
        
        # Epoch averages
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        
        writer.add_scalar('Train/D_loss_epoch', avg_d_loss, epoch)
        writer.add_scalar('Train/G_loss_epoch', avg_g_loss, epoch)
        
        # ---------------------
        # Validation
        # ---------------------
        generator.eval()
        discriminator.eval()
        
        val_d_loss = 0.0
        val_g_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for real_images, conds, _ in val_loader:
                batch_size = real_images.size(0)
                real_images = real_images.to(device)
                conds = conds.to(device)
                
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)
                
                # D loss on real
                output_real = discriminator(real_images, conds)
                loss_d_real = criterion(output_real, real_labels)
                
                # D loss on fake
                noise = torch.randn(batch_size, args.latent_dim, device=device)
                fake_images = generator(noise, conds)
                output_fake = discriminator(fake_images, conds)
                loss_d_fake = criterion(output_fake, fake_labels)
                
                loss_d = (loss_d_real + loss_d_fake) / 2
                loss_g = criterion(output_fake, real_labels)
                
                val_d_loss += loss_d.item()
                val_g_loss += loss_g.item()
                val_batches += 1
        
        avg_val_d_loss = val_d_loss / val_batches
        avg_val_g_loss = val_g_loss / val_batches
        
        writer.add_scalar('Val/D_loss', avg_val_d_loss, epoch)
        writer.add_scalar('Val/G_loss', avg_val_g_loss, epoch)
        
        print(f"Epoch {epoch}: Train D={avg_d_loss:.4f} G={avg_g_loss:.4f} | "
              f"Val D={avg_val_d_loss:.4f} G={avg_val_g_loss:.4f}")
        
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
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, exp_dir / f"checkpoint_epoch_{epoch:04d}.pt")
        
        # ---------------------
        # Early stopping (based on combined validation loss)
        # ---------------------
        val_combined = avg_val_d_loss + avg_val_g_loss
        
        if val_combined < best_val_loss:
            best_val_loss = val_combined
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'cond_dim': cond_dim,
                'use_channels': args.use_channels,
                'image_size': args.image_size,
                'latent_dim': args.latent_dim,
                'ngf': args.ngf,
                'ndf': args.ndf,
            }, exp_dir / "best_model.pt")
            print(f"  -> New best model saved (val_loss={val_combined:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{args.patience})")
        
        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    writer.close()
    print(f"Training complete. Checkpoints saved to {exp_dir}")
    
    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="Train cGAN for corrosion images")
    
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
                        help="Discriminator feature map size")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr_g", type=float, default=2e-4,
                        help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=2e-4,
                        help="Discriminator learning rate")
    parser.add_argument("--label_smooth", type=float, default=1.0,
                        help="Label smoothing for real labels (1.0 = no smoothing)")
    parser.add_argument("--d_steps", type=int, default=1,
                        help="Train D every N steps (1=every step)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default="logs_ext/cGAN",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("cGAN Training Configuration")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)
    
    train_cgan(args)


if __name__ == "__main__":
    main()
