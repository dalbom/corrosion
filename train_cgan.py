"""
Training script for WGAN-GP (Wasserstein GAN with Gradient Penalty) on corrosion images.
Uses MACE (Mean Absolute Corrosion Error) for model selection (best checkpoint).
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
import torch.nn.functional as F
import torch.optim as optim
from pytorch_msssim import ssim as ssim_fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

import lpips as lpips_lib

from cgan.dataset import CorrosionCGANDataset
from cgan.models import Generator, Critic, PatchCritic, compute_gradient_penalty, weights_init


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


def compute_validation_recon(
    generator: torch.nn.Module,
    val_dataset: CorrosionCGANDataset,
    real_corrosion_map: dict,
    device: torch.device,
    latent_dim: int,
    sample_size: int = 64,
    lpips_fn: torch.nn.Module = None,
) -> tuple:
    """
    Compute L1, SSIM, LPIPS, and MACE on a subset of validation samples.
    All metrics are computed on raw generator output (no histogram matching).

    Returns:
        (val_l1, val_ssim, val_lpips, val_mace) — L1/LPIPS/MACE lower-better, SSIM higher-better.
        val_lpips is None if lpips_fn is None.
    """
    generator.eval()

    n_samples = min(sample_size, len(val_dataset))
    indices = random.sample(range(len(val_dataset)), n_samples)

    l1_errors = []
    ssim_vals = []
    lpips_vals = []
    mace_errors = []

    with torch.no_grad():
        for idx in indices:
            real_image, cond, filename = val_dataset[idx]
            real_image = real_image.unsqueeze(0).to(device)  # [1, 1, H, W] in [-1, 1]
            cond = cond.unsqueeze(0).to(device)

            noise = torch.randn(1, latent_dim, device=device)
            fake_image = generator(noise, cond)  # [1, 1, H, W] in [-1, 1]

            l1_errors.append(F.l1_loss(fake_image, real_image).item())
            ssim_vals.append(
                ssim_fn(fake_image, real_image, data_range=2.0, size_average=True).item()
            )
            if lpips_fn is not None:
                lpips_vals.append(
                    lpips_fn(fake_image.repeat(1, 3, 1, 1), real_image.repeat(1, 3, 1, 1)).mean().item()
                )

            img_np = fake_image.cpu().numpy()[0, 0]
            img_np = ((img_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            gen_corrosion = compute_corrosion_score(img_np)

            if not filename.endswith('.png'):
                filename = filename + '.png'
            real_corrosion = real_corrosion_map.get(filename)
            if real_corrosion is not None:
                mace_errors.append(abs(real_corrosion - gen_corrosion))

    val_l1 = float(np.mean(l1_errors)) if l1_errors else float('inf')
    val_ssim = float(np.mean(ssim_vals)) if ssim_vals else 0.0
    val_lpips = float(np.mean(lpips_vals)) if lpips_vals else None
    val_mace = float(np.mean(mace_errors)) if mace_errors else float('inf')

    return val_l1, val_ssim, val_lpips, val_mace


def train_wgan_gp(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    channels_str = "_".join(args.use_channels)
    use_l1ssim = args.lambda_l1 > 0 or args.lambda_ssim > 0
    use_perc = args.lambda_perceptual > 0
    use_recon = use_l1ssim or use_perc
    suffix_parts = ["wgangp"]
    if use_l1ssim:
        suffix_parts.append("l1ssim")
    if use_perc:
        suffix_parts.append("perc")
    if args.use_patchgan:
        suffix_parts.append("patch")
    if args.cond_norm_type == 'batchnorm':
        suffix_parts.append("bn")
    elif args.normalize_cond:
        suffix_parts.append("zscore")
    suffix = "_" + "_".join(suffix_parts)
    exp_dir = Path(args.log_dir) / f"{timestamp}_{channels_str}{suffix}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    if use_recon:
        print(f"Reconstruction loss active: lambda_l1={args.lambda_l1}, "
              f"lambda_ssim={args.lambda_ssim}, lambda_perceptual={args.lambda_perceptual}")

    # Initialize LPIPS perceptual loss if active. Frozen, eval mode.
    lpips_fn = None
    if use_perc:
        print("Initializing LPIPS (VGG) perceptual loss...")
        lpips_fn = lpips_lib.LPIPS(net='vgg', verbose=False).to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
    
    # TensorBoard writer
    writer = SummaryWriter(exp_dir / "tensorboard")
    
    # Precompute real corrosion scores for validation
    print("Precomputing real validation corrosion scores...")
    real_corrosion_map = load_real_corrosion_scores(
        args.val_csv, 
        os.path.join(args.img_root, "corrosion_img")
    )
    print(f"Loaded {len(real_corrosion_map)} real corrosion scores")
    
    # Create datasets. Conditioning normalization (if enabled) uses TRAIN-set
    # statistics for both train and val to avoid distribution shift.
    train_dataset = CorrosionCGANDataset(
        csv_path=args.csv,
        img_root=args.img_root,
        use_channels=args.use_channels,
        image_size=args.image_size,
        normalize_cond=args.normalize_cond,
    )
    val_dataset = CorrosionCGANDataset(
        csv_path=args.val_csv,
        img_root=args.img_root,
        use_channels=args.use_channels,
        image_size=args.image_size,
        normalize_cond=args.normalize_cond,
        cond_stats=train_dataset.cond_stats,
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
        cond_norm_type=args.cond_norm_type,
    ).to(device)

    if args.use_patchgan:
        print("Using PatchGAN-style critic (70x70 receptive field, mean over patches).")
        critic = PatchCritic(
            cond_dim=cond_dim,
            image_size=args.image_size,
            ndf=args.ndf,
            cond_norm_type=args.cond_norm_type,
        ).to(device)
    else:
        critic = Critic(
            cond_dim=cond_dim,
            image_size=args.image_size,
            ndf=args.ndf,
            cond_norm_type=args.cond_norm_type,
        ).to(device)
    if args.cond_norm_type != 'none':
        print(f"Cond input normalization: {args.cond_norm_type}")
    
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
    
    # Best-model selection score (lower = better).
    # When recon loss is active: val_l1 + (lambda_ssim/lambda_l1)*(1 - val_ssim) + (lambda_perceptual/lambda_l1)*val_lpips
    # — mirrors the supervised part of the training objective (adversarial term excluded; its scale drifts).
    # Otherwise: val_mace — preserves prior behavior.
    best_score = float('inf')
    best_score_metrics = {}
    if args.lambda_l1 > 0:
        ssim_to_l1_ratio = args.lambda_ssim / args.lambda_l1
        perc_to_l1_ratio = args.lambda_perceptual / args.lambda_l1
    else:
        ssim_to_l1_ratio = 0.0
        perc_to_l1_ratio = 0.0

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

            # Adversarial term: maximize C(fake) -> minimize -C(fake)
            loss_g_adv = -score_fake.mean()

            # Reconstruction terms (R-channel only; both fake and real are [B,1,H,W] in [-1,1])
            if args.lambda_l1 > 0:
                loss_g_l1 = F.l1_loss(fake_images, real_images)
            else:
                loss_g_l1 = torch.tensor(0.0, device=device)

            if args.lambda_ssim > 0:
                ssim_value = ssim_fn(fake_images, real_images, data_range=2.0, size_average=True)
                loss_g_ssim = 1.0 - ssim_value
            else:
                loss_g_ssim = torch.tensor(0.0, device=device)

            # LPIPS expects 3-channel [-1, 1] inputs; replicate single R channel 3x.
            if lpips_fn is not None:
                loss_g_perc = lpips_fn(
                    fake_images.repeat(1, 3, 1, 1),
                    real_images.repeat(1, 3, 1, 1),
                ).mean()
            else:
                loss_g_perc = torch.tensor(0.0, device=device)

            loss_g = (loss_g_adv
                      + args.lambda_l1 * loss_g_l1
                      + args.lambda_ssim * loss_g_ssim
                      + args.lambda_perceptual * loss_g_perc)
            loss_g.backward()
            optimizer_G.step()

            # Logging
            epoch_c_loss += loss_c.item()
            epoch_g_loss += loss_g.item()
            epoch_gp += gp.item()
            epoch_w_distance += w_distance.item()
            num_batches += 1

            postfix = {
                'W_dist': f'{w_distance.item():.3f}',
                'GP': f'{gp.item():.3f}',
                'G_adv': f'{loss_g_adv.item():.3f}',
            }
            if use_l1ssim:
                postfix['L1'] = f'{loss_g_l1.item():.3f}'
                postfix['1-SSIM'] = f'{loss_g_ssim.item():.3f}'
            if use_perc:
                postfix['LPIPS'] = f'{loss_g_perc.item():.3f}'
            pbar.set_postfix(postfix)

            # TensorBoard step logging
            if global_step % 100 == 0:
                writer.add_scalar('Train/W_distance_step', w_distance.item(), global_step)
                writer.add_scalar('Train/GP_step', gp.item(), global_step)
                writer.add_scalar('Train/G_loss_step', loss_g.item(), global_step)
                writer.add_scalar('Train/G_adv_step', loss_g_adv.item(), global_step)
                if use_l1ssim:
                    writer.add_scalar('Train/G_l1_step', loss_g_l1.item(), global_step)
                    writer.add_scalar('Train/G_ssim_loss_step', loss_g_ssim.item(), global_step)
                if use_perc:
                    writer.add_scalar('Train/G_lpips_step', loss_g_perc.item(), global_step)

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

        # Compute validation L1, SSIM, LPIPS, MACE on raw generator output (no histogram matching)
        val_l1, val_ssim, val_lpips, val_mace = compute_validation_recon(
            generator, val_dataset, real_corrosion_map,
            device, args.latent_dim, args.mace_sample_size,
            lpips_fn=lpips_fn,
        )

        writer.add_scalar('Val/W_distance', avg_val_w_distance, epoch)
        writer.add_scalar('Val/G_loss', avg_val_g_loss, epoch)
        writer.add_scalar('Val/L1', val_l1, epoch)
        writer.add_scalar('Val/SSIM', val_ssim, epoch)
        writer.add_scalar('Val/MACE', val_mace, epoch)
        if val_lpips is not None:
            writer.add_scalar('Val/LPIPS', val_lpips, epoch)

        lpips_str = f" LPIPS={val_lpips:.4f}" if val_lpips is not None else ""
        print(f"Epoch {epoch}: Train W={avg_w_distance:.4f} GP={avg_gp:.4f} G={avg_g_loss:.4f} | "
              f"Val W={avg_val_w_distance:.4f} G={avg_val_g_loss:.4f} "
              f"L1={val_l1:.4f} SSIM={val_ssim:.4f}{lpips_str} MACE={val_mace:.2f}")
        
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
                'cond_dim': cond_dim,
                'use_channels': args.use_channels,
                'image_size': args.image_size,
                'latent_dim': args.latent_dim,
                'ngf': args.ngf,
                'ndf': args.ndf,
                'val_l1': val_l1,
                'val_ssim': val_ssim,
                'val_lpips': val_lpips,
                'val_mace': val_mace,
                'lambda_l1': args.lambda_l1,
                'lambda_ssim': args.lambda_ssim,
                'lambda_perceptual': args.lambda_perceptual,
                'use_patchgan': args.use_patchgan,
                'normalize_cond': args.normalize_cond,
                'cond_stats': train_dataset.cond_stats,
                'cond_norm_type': args.cond_norm_type,
            }, exp_dir / f"checkpoint_epoch_{epoch:04d}.pt")

        # ---------------------
        # Save best model
        # ---------------------
        # Selection key: when recon loss is active, mirror the supervised training objective:
        #   val_l1 + (lambda_ssim/lambda_l1)*(1-SSIM) + (lambda_perceptual/lambda_l1)*val_lpips
        # When inactive, fall back to val MACE (preserves prior behavior for legacy runs).
        if use_recon:
            score_components = []
            criterion_parts = ["L1"]
            current_score = val_l1
            if args.lambda_ssim > 0:
                current_score = current_score + ssim_to_l1_ratio * (1.0 - val_ssim)
                criterion_parts.append("wSSIM")
            if val_lpips is not None and args.lambda_perceptual > 0:
                current_score = current_score + perc_to_l1_ratio * val_lpips
                criterion_parts.append("wLPIPS")
            criterion_label = "+".join(criterion_parts)
        else:
            current_score = val_mace
            criterion_label = "MACE"

        if current_score < best_score:
            best_score = current_score
            best_score_metrics = {
                'epoch': epoch,
                'val_l1': val_l1,
                'val_ssim': val_ssim,
                'val_lpips': val_lpips,
                'val_mace': val_mace,
                'score': current_score,
            }

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
                'val_l1': val_l1,
                'val_ssim': val_ssim,
                'val_lpips': val_lpips,
                'val_mace': val_mace,
                'lambda_l1': args.lambda_l1,
                'lambda_ssim': args.lambda_ssim,
                'lambda_perceptual': args.lambda_perceptual,
                'use_patchgan': args.use_patchgan,
                'normalize_cond': args.normalize_cond,
                'cond_stats': train_dataset.cond_stats,
                'cond_norm_type': args.cond_norm_type,
                'selection_criterion': criterion_label,
                'selection_score': current_score,
            }, exp_dir / "best_model.pt")
            lpips_disp = f" LPIPS={val_lpips:.4f}" if val_lpips is not None else ""
            print(f"  -> New best model saved ({criterion_label}={current_score:.4f}, "
                  f"L1={val_l1:.4f} SSIM={val_ssim:.4f}{lpips_disp} MACE={val_mace:.2f})")
        else:
            print(f"  -> No improvement ({criterion_label} best={best_score:.4f})")

    writer.close()
    bsm = best_score_metrics
    lpips_final = f" LPIPS={bsm.get('val_lpips'):.4f}" if bsm.get('val_lpips') is not None else ""
    print(f"Training complete. Best {criterion_label}={best_score:.4f} at epoch {bsm.get('epoch', '?')} "
          f"(L1={bsm.get('val_l1', float('nan')):.4f} "
          f"SSIM={bsm.get('val_ssim', float('nan')):.4f}{lpips_final} "
          f"MACE={bsm.get('val_mace', float('nan')):.2f})")
    print(f"Checkpoints saved to {exp_dir}")

    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="Train WGAN-GP for corrosion images")
    
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
    parser.add_argument("--lambda_l1", type=float, default=0.0,
                        help="Pixel-wise L1 reconstruction weight on the generator (0 disables; pix2pix uses 100)")
    parser.add_argument("--lambda_ssim", type=float, default=0.0,
                        help="(1 - SSIM) reconstruction weight on the generator (0 disables)")
    parser.add_argument("--lambda_perceptual", type=float, default=0.0,
                        help="LPIPS (VGG) perceptual loss weight on the generator (0 disables; pix2pix-HD uses 10)")
    parser.add_argument("--use_patchgan", action="store_true",
                        help="Use PatchGAN-style critic (70x70 RF, spatial mean) instead of the global projection critic.")
    parser.add_argument("--normalize_cond", action="store_true",
                        help="Z-score normalize each conditioning entry at the dataset level. Legacy fix for 4ch S+Phase scale mismatch; prefer --cond_norm_type batchnorm for new runs.")
    parser.add_argument("--cond_norm_type", type=str, default='none', choices=['none', 'batchnorm'],
                        help="Model-level cond input normalization. 'batchnorm' inserts nn.BatchNorm1d(cond_dim) at the start of Generator and Critic conditioning paths — uniform fix for any channel-scale mismatch with learnable affine recovery.")
    parser.add_argument("--patience", type=int, default=0,
                        help="Deprecated, kept for compatibility")
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
    print("WGAN-GP Training Configuration")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)
    
    train_wgan_gp(args)


if __name__ == "__main__":
    main()
