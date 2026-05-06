"""
Train a DiT (Diffusion Transformer) to generate corrosion images conditioned
on S-parameter measurement vectors.

Based on train.py but replaces KarrasUnet with DiT-S/4. Conditioning is
handled inside the DiT model via adaLN-Zero (no external projection MLP).

Usage:
    python train_dit.py \
        --csv datasets/Corrosion_cGAN_train.csv \
        --val_csv datasets/Corrosion_cGAN_validation.csv \
        --img_root datasets/corrosion_img \
        --use_channels S11 S21 \
        --image_size 128 --num_steps 1000000 \
        --batch_size 8 --lr 1e-4
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from datetime import datetime
from einops import reduce
from tqdm import tqdm
from PIL import Image

from denoising_diffusion_pytorch import GaussianDiffusion
from dit.model import DiT
from train import CorrosionDataset, conditioned_sample, extract, to_red_rgb


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Device: {device} | TF32: enabled | cuDNN benchmark: True | compile: {args.compile}")

    # Per-run directory
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.log_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))

    # Training dataset
    dataset = CorrosionDataset(
        csv_path=args.csv,
        img_root=args.img_root,
        use_channels=args.use_channels,
        image_size=args.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Validation dataset and fixed batch
    val_dataset = CorrosionDataset(
        csv_path=args.val_csv,
        img_root=args.img_root,
        use_channels=args.use_channels,
        image_size=args.image_size,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
    val_images, val_cond = next(iter(val_dataloader))
    val_cond = val_cond.to(device)

    # Determine original aspect ratio for validation visualizations
    first_row = val_dataset.df.iloc[0]
    first_filename = first_row["filename"]
    first_sample_index = first_filename.split("_")[1]
    first_img_path = val_dataset.img_root / first_sample_index / f"{first_filename}.png"
    with Image.open(first_img_path) as _im:
        orig_w, orig_h = _im.size
    if orig_w >= orig_h:
        restored_target_size = (args.image_size, int(round(args.image_size * orig_w / orig_h)))
    else:
        restored_target_size = (int(round(args.image_size * orig_h / orig_w)), args.image_size)

    cond_dim = dataset.cond_dim

    # Instantiate DiT
    model = DiT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        channels=1,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=cond_dim,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"DiT params: {param_count:,} | cond_dim: {cond_dim}")

    if args.compile != "none":
        print(f"Compiling model with mode={args.compile}...")
        model = torch.compile(model, mode=args.compile)
        print("Model compiled.")

    # Diffusion wrapper
    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective="pred_noise",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    loss_weight = diffusion.loss_weight

    step = 0
    while step < args.num_steps:
        for images, cond in tqdm(dataloader):
            images = images.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            b = images.size(0)

            # Forward diffusion
            x_start = diffusion.normalize(images)
            t = torch.randint(0, diffusion.num_timesteps, (b,), device=device).long()
            noise = torch.randn_like(x_start)
            x_noisy = diffusion.q_sample(x_start=x_start, t=t, noise=noise)
            pred_noise = model(x_noisy, t, class_labels=cond)
            loss = F.mse_loss(pred_noise, noise, reduction="none")
            loss = reduce(loss, "b c h w -> b", "mean")
            weights = extract(loss_weight, t, loss.shape)
            loss = (loss * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss", loss.item(), step)

            step += 1
            if step % args.log_every == 0:
                print(f"step {step} / {args.num_steps}, loss: {loss.item():.6f}")

                # Validation images
                diffusion.eval()
                with torch.no_grad():
                    sampled_images = conditioned_sample(
                        diffusion,
                        batch_size=args.val_batch_size,
                        class_labels=val_cond,
                    )
                diffusion.train()

                sampled_images_rgb = to_red_rgb(sampled_images)
                grid = make_grid(sampled_images_rgb, nrow=4)
                writer.add_image("validation/sampled_images", grid, step)

                restored_images = F.interpolate(
                    sampled_images_rgb, size=restored_target_size, mode="bilinear", align_corners=False,
                )
                restored_grid = make_grid(restored_images, nrow=4)
                writer.add_image("validation/sampled_images_restored_aspect", restored_grid, step)

                if step == args.log_every:
                    val_images_rgb = to_red_rgb(val_images)
                    val_images_grid = make_grid(val_images_rgb, nrow=4)
                    writer.add_image("validation/original_images", val_images_grid, 0)
                    val_images_restored = F.interpolate(
                        val_images_rgb, size=restored_target_size, mode="bilinear", align_corners=False,
                    )
                    writer.add_image(
                        "validation/original_images_restored_aspect",
                        make_grid(val_images_restored, nrow=4), 0,
                    )

            if args.save_every > 0 and step % args.save_every == 0:
                checkpoint_path = run_dir / f"model_step_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                    },
                    checkpoint_path,
                )
                print(f"Saved checkpoint: {checkpoint_path}")

            if step >= args.num_steps:
                break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DiT diffusion model on corrosion dataset")
    p.add_argument("--csv", type=str, required=True, help="Path to training CSV")
    p.add_argument("--val_csv", type=str, default="./datasets/Corrosion_test.csv", help="Path to validation CSV")
    p.add_argument("--img_root", type=str, required=True, help="Root directory containing corrosion_img")
    p.add_argument("--use_channels", nargs="+", default=["S11", "S21"], help="Measurement columns for conditioning")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sampling_timesteps", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_steps", type=int, default=1_000_000)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=100_000)
    p.add_argument("--save_every", type=int, default=1_000_000)
    p.add_argument("--log_dir", type=str, default="./logs_ext/dit")
    p.add_argument("--val_batch_size", type=int, default=16)
    # DiT architecture args
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--hidden_size", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--compile", type=str, default="reduce-overhead",
                   choices=["none", "default", "reduce-overhead", "max-autotune"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
