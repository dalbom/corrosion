"""
Improved diffusion training with:
  - Classifier-Free Guidance (CFG) conditioning dropout
  - x0-prediction objective (instead of noise prediction)
  - MACE auxiliary loss on predicted x0

Supports both KarrasUnet (DDPM) and DiT architectures.

Usage:
    # DiT with CFG + x0-pred + MACE loss
    python improved/train_improved.py \
        --model_type dit \
        --csv datasets/Corrosion_cGAN_train.csv \
        --val_csv datasets/Corrosion_cGAN_validation.csv \
        --img_root datasets/corrosion_img \
        --use_channels S11 Phase21 \
        --batch_size 32 --lr 2e-4 \
        --p_uncond 0.1 --lambda_mace 10.0

    # DDPM (KarrasUnet) with same improvements
    python improved/train_improved.py \
        --model_type ddpm \
        --csv datasets/Corrosion_cGAN_train.csv \
        --val_csv datasets/Corrosion_cGAN_validation.csv \
        --img_root datasets/corrosion_img \
        --use_channels S11 Phase21 \
        --batch_size 32 --lr 2e-4 \
        --p_uncond 0.1 --lambda_mace 10.0
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from einops import reduce
from tqdm import tqdm

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from denoising_diffusion_pytorch import KarrasUnet, GaussianDiffusion
from dit.model import DiT
from train import CorrosionDataset, to_red_rgb, extract


@torch.inference_mode()
def conditioned_sample_cfg(
    diffusion: GaussianDiffusion,
    model: nn.Module,
    *,
    batch_size: int,
    class_labels: torch.Tensor,
    guidance_scale: float = 1.0,
):
    """Sample with Classifier-Free Guidance.

    When guidance_scale == 1.0, this reduces to standard conditional sampling.
    Unconditional signal is always zeros (matching training dropout).
    """
    device = class_labels.device
    image_size = diffusion.image_size
    channels = diffusion.channels
    if isinstance(image_size, (tuple, list)):
        h, w = image_size
    else:
        h, w = image_size, image_size
    shape = (batch_size, channels, h, w)

    use_cfg = guidance_scale != 1.0
    null_labels = torch.zeros_like(class_labels)

    def _model_predictions(x, t_tensor):
        if use_cfg:
            # Batch both passes for efficiency
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t_tensor, t_tensor], dim=0)
            labels_double = torch.cat([class_labels, null_labels], dim=0)
            pred_both = model(x_double, t_double, class_labels=labels_double)
            pred_cond, pred_uncond = pred_both.chunk(2, dim=0)
            model_out = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            model_out = model(x, t_tensor, class_labels=class_labels)

        if diffusion.objective == "pred_noise":
            pred_noise = model_out
            x_start = diffusion.predict_start_from_noise(x, t_tensor, pred_noise)
            x_start = torch.clamp(x_start, -1.0, 1.0)
            pred_noise = diffusion.predict_noise_from_start(x, t_tensor, x_start)
        elif diffusion.objective == "pred_x0":
            x_start = torch.clamp(model_out, -1.0, 1.0)
            pred_noise = diffusion.predict_noise_from_start(x, t_tensor, x_start)
        elif diffusion.objective == "pred_v":
            x_start = diffusion.predict_start_from_v(x, t_tensor, model_out)
            x_start = torch.clamp(x_start, -1.0, 1.0)
            pred_noise = diffusion.predict_noise_from_start(x, t_tensor, x_start)
        else:
            raise ValueError(f"unknown objective {diffusion.objective}")

        return pred_noise, x_start

    # DDIM sampling
    total_timesteps = diffusion.num_timesteps
    sampling_timesteps = diffusion.sampling_timesteps
    eta = diffusion.ddim_sampling_eta

    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))

    img = torch.randn(shape, device=device)

    for time, time_next in tqdm(time_pairs, desc="sampling (CFG)", leave=False):
        time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
        pred_noise, x_start = _model_predictions(img, time_cond)

        if time_next < 0:
            img = x_start
            continue

        alpha = diffusion.alphas_cumprod[time]
        alpha_next = diffusion.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(img)
        img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

    return diffusion.unnormalize(img)


def create_model(args, cond_dim, device):
    """Create model based on model_type argument."""
    if args.model_type == "dit":
        model = DiT(
            image_size=args.image_size,
            patch_size=args.patch_size,
            channels=1,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            num_classes=cond_dim,
        )
    elif args.model_type == "ddpm":
        model = KarrasUnet(
            dim=args.dim,
            channels=1,
            image_size=args.image_size,
            num_classes=cond_dim,
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{args.model_type.upper()} params: {param_count:,} | cond_dim: {cond_dim}")
    return model


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Device: {device} | Model: {args.model_type} | Objective: {args.objective}")
    print(f"CFG p_uncond: {args.p_uncond} | MACE lambda: {args.lambda_mace} | "
          f"Guidance scale: {args.guidance_scale}")

    # Per-run directory
    channel_str = "_".join(args.use_channels)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.log_dir) / f"{args.model_type}_improved" / channel_str / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))

    # Datasets
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

    val_dataset = CorrosionDataset(
        csv_path=args.val_csv,
        img_root=args.img_root,
        use_channels=args.use_channels,
        image_size=args.image_size,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
    val_images, val_cond = next(iter(val_dataloader))
    val_cond = val_cond.to(device)

    cond_dim = dataset.cond_dim

    # Create model
    model = create_model(args, cond_dim, device)

    if args.compile != "none":
        print(f"Compiling model with mode={args.compile}...")
        model = torch.compile(model, mode=args.compile)

    # Diffusion wrapper
    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective=args.objective,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    loss_weight = diffusion.loss_weight
    use_cfg_dropout = args.p_uncond > 0

    step = 0
    while step < args.num_steps:
        for images, cond in tqdm(dataloader, desc="epoch"):
            images = images.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            b = images.size(0)

            # CFG: random conditioning dropout (zeros = unconditional)
            if use_cfg_dropout:
                cond_in = cond.clone()
                mask = torch.rand(b, device=device) < args.p_uncond
                cond_in[mask] = 0.0
            else:
                cond_in = cond

            # Forward diffusion
            x_start = diffusion.normalize(images)
            t = torch.randint(0, diffusion.num_timesteps, (b,), device=device).long()
            noise = torch.randn_like(x_start)
            x_noisy = diffusion.q_sample(x_start=x_start, t=t, noise=noise)

            # Model prediction
            model_out = model(x_noisy, t, class_labels=cond_in)

            # Compute target and pred_x0 based on objective
            if args.objective == "pred_x0":
                target = x_start
                pred_x0 = model_out
            elif args.objective == "pred_noise":
                target = noise
                pred_x0 = diffusion.predict_start_from_noise(x_noisy, t, model_out)
            elif args.objective == "pred_v":
                target = diffusion.predict_v(x_start, t, noise)
                pred_x0 = diffusion.predict_start_from_v(x_noisy, t, model_out)
            else:
                raise ValueError(f"Unknown objective: {args.objective}")

            # MSE loss with SNR weighting
            loss_mse = F.mse_loss(model_out, target, reduction="none")
            loss_mse = reduce(loss_mse, "b c h w -> b", "mean")
            weights = extract(loss_weight, t, loss_mse.shape)
            loss_mse = (loss_mse * weights).mean()

            # MACE auxiliary loss on predicted x0
            mace_loss = torch.abs(
                pred_x0.clamp(-1, 1).mean(dim=[1, 2, 3]) - x_start.mean(dim=[1, 2, 3])
            ).mean()

            # Total loss
            loss = loss_mse + args.lambda_mace * mace_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Log to TensorBoard every 100 steps
            if step % 100 == 0:
                writer.add_scalar("loss/total", loss.item(), step)
                writer.add_scalar("loss/mse", loss_mse.item(), step)
                writer.add_scalar("loss/mace", mace_loss.item(), step)

            step += 1
            if step % args.log_every == 0:
                print(f"step {step}/{args.num_steps} | "
                      f"loss: {loss.item():.6f} | mse: {loss_mse.item():.6f} | "
                      f"mace: {mace_loss.item():.6f}")

                # Validation sampling with CFG
                model.eval()
                sampled = conditioned_sample_cfg(
                    diffusion, model,
                    batch_size=args.val_batch_size,
                    class_labels=val_cond,
                    guidance_scale=args.guidance_scale,
                )
                model.train()

                sampled_rgb = to_red_rgb(sampled)
                grid = make_grid(sampled_rgb, nrow=4)
                writer.add_image("validation/sampled", grid, step)

                if step == args.log_every:
                    val_rgb = to_red_rgb(val_images)
                    writer.add_image("validation/original", make_grid(val_rgb, nrow=4), 0)

            # Save checkpoint
            if args.save_every > 0 and step % args.save_every == 0:
                ckpt_path = run_dir / f"model_step_{step}.pt"
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                print(f"Saved: {ckpt_path}")

            if step >= args.num_steps:
                break

    # Final checkpoint (skip if already saved by periodic save)
    if args.save_every <= 0 or step % args.save_every != 0:
        ckpt_path = run_dir / f"model_step_{step}.pt"
        torch.save({
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        print(f"Saved final: {ckpt_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Improved diffusion training: CFG + x0-prediction + MACE loss"
    )
    # Data
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, default="./datasets/Corrosion_test.csv")
    p.add_argument("--img_root", type=str, required=True)
    p.add_argument("--use_channels", nargs="+", default=["S11", "Phase21"])

    # Model
    p.add_argument("--model_type", type=str, required=True, choices=["ddpm", "dit"])
    p.add_argument("--image_size", type=int, default=128)
    # DDPM (KarrasUnet) args
    p.add_argument("--dim", type=int, default=64, help="KarrasUnet base dim")
    # DiT args
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--hidden_size", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=6)

    # Diffusion
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sampling_timesteps", type=int, default=250)
    p.add_argument("--objective", type=str, default="pred_x0",
                   choices=["pred_noise", "pred_x0", "pred_v"])

    # Improvements
    p.add_argument("--p_uncond", type=float, default=0.1,
                   help="CFG conditioning dropout probability (0 disables CFG)")
    p.add_argument("--lambda_mace", type=float, default=10.0,
                   help="Weight for MACE auxiliary loss (0 disables)")
    p.add_argument("--guidance_scale", type=float, default=2.0,
                   help="CFG guidance scale at inference (1.0 = no guidance)")

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_steps", type=int, default=250_000)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=50_000)
    p.add_argument("--save_every", type=int, default=200_000)
    p.add_argument("--log_dir", type=str, default="./logs_ext")
    p.add_argument("--val_batch_size", type=int, default=16)
    p.add_argument("--compile", type=str, default="reduce-overhead",
                   choices=["none", "default", "reduce-overhead", "max-autotune"])

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
