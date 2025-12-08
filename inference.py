"""
Generate Corrosion Images from Sensor Data (inference.py)
==========================================================

This script generates corrosion images using a trained conditional diffusion model.
Given sensor measurement vectors (S-parameters), it produces corresponding 
corrosion pattern images.

The generated images represent predicted corrosion patterns based on the
electromagnetic sensor readings, with intensity encoded in the RED channel.

Usage:
    python inference.py \\
        --checkpoint logs_ext/model_step_1000000.pt \\
        --csv datasets/Corrosion_test.csv \\
        --output generated/S11_S21 \\
        --use_channels S11 S21 \\
        --image_size 128

Key Features:
    - Supports both DDPM and DDIM sampling strategies
    - Handles multiple sensor channels (S11, S21, Phase11, Phase21)
    - Automatically resizes output to match original image dimensions
    - Optional MLP projection for conditioning vectors

Output:
    - Generated images saved to output directory, organized by sample index
    - Images saved as PNG with corrosion intensity in RED channel

Author: Corrosion Diffusion Project
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch.nn.functional as F

from denoising_diffusion_pytorch import KarrasUnet, GaussianDiffusion
import torch.nn as nn


def to_red_rgb(imgs: torch.Tensor) -> torch.Tensor:
    """Convert [B,1,H,W] grayscale to [B,3,H,W] keeping values only in red channel."""
    assert imgs.dim() == 4, "expected 4D tensor [B,C,H,W]"
    if imgs.size(1) == 3:
        return imgs
    assert imgs.size(1) == 1, "expected single-channel images"
    zeros = torch.zeros_like(imgs)
    return torch.cat([imgs, zeros, zeros], dim=1)


@torch.inference_mode()
def conditioned_sample(
    diffusion: GaussianDiffusion,
    *,
    batch_size: int,
    class_labels: torch.Tensor,
    return_all_timesteps: bool = False,
):
    device = diffusion.device
    (h, w), channels = diffusion.image_size, diffusion.channels
    assert class_labels.shape[0] == batch_size

    def _model_predictions(x: torch.Tensor, t_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        model_out = diffusion.model(x, t_tensor, class_labels=class_labels)
        if diffusion.objective == "pred_noise":
            pred_noise = model_out
            x_start = diffusion.predict_start_from_noise(x, t_tensor, pred_noise)
            x_start = torch.clamp(x_start, -1.0, 1.0)
            pred_noise = diffusion.predict_noise_from_start(x, t_tensor, x_start)
            return pred_noise, x_start
        elif diffusion.objective == "pred_x0":
            x_start = torch.clamp(model_out, -1.0, 1.0)
            pred_noise = diffusion.predict_noise_from_start(x, t_tensor, x_start)
            return pred_noise, x_start
        elif diffusion.objective == "pred_v":
            v = model_out
            x_start = diffusion.predict_start_from_v(x, t_tensor, v)
            x_start = torch.clamp(x_start, -1.0, 1.0)
            pred_noise = diffusion.predict_noise_from_start(x, t_tensor, x_start)
            return pred_noise, x_start
        else:
            raise ValueError(f"unknown objective {diffusion.objective}")

    def _p_sample_loop(shape):
        img = torch.randn(shape, device=device)
        imgs = [img]
        for t in tqdm(
            reversed(range(0, diffusion.num_timesteps)),
            desc="sampling loop time step",
            total=diffusion.num_timesteps,
        ):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            _, x_start = _model_predictions(img, t_tensor)
            model_mean, _, model_log_variance, _ = diffusion.q_posterior(
                x_start=x_start, x_t=img, t=t_tensor
            )
            noise = torch.randn_like(img) if t > 0 else 0
            img = model_mean + (0.5 * model_log_variance).exp() * noise
            imgs.append(img)
        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = diffusion.unnormalize(ret)
        return ret

    def _ddim_sample(shape):
        total_timesteps = diffusion.num_timesteps
        sampling_timesteps = diffusion.sampling_timesteps
        eta = diffusion.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        imgs = [img]
        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            pred_noise, x_start = _model_predictions(img, time_cond)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = diffusion.alphas_cumprod[time]
            alpha_next = diffusion.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = diffusion.unnormalize(ret)
        return ret

    shape = (batch_size, channels, h, w)
    sample_fn = _p_sample_loop if not diffusion.is_ddim_sampling else _ddim_sample
    return sample_fn(shape)


class InferenceDataset(Dataset):
    def __init__(self, csv_path: str, use_channels: List[str]):
        self.df = pd.read_csv(csv_path)
        self.use_channels = use_channels
        self.cond_dim = 201 * len(use_channels)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        cond_vectors = []
        for ch in self.use_channels:
            values = [float(x) for x in str(row[ch]).split()]
            if len(values) != 201:
                raise ValueError(
                    f"Column {ch} does not contain 201 values; got {len(values)}"
                )
            cond_vectors.append(torch.tensor(values, dtype=torch.float))
        cond = torch.cat(cond_vectors, dim=0)
        filename = row["filename"]
        try:
            sample_index = filename.split("_")[1]
        except IndexError:
            raise ValueError(f"Unexpected filename format: {filename}")
        return filename, sample_index, cond


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditioned diffusion inference")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--csv", type=str, required=True, help="Path to CSV with measurement vectors")
    p.add_argument("--output", type=str, required=True, help="Output directory for generated images")
    p.add_argument("--img_root", type=str, default="./datasets/corrosion_img", help="Root directory for original images to read target sizes")
    p.add_argument("--use_channels", nargs="+", default=["S11", "S21"], help="Columns to use as conditioning")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sampling_timesteps", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--use_mlp",
        type=int,
        default=0,
        help="If > 0, enable the same projection MLP on conditioning with this hidden size",
    )
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = InferenceDataset(csv_path=args.csv, use_channels=args.use_channels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    cond_dim = ds.cond_dim

    # load checkpoint first so we can infer architecture (channels, dim) if needed
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt)

    model = KarrasUnet(
        dim=64,
        channels=1,
        image_size=args.image_size,
        num_classes=cond_dim,
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective="pred_noise",
    ).to(device)

    # load weights non-strict to tolerate minor differences in attention placement, etc.
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        print(f"[inference] loaded with {len(missing)} missing and {len(unexpected)} unexpected keys")
    
    # optional projection MLP (must match training if used)
    projection: nn.Module
    if args.use_mlp > 0:
        projection = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, args.use_mlp),
            nn.SiLU(),
            nn.Linear(args.use_mlp, cond_dim),
        ).to(device)
        # try to load projection weights if present in checkpoint
        proj_state = ckpt.get("proj")
        if proj_state is not None:
            projection.load_state_dict(proj_state, strict=False)
        else:
            print("[inference] projection enabled but no 'proj' weights in checkpoint; using randomly initialized projection")
    else:
        projection = nn.Identity().to(device)
    model.eval()
    diffusion.eval()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dl, desc="inference"):
            filenames, sample_indices, cond = batch
            cond = cond.to(device)
            cond = projection(cond)
            bsz = cond.size(0)
            imgs = conditioned_sample(diffusion, batch_size=bsz, class_labels=cond)
            imgs_rgb = to_red_rgb(imgs)

            for img, fname, sidx in zip(imgs_rgb, filenames, sample_indices):
                save_dir = out_root / sidx
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{fname}.png"
                # Resize to original image resolution if available
                orig_path = Path(args.img_root) / sidx / f"{fname}.png"
                if orig_path.exists():
                    with Image.open(orig_path) as _im:
                        orig_w, orig_h = _im.size
                    img_resized = F.interpolate(
                        img.unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False
                    ).squeeze(0)
                    save_image(img_resized, str(save_path))
                else:
                    # save as-is if original not found
                    save_image(img, str(save_path))


if __name__ == "__main__":
    main(parse_args())


