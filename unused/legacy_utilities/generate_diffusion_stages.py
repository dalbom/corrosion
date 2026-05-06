"""
Generate Diffusion Process Visualization (generate_diffusion_stages.py)
=========================================================================

This script generates images showing the diffusion denoising process at
different stages, useful for visualizing how the model progressively
refines noise into a coherent corrosion pattern.

Output Images:
    - noise_XT.png: Pure Gaussian noise (initial state at T=1000)
    - noisy_Xt.png: Intermediate noisy image at specified timestep
    - generated_X0.png: Final generated image (fully denoised)

Usage:
    python generate_diffusion_stages.py

Configuration (hardcoded in script):
    - checkpoint_path: Path to trained model checkpoint
    - capture_timestep: Which intermediate step to capture (default: 400)
    - use_channels: Sensor channels for conditioning

Output:
    - Images saved to output/figure_images/
    - Useful for paper figures explaining the diffusion process

Note:
    - Uses the first sample from test CSV for demonstration
    - Captures intermediate state mid-way through denoising

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
import torch.nn as nn

from denoising_diffusion_pytorch import KarrasUnet, GaussianDiffusion

def to_red_rgb(imgs: torch.Tensor) -> torch.Tensor:
    """Convert [B,1,H,W] grayscale to [B,3,H,W] keeping values only in red channel."""
    assert imgs.dim() == 4, "expected 4D tensor [B,C,H,W]"
    if imgs.size(1) == 3:
        return imgs
    assert imgs.size(1) == 1, "expected single-channel images"
    zeros = torch.zeros_like(imgs)
    return torch.cat([imgs, zeros, zeros], dim=1)

@torch.inference_mode()
def conditioned_sample_capture(
    diffusion: GaussianDiffusion,
    *,
    batch_size: int,
    class_labels: torch.Tensor,
    capture_timestep: int
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

    # Use standard p_sample_loop for simplicity and clarity in capturing steps
    shape = (batch_size, channels, h, w)
    img = torch.randn(shape, device=device)
    
    # Capture initial noise X_T
    noise_XT = img.clone()
    noisy_Xt = None

    for t in tqdm(
        reversed(range(0, diffusion.num_timesteps)),
        desc="sampling loop time step",
        total=diffusion.num_timesteps,
    ):
        # Capture intermediate noisy image at specified timestep
        if t == capture_timestep:
            noisy_Xt = img.clone()

        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        _, x_start = _model_predictions(img, t_tensor)
        model_mean, _, model_log_variance = diffusion.q_posterior(
            x_start=x_start, x_t=img, t=t_tensor
        )
        noise = torch.randn_like(img) if t > 0 else 0
        img = model_mean + (0.5 * model_log_variance).exp() * noise
    
    ret = diffusion.unnormalize(img)
    noise_XT = diffusion.unnormalize(noise_XT)
    if noisy_Xt is not None:
        noisy_Xt = diffusion.unnormalize(noisy_Xt)
    else:
        # Fallback if timestep not hit exactly (unlikely with range)
        noisy_Xt = torch.zeros_like(ret) 

    return noise_XT, noisy_Xt, ret

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

def main():
    # Hardcoded configuration based on run.sh and requirements
    checkpoint_path = "logs_ext/20251016-222036_S11_S21_Ph11_real/model_step_1000000.pt"
    csv_path = "datasets/Corrosion_test.csv"
    output_dir = "output/figure_images"
    use_channels = ["S11", "S21", "Phase11"]
    image_size = 128
    timesteps = 1000
    sampling_timesteps = 1000 # Use full timesteps to match training/capture logic easily
    capture_timestep = 400 # Middle of the process
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load one sample
    ds = InferenceDataset(csv_path=csv_path, use_channels=use_channels)
    # Just take the first item
    filename, sample_index, cond = ds[0]
    cond = cond.unsqueeze(0).to(device) # Add batch dimension

    cond_dim = ds.cond_dim

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt)

    model = KarrasUnet(
        dim=64,
        channels=1,
        image_size=image_size,
        num_classes=cond_dim,
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps, 
        objective="pred_noise",
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        print(f"Loaded with {len(missing)} missing and {len(unexpected)} unexpected keys")
    
    # Projection (Identity as use_mlp is likely 0 based on run.sh comments for this model, 
    # but checking run.sh line 16 it says # --use_mlp 256 is commented out. 
    # However, line 46-51 block doesn't specify --use_mlp, so it defaults to 0.
    projection = nn.Identity().to(device)
    
    model.eval()
    diffusion.eval()

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Generating images for {filename}...")
    with torch.no_grad():
        cond = projection(cond)
        noise_XT, noisy_Xt, generated_X0 = conditioned_sample_capture(
            diffusion, 
            batch_size=1, 
            class_labels=cond,
            capture_timestep=capture_timestep
        )
        
        # Convert to RGB (red channel)
        noise_XT_rgb = to_red_rgb(noise_XT)
        noisy_Xt_rgb = to_red_rgb(noisy_Xt)
        generated_X0_rgb = to_red_rgb(generated_X0)

        save_image(noise_XT_rgb, out_root / "noise_XT.png")
        save_image(noisy_Xt_rgb, out_root / "noisy_Xt.png")
        save_image(generated_X0_rgb, out_root / "generated_X0.png")
        
        print(f"Saved images to {out_root}")

if __name__ == "__main__":
    main()
