"""
Train a diffusion model to generate corrosion images conditioned on
measurement vectors.

This script is designed for the corrosion dataset described by the user.

Dataset description
-------------------
The training CSV (``Corrosion_train.csv``) contains a ``filename`` column
and four measurement columns (``S11``, ``S21``, ``Phase11``, ``Phase21``),
each of which holds 201 space‑separated floating point values.  The target
image corresponding to a measurement row can be found under
``datasets/corrosion_img/<SAMPLE_INDEX>/<filename>.png``, where
``<SAMPLE_INDEX>`` is extracted from the second component of
``filename`` after splitting on underscores.  For example, a row with
``filename = '0525_61_30.89263840450541_augmented'`` should load its
target image from ``datasets/corrosion_img/61/0525_61_30.89263840450541_augmented.png``.

Conditioning strategy
---------------------
Phil Wang’s ``denoising_diffusion_pytorch`` library includes an
implementation of the magnitude‑preserving Karras U‑Net (`KarrasUnet`) that
supports class‑conditional generation.  If the model is constructed with
``num_classes`` set to a positive integer, it will project the provided
``class_labels`` into the time‑embedding space and add the resulting
embedding to the standard time embedding.  Crucially, the implementation
expects the ``class_labels`` tensor to have final dimension equal to
``num_classes`` when it is a floating‑point tensor; in that case the
embedding is computed by a linear layer of shape ``[num_classes, 4 * dim]``
and added to the time embedding【234346147085393†L480-L619】.  This means
continuous feature vectors can be passed directly through the
``class_labels`` argument provided their last dimension matches
``num_classes``【234346147085393†L480-L619】.  We leverage this
mechanism by setting ``num_classes`` equal to the dimensionality of the
concatenated measurement vector (201 for a single channel or 402 for
two channels).  During training, we pass the measurement vectors as
floating‑point ``class_labels`` to the network.  The built‑in
``Trainer`` class cannot handle this conditioning because it ignores
``class_labels``; therefore, we implement a custom training loop.

Usage
-----
Install the ``denoising_diffusion_pytorch`` package and ensure that
``Corrosion_train.csv`` and the ``corrosion_img`` directory are placed
under ``./datasets``.  Then run::

    python train.py --csv ./datasets/Corrosion_train.csv \
        --img_root ./datasets/corrosion_img \
        --use_channels S11 S21 \
        --image_size 128 --batch_size 32 \
        --lr 8e-5 --num_steps 700000 --save_every 10000

Arguments such as the conditioning channels and number of training steps
can be adjusted via command‑line flags.

Note
----
This training script focuses on the forward diffusion and loss
computation.  Sampling conditioned on arbitrary measurement vectors
requires overriding the sampling loop to pass ``class_labels`` at each
denoising step; this functionality is not included here and should be
implemented separately if needed.
"""

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from datetime import datetime

from denoising_diffusion_pytorch import KarrasUnet, GaussianDiffusion
import torch.nn.functional as F
import torch.nn as nn
from einops import reduce
from tqdm import tqdm


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract values from a 1‑D tensor for a batch of indices and
    reshape to match ``x_shape`` for broadcasting.

    This helper replicates the behaviour of the ``extract`` function
    defined in ``denoising_diffusion_pytorch``.

    Args:
        a: Tensor of shape [num_timesteps].
        t: Long tensor of shape [batch] containing time indices.
        x_shape: Target shape to broadcast to.

    Returns:
        Extracted values reshaped to ``[batch, 1, 1, 1]`` (for images).
    """
    b = t.size(0)
    out = a.gather(-1, t)
    return out.view(b, *([1] * (len(x_shape) - 1)))


@torch.inference_mode()
def conditioned_sample(
    diffusion: GaussianDiffusion,
    *,
    batch_size: int,
    class_labels: torch.Tensor,
    return_all_timesteps: bool = False,
):
    """Sample images while conditioning the UNet on ``class_labels`` at every step.

    This mirrors the library's sampling logic but forwards ``class_labels`` to the
    underlying ``KarrasUnet``.
    """
    device = diffusion.device
    (h, w), channels = diffusion.image_size, diffusion.channels
    assert (
        class_labels.shape[0] == batch_size
    ), "class_labels batch dimension must match batch_size"

    def _model_predictions(x: torch.Tensor, t_tensor: torch.Tensor):
        # model predicts noise for objective == 'pred_noise'
        model_out = diffusion.model(x, t_tensor, class_labels=class_labels)
        if diffusion.objective == "pred_noise":
            pred_noise = model_out
            x_start = diffusion.predict_start_from_noise(x, t_tensor, pred_noise)
            x_start = torch.clamp(x_start, -1.0, 1.0)
            # re-derive pred_noise after clamping to mimic library behavior when needed (DDIM)
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


def to_red_rgb(imgs: torch.Tensor) -> torch.Tensor:
    """Convert [B,1,H,W] grayscale to [B,3,H,W] with values only in red channel.
    If already 3 channels, returns input unchanged.
    """
    assert imgs.dim() == 4, "expected 4D tensor [B,C,H,W]"
    if imgs.size(1) == 3:
        return imgs
    assert imgs.size(1) == 1, "expected single-channel images"
    zeros = torch.zeros_like(imgs)
    return torch.cat([imgs, zeros, zeros], dim=1)


class CorrosionDataset(Dataset):
    """PyTorch dataset for the corrosion diffusion task.

    Each item consists of a normalized image tensor and a concatenated
    measurement vector serving as the conditioning signal.  Measurement
    values are parsed from the specified CSV file.
    """

    def __init__(
        self, csv_path: str, img_root: str, use_channels: List[str], image_size: int
    ):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.use_channels = use_channels
        # Determine dimensionality of the conditioning vector
        self.cond_dim = 201 * len(use_channels)
        # Build a simple transform: resize, center crop and convert to tensor
        # Normalization to [0,1] is handled by ToTensor; we scale to [-1,1] later.
        self.transform = transforms.Compose(
            [
                # Resize directly to a square without cropping to preserve full content
                transforms.Resize(
                    (image_size, image_size), interpolation=Image.BICUBIC
                ),
                transforms.ToTensor(),
            ]
        )

        # Pre-parse all conditioning vectors and image paths at init time to
        # avoid costly string→float conversion on every __getitem__ call.
        self._cond_cache: List[torch.Tensor] = []
        self._img_paths: List[Path] = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            # Parse measurement values once
            cond_vectors = []
            for ch in self.use_channels:
                values = [float(x) for x in str(row[ch]).split()]
                if len(values) != 201:
                    raise ValueError(
                        f"Column {ch} does not contain 201 values; got {len(values)}"
                    )
                cond_vectors.append(torch.tensor(values, dtype=torch.float))
            self._cond_cache.append(torch.cat(cond_vectors, dim=0))
            # Resolve image path once
            filename = row["filename"]
            try:
                sample_index = filename.split("_")[1]
            except IndexError:
                raise ValueError(f"Unexpected filename format: {filename}")
            img_path = self.img_root / sample_index / f"{filename}.png"
            if not img_path.exists():
                raise FileNotFoundError(f"Target image not found: {img_path}")
            self._img_paths.append(img_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        cond = self._cond_cache[idx]
        # Load image
        img = Image.open(self._img_paths[idx])
        # Ensure RGB, then keep only the red channel as a single-channel tensor
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_tensor = self.transform(img)  # [3,H,W] in [0,1]
        img_tensor = img_tensor[0:1, :, :]  # keep red -> [1,H,W]
        return img_tensor, cond


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable cuDNN benchmark for optimal convolution algorithm selection
    torch.backends.cudnn.benchmark = True

    # TF32: faster matmul on Ampere+ with no code overhead (unlike AMP)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Device: {device} | TF32: enabled | cuDNN benchmark: True | compile: {args.compile}")

    # Create a per-run directory under the base logs directory, formatted as YYYYMMDD-HHMMSS
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.log_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Tensorboard
    writer = SummaryWriter(log_dir=str(run_dir))

    # Prepare training dataset and dataloader
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

    # Prepare validation dataset and a fixed batch for validation imaging
    val_dataset = CorrosionDataset(
        csv_path=args.val_csv,
        img_root=args.img_root,
        use_channels=args.use_channels,
        image_size=args.image_size,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
    val_images, val_cond = next(iter(val_dataloader))
    val_cond = val_cond.to(device)

    # Determine original aspect ratio from the first validation image for restoring
    # aspect ratio in validation visualizations
    first_row = val_dataset.df.iloc[0]
    first_filename = first_row["filename"]
    try:
        first_sample_index = first_filename.split("_")[1]
    except IndexError:
        raise ValueError(f"Unexpected filename format: {first_filename}")
    first_img_path = val_dataset.img_root / first_sample_index / f"{first_filename}.png"
    if not first_img_path.exists():
        raise FileNotFoundError(f"Target image not found: {first_img_path}")
    with Image.open(first_img_path) as _im:
        orig_w, orig_h = _im.size
    if orig_w >= orig_h:
        restored_target_size = (
            args.image_size,
            int(round(args.image_size * orig_w / orig_h)),
        )  # (H, W)
    else:
        restored_target_size = (
            int(round(args.image_size * orig_h / orig_w)),
            args.image_size,
        )  # (H, W)

    # Determine conditioning dimension
    cond_dim = dataset.cond_dim

    # Optional conditioning projection MLP
    projection: nn.Module
    if args.use_mlp > 0:
        projection = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, args.use_mlp),
            nn.SiLU(),
            nn.Linear(args.use_mlp, cond_dim),
        ).to(device)
    else:
        projection = nn.Identity().to(device)

    # Instantiate KarrasUnet with class conditioning
    # Use single-channel images (red channel only)
    model = KarrasUnet(
        dim=64,
        channels=1,
        image_size=args.image_size,
        num_classes=cond_dim,
    ).to(device)

    # torch.compile: fuses small CUDA kernels to reduce launch overhead
    if args.compile != "none":
        print(f"Compiling model with mode={args.compile}...")
        model = torch.compile(model, mode=args.compile)
        print("Model compiled.")

    # Diffusion model
    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective="pred_noise",
    ).to(device)

    optimizer = torch.optim.Adam(
        list(diffusion.parameters()) + list(projection.parameters()), lr=args.lr
    )

    # Extract precomputed loss weights from the diffusion object
    loss_weight = diffusion.loss_weight

    step = 0
    while step < args.num_steps:
        for images, cond in tqdm(dataloader):
            images = images.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            b = images.size(0)

            # Forward pass
            x_start = diffusion.normalize(images)
            t = torch.randint(0, diffusion.num_timesteps, (b,), device=device).long()
            noise = torch.randn_like(x_start)
            x_noisy = diffusion.q_sample(x_start=x_start, t=t, noise=noise)
            cond_in = projection(cond)
            pred_noise = model(x_noisy, t, class_labels=cond_in)
            loss = F.mse_loss(pred_noise, noise, reduction="none")
            loss = reduce(loss, "b c h w -> b", "mean")
            weights = extract(loss_weight, t, loss.shape)
            loss = (loss * weights).mean()

            # Backward pass + optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log loss to Tensorboard
            writer.add_scalar("loss", loss.item(), step)

            step += 1
            if step % args.log_every == 0:
                print(f"step {step} / {args.num_steps}, loss: {loss.item():.6f}")

                # Generate and log validation images
                diffusion.eval()
                with torch.no_grad():
                    # Prepare projected validation conditioning if MLP is used
                    val_labels = projection(val_cond)
                    sampled_images = conditioned_sample(
                        diffusion,
                        batch_size=args.val_batch_size,
                        class_labels=val_labels,
                    )
                diffusion.train()

                # Scale generated images to [0, 1] for visualization and convert to red-only RGB
                #sampled_images = (sampled_images + 1) * 0.5  # [B,1,H,W]
                sampled_images_rgb = to_red_rgb(sampled_images)
                grid = make_grid(sampled_images_rgb, nrow=4)
                writer.add_image("validation/sampled_images", grid, step)

                # Also visualize generated images resized back to original aspect ratio
                restored_images = F.interpolate(
                    sampled_images_rgb,
                    size=restored_target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                restored_grid = make_grid(restored_images, nrow=4)
                writer.add_image(
                    "validation/sampled_images_restored_aspect", restored_grid, step
                )
                # Log original validation images once for comparison
                if step == args.log_every:
                    # val_images are [B,1,H,W] in [0,1]; convert to red-only RGB for logging
                    val_images_rgb = to_red_rgb(val_images)
                    val_images_grid = make_grid(val_images_rgb, nrow=4)
                    writer.add_image("validation/original_images", val_images_grid, 0)

                    # Additionally, log the validation images resized back to original aspect ratio
                    val_images_restored = F.interpolate(
                        val_images_rgb,
                        size=restored_target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    val_images_restored_grid = make_grid(val_images_restored, nrow=4)
                    writer.add_image(
                        "validation/original_images_restored_aspect",
                        val_images_restored_grid,
                        0,
                    )

            # Save checkpoints periodically (under the same run directory)
            if args.save_every > 0 and step % args.save_every == 0:
                checkpoint_path = run_dir / f"model_step_{step}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        **({"proj": projection.state_dict()} if args.use_mlp > 0 else {}),
                    },
                    checkpoint_path,
                )
            if step >= args.num_steps:
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a conditioned diffusion model on the corrosion dataset"
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to Corrosion_train.csv"
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="./datasets/Corrosion_test.csv",
        help="Path to Corrosion_test.csv for validation",
    )
    parser.add_argument(
        "--img_root",
        type=str,
        required=True,
        help="Path to root directory containing corrosion_img",
    )
    parser.add_argument(
        "--use_channels",
        nargs="+",
        default=["S11", "S21"],
        help="List of measurement columns to use as conditioning (e.g., S11, S21, Phase11, Phase21)",
    )
    parser.add_argument(
        "--image_size", type=int, default=128, help="Resolution of training images"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--sampling_timesteps",
        type=int,
        default=250,
        help="Number of timesteps for sampling (DDIM)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=8e-5, help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--num_steps", type=int, default=700_000, help="Total number of training steps"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--log_every", type=int, default=100, help="Logging interval in steps"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Checkpoint interval in steps (0 disables checkpoints)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./checkpoints",
        help="Directory to store checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs_ext",
        help="Directory to store Tensorboard logs",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=16,
        help="Batch size for generating validation images",
    )
    parser.add_argument(
        "--use_mlp",
        type=int,
        default=0,
        help="If > 0, enable a projection MLP on conditioning with this hidden size",
    )
    parser.add_argument(
        "--compile",
        type=str,
        default="reduce-overhead",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: reduce-overhead)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
