"""
Generate corrosion images using a trained DiT diffusion model.

Usage:
    python inference_dit.py \
        --checkpoint logs_ext/dit/S11_S21/<run>/model_step_1000000.pt \
        --csv datasets/Corrosion_test.csv \
        --output generated_dit/S11_S21 \
        --img_root datasets/corrosion_img \
        --use_channels S11 S21
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

from denoising_diffusion_pytorch import GaussianDiffusion
from dit.model import DiT
from inference import InferenceDataset, conditioned_sample, to_red_rgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DiT diffusion inference")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to DiT checkpoint (.pt)")
    p.add_argument("--csv", type=str, required=True, help="Path to CSV with measurement vectors")
    p.add_argument("--output", type=str, required=True, help="Output directory for generated images")
    p.add_argument("--img_root", type=str, default="./datasets/corrosion_img",
                   help="Root directory for original images (to read target sizes)")
    p.add_argument("--use_channels", nargs="+", default=["S11", "S21"], help="Conditioning columns")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sampling_timesteps", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    # DiT architecture args (overridden by checkpoint if available)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--hidden_size", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=6)
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = InferenceDataset(csv_path=args.csv, use_channels=args.use_channels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    cond_dim = ds.cond_dim

    # Load checkpoint and extract architecture params if saved
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    # Strip torch.compile prefix if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Use saved args from checkpoint if available, otherwise fall back to CLI args
    saved_args = ckpt.get("args", {})
    patch_size = saved_args.get("patch_size", args.patch_size)
    hidden_size = saved_args.get("hidden_size", args.hidden_size)
    depth = saved_args.get("depth", args.depth)
    num_heads = saved_args.get("num_heads", args.num_heads)
    image_size = saved_args.get("image_size", args.image_size)
    timesteps = saved_args.get("timesteps", args.timesteps)
    sampling_timesteps = saved_args.get("sampling_timesteps", args.sampling_timesteps)

    print(f"DiT config: image_size={image_size}, patch_size={patch_size}, "
          f"hidden_size={hidden_size}, depth={depth}, num_heads={num_heads}, cond_dim={cond_dim}")

    model = DiT(
        image_size=image_size,
        patch_size=patch_size,
        channels=1,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        num_classes=cond_dim,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        print(f"[inference] loaded with {len(missing)} missing and {len(unexpected)} unexpected keys")

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        objective="pred_noise",
    ).to(device)

    model.eval()
    diffusion.eval()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dl, desc="inference"):
            filenames, sample_indices, cond = batch
            cond = cond.to(device)
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
                    save_image(img, str(save_path))


if __name__ == "__main__":
    main(parse_args())
