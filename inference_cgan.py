"""
Inference script for cGAN to generate corrosion images.
Generates images for the test set and saves them resized to original resolution (300x110).
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from cgan.dataset import CorrosionCGANDataset
from cgan.models import Generator


def generate_images(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Extract model config from checkpoint (with defaults for older checkpoints)
    cond_dim = checkpoint.get('cond_dim', 201)  # Default for S11 (201 values)
    use_channels = checkpoint.get('use_channels', ['S11'])
    image_size = checkpoint.get('image_size', 128)
    latent_dim = checkpoint.get('latent_dim', 128)
    ngf = checkpoint.get('ngf', 128)
    
    print(f"Model config: cond_dim={cond_dim}, channels={use_channels}, "
          f"image_size={image_size}, latent_dim={latent_dim}")
    
    # Override channels if specified
    if args.use_channels:
        if args.use_channels != use_channels:
            print(f"WARNING: Overriding checkpoint channels {use_channels} with {args.use_channels}")
        use_channels = args.use_channels
    
    # Create generator and load weights
    generator = Generator(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        image_size=image_size,
        ngf=ngf,
    ).to(device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Create dataset for conditioning vectors
    test_dataset = CorrosionCGANDataset(
        csv_path=args.test_csv,
        img_root=args.img_root,
        use_channels=use_channels,
        image_size=image_size,
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target size for output (original resolution)
    target_size = (110, 300)  # (H, W)
    
    print(f"Generating {len(test_dataset)} images...")
    print(f"Output directory: {output_dir}")
    
    total_generated = 0
    
    with torch.no_grad():
        for batch_idx, (_, conds, filenames) in enumerate(tqdm(test_loader)):
            batch_size = conds.size(0)
            conds = conds.to(device)
            
            # Generate noise
            noise = torch.randn(batch_size, latent_dim, device=device)
            
            # Generate images
            fake_images = generator(noise, conds)  # [B, 1, 128, 128], range [-1, 1]
            
            # Resize to original resolution
            fake_images = torch.nn.functional.interpolate(
                fake_images, size=target_size, mode='bilinear', align_corners=False
            )
            
            # Convert from [-1, 1] to [0, 1]
            fake_images = (fake_images + 1) / 2
            fake_images = fake_images.clamp(0, 1)
            
            # Save each image
            for i, filename in enumerate(filenames):
                img = fake_images[i]  # [1, H, W]
                
                # Convert to RGB (replicate red channel to all 3 channels)
                img_rgb = img.repeat(3, 1, 1)  # [3, H, W]
                
                # Only keep red channel (set G and B to 0)
                img_rgb[1] = 0  # Green
                img_rgb[2] = 0  # Blue
                
                # Parse specimen index from filename (e.g., "0525_61_30.89_augmented" -> "61")
                parts = filename.split('_')
                specimen_idx = parts[1]
                
                # Create specimen subfolder
                specimen_dir = output_dir / specimen_idx
                specimen_dir.mkdir(parents=True, exist_ok=True)
                
                # Create output filename - ensure .png extension
                if not filename.endswith('.png'):
                    filename = filename + '.png'
                output_path = specimen_dir / filename
                
                # Save image
                save_image(img_rgb, output_path)
                total_generated += 1
    
    print(f"Generated {total_generated} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate corrosion images with cGAN")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to test CSV file")
    parser.add_argument("--img_root", type=str, default="datasets",
                        help="Root directory containing corrosion_img (for dataset)")
    parser.add_argument("--output_dir", type=str, default="generated_cGAN",
                        help="Output directory for generated images")
    parser.add_argument("--use_channels", type=str, nargs='+', default=None,
                        help="Override sensor channels (defaults to checkpoint config)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for generation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    
    args = parser.parse_args()
    
    generate_images(args)


if __name__ == "__main__":
    main()
