"""
Dataset class for cGAN training on corrosion images.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CorrosionCGANDataset(Dataset):
    """
    Dataset for corrosion images with sensor conditioning.
    
    Args:
        csv_path: Path to CSV file with image filenames and sensor data
        img_root: Root directory containing corrosion_img subdirectories
        use_channels: List of sensor channels to use for conditioning
                     (e.g., ['S11', 'S21', 'Phase11', 'Phase21'])
        image_size: Target image size (square)
    """
    
    SENSOR_COLUMNS = ['S11', 'S21', 'Phase11', 'Phase21']
    
    def __init__(
        self,
        csv_path: str,
        img_root: str,
        use_channels: List[str],
        image_size: int = 128,
    ):
        self.csv_path = Path(csv_path)
        self.img_root = Path(img_root)
        self.use_channels = use_channels
        self.image_size = image_size
        
        # Validate channels
        for ch in use_channels:
            if ch not in self.SENSOR_COLUMNS:
                raise ValueError(f"Unknown channel: {ch}. Must be one of {self.SENSOR_COLUMNS}")
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Image transform: resize to square, keep red channel only, normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize([0.5], [0.5]),  # [-1, 1] for single channel
        ])
        
        # Parse conditioning vectors
        self._parse_conditioning()
    
    def _parse_conditioning(self):
        """Parse conditioning vectors from CSV columns."""
        self.conditions = []
        
        for idx in range(len(self.df)):
            cond_parts = []
            for ch in self.use_channels:
                values_str = self.df.iloc[idx][ch]
                values = np.array([float(x) for x in values_str.split()], dtype=np.float32)
                cond_parts.append(values)
            
            # Concatenate all channel values
            cond = np.concatenate(cond_parts)
            self.conditions.append(cond)
        
        self.cond_dim = len(self.conditions[0])
        print(f"Conditioning dimension: {self.cond_dim} ({len(self.use_channels)} channels)")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        filename = row['filename']
        
        # Parse sample index from filename (e.g., "0525_61_30.89_augmented.png" -> "61")
        parts = filename.split('_')
        sample_idx = parts[1]
        
        # Load image
        # Add .png extension if not present
        if not filename.endswith('.png'):
            filename_with_ext = filename + '.png'
        else:
            filename_with_ext = filename
        
        img_path = self.img_root / "corrosion_img" / sample_idx / filename_with_ext
        img = Image.open(img_path).convert('RGB')
        
        # Extract red channel only
        r_channel = img.split()[0]
        
        # Apply transforms
        img_tensor = self.transform(r_channel)  # [1, H, W], range [-1, 1]
        
        # Get conditioning vector
        cond = torch.from_numpy(self.conditions[idx])
        
        return img_tensor, cond, filename
    
    def get_cond_dim(self) -> int:
        """Return conditioning vector dimension."""
        return self.cond_dim


def get_sensor_combinations() -> List[List[str]]:
    """
    Generate all 15 non-empty combinations of sensor channels.
    Returns list of channel lists, ordered by number of channels.
    """
    from itertools import combinations
    
    channels = ['S11', 'S21', 'Phase11', 'Phase21']
    all_combos = []
    
    for r in range(1, len(channels) + 1):
        for combo in combinations(channels, r):
            all_combos.append(list(combo))
    
    return all_combos
