from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


class CorrosionDomainAdapter:
    """Corrosion filename parsing, red-channel target codec, and condition parsing."""

    name = "corrosion"
    sensor_columns = ("S11", "S21", "Phase11", "Phase21")

    def __init__(self, dataset_config: dict):
        img_root = dataset_config.get("img_root", "datasets")
        self.img_root = Path(img_root)
        if self.img_root.name == "corrosion_img":
            self.image_root = self.img_root
        else:
            self.image_root = self.img_root / "corrosion_img"

    def parse_condition(self, row: pd.Series, channels: Sequence[str]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for channel in channels:
            if channel not in self.sensor_columns:
                raise ValueError(f"Unknown corrosion channel: {channel}")
            values = np.fromstring(str(row[channel]), sep=" ", dtype=np.float32)
            if values.size == 0:
                raise ValueError(f"Column {channel} has no parseable condition values")
            parts.append(values)
        return np.concatenate(parts).astype(np.float32)

    def target_path(self, row: pd.Series) -> Path:
        filename = str(row["filename"])
        filename_with_ext = filename if filename.endswith(".png") else f"{filename}.png"
        parts = filename.replace(".png", "").split("_")
        if len(parts) < 2:
            raise ValueError(f"Unexpected corrosion filename format: {filename}")
        specimen_id = parts[1]
        return self.image_root / specimen_id / filename_with_ext

    def target_to_tensor(self, image: Image.Image, image_size: int) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        red = image.split()[0]
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        return transform(red)

    def sample_id(self, row: pd.Series) -> str:
        return str(row["filename"])
