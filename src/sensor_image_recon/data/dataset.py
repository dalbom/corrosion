from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from sensor_image_recon.domains.base import DomainAdapter


class SensorImageDataset(Dataset):
    """Generic sensor-conditioned image dataset delegated to a domain adapter."""

    def __init__(
        self,
        *,
        csv_path: str | Path,
        domain_adapter: DomainAdapter,
        channels: Sequence[str],
        image_size: int,
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.domain_adapter = domain_adapter
        self.channels = list(channels)
        self.image_size = int(image_size)
        self._conditions = [
            torch.from_numpy(domain_adapter.parse_condition(row, self.channels))
            for _, row in self.df.iterrows()
        ]
        if not self._conditions:
            raise ValueError(f"Empty dataset: {self.csv_path}")
        self.cond_dim = int(self._conditions[0].numel())
        self._target_paths = [
            domain_adapter.target_path(row)
            for _, row in self.df.iterrows()
        ]
        self._sample_ids = [
            domain_adapter.sample_id(row)
            for _, row in self.df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        path = self._target_paths[idx]
        if not path.exists():
            raise FileNotFoundError(f"Target image not found: {path}")
        with Image.open(path) as image:
            target = self.domain_adapter.target_to_tensor(image, self.image_size)
        return target, self._conditions[idx].clone(), self._sample_ids[idx]

    def get_cond_dim(self) -> int:
        return self.cond_dim
