from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image


class DomainAdapter(Protocol):
    name: str

    def parse_condition(self, row: pd.Series, channels: Sequence[str]) -> np.ndarray:
        ...

    def target_path(self, row: pd.Series) -> Path:
        ...

    def target_to_tensor(self, image: Image.Image, image_size: int) -> torch.Tensor:
        ...

    def sample_id(self, row: pd.Series) -> str:
        ...
