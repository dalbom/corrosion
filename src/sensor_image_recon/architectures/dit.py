from __future__ import annotations

import torch
import torch.nn as nn

from dit.model import DiT

from sensor_image_recon.architectures.conditioning import make_cond_norm


class ConditionedDiT(nn.Module):
    """DiT wrapper with conditioning-vector normalization before the DiT embedder."""

    def __init__(
        self,
        *,
        cond_dim: int,
        image_size: int,
        patch_size: int = 4,
        channels: int = 1,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        cond_norm_type: str = "batchnorm",
    ):
        super().__init__()
        self.channels = channels
        self.out_dim = channels
        self.self_condition = False
        self.cond_norm_type = cond_norm_type
        self.cond_norm = make_cond_norm(cond_dim, cond_norm_type)
        self.model = DiT(
            image_size=image_size,
            patch_size=patch_size,
            channels=channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            num_classes=cond_dim,
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor, self_cond=None, class_labels=None):
        if class_labels is not None:
            class_labels = self.cond_norm(class_labels)
        return self.model(x, time, self_cond=self_cond, class_labels=class_labels)
