from __future__ import annotations

import torch
import torch.nn as nn

from denoising_diffusion_pytorch import KarrasUnet

from sensor_image_recon.architectures.conditioning import make_cond_norm


class ConditionedKarrasUnet(nn.Module):
    """Karras U-Net wrapper with conditioning-vector normalization."""

    def __init__(
        self,
        *,
        cond_dim: int,
        image_size: int,
        channels: int = 1,
        cond_norm_type: str = "batchnorm",
        dim: int = 64,
        dim_max: int = 256,
        num_downsamples: int = 3,
        num_blocks_per_stage: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.out_dim = channels
        self.self_condition = False
        self.cond_norm_type = cond_norm_type
        self.cond_norm = make_cond_norm(cond_dim, cond_norm_type)
        self.model = KarrasUnet(
            image_size=image_size,
            dim=dim,
            dim_max=dim_max,
            channels=channels,
            num_classes=cond_dim,
            num_downsamples=num_downsamples,
            num_blocks_per_stage=num_blocks_per_stage,
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor, self_cond=None, class_labels=None):
        if class_labels is not None:
            class_labels = self.cond_norm(class_labels)
        return self.model(x, time, self_cond=self_cond, class_labels=class_labels)
