from __future__ import annotations

import torch.nn as nn


def make_cond_norm(cond_dim: int, norm_type: str) -> nn.Module:
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "batchnorm":
        return nn.BatchNorm1d(cond_dim, affine=True)
    raise ValueError(f"Unknown conditioning normalization type: {norm_type}")
