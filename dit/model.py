"""
Diffusion Transformer (DiT) for corrosion image generation.

Architecture: DiT-S/4 with adaLN-Zero conditioning.
Compatible with GaussianDiffusion from denoising_diffusion_pytorch.
"""

import math

import torch
import torch.nn as nn


def modulate(x, shift, scale):
    """Apply adaptive layer norm modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into vector representations via sinusoidal encoding + MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class ConditionEmbedder(nn.Module):
    """Project S-parameter conditioning vector to hidden_size."""

    def __init__(self, cond_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, cond):
        return self.net(cond)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero modulation."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        # adaLN-Zero: project conditioning to 6 * hidden_size modulation params
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # Zero-init the modulation projection so blocks are identity at init
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        # c: [B, hidden_size] conditioning embedding
        mod = self.adaLN_modulation(c)  # [B, 6 * hidden_size]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)

        # Attention branch
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP branch
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class FinalLayer(nn.Module):
    """Final layer with adaLN modulation and linear projection to patch pixels."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        # Zero-init
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        mod = self.adaLN_modulation(c)
        shift, scale = mod.chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT-S/4) for corrosion image generation.

    Interface compatible with GaussianDiffusion from denoising_diffusion_pytorch:
        forward(x, time, self_cond=None, class_labels=None) -> predicted noise

    Args:
        image_size: Input image resolution (must be divisible by patch_size).
        patch_size: Patch size for patchification.
        channels: Number of input/output image channels (1 for red-only).
        hidden_size: Transformer hidden dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        num_classes: Dimensionality of conditioning vector (cond_dim).
        mlp_ratio: MLP expansion ratio in transformer blocks.
    """

    def __init__(
        self,
        image_size=128,
        patch_size=4,
        channels=1,
        hidden_size=384,
        depth=12,
        num_heads=6,
        num_classes=402,
        mlp_ratio=4.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_size = hidden_size

        # GaussianDiffusion compatibility attributes
        self.channels = channels
        self.out_dim = channels
        self.self_condition = False

        # Patch embedding: Conv2d acts as linear projection of flattened patches
        self.patch_embed = nn.Conv2d(channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # Timestep and conditioning embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = ConditionEmbedder(num_classes, hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, channels)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize patch embedding like a linear layer
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)

        # Initialize transformer blocks
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Re-initialize timestep and condition embedders (MLP last layers)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.c_embedder.net[1].weight, std=0.02)
        nn.init.normal_(self.c_embedder.net[3].weight, std=0.02)

        # Re-zero adaLN modulation layers (already done in DiTBlock/FinalLayer __init__)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[1].weight)
            nn.init.zeros_(block.adaLN_modulation[1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x):
        """Convert [B, num_patches, patch_size^2 * C] back to [B, C, H, W]."""
        p = self.patch_size
        c = self.channels
        h = w = self.image_size // p
        x = x.reshape(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, h, p, w, p]
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def forward(self, x, time, self_cond=None, class_labels=None):
        """
        Forward pass predicting noise.

        Args:
            x: Noisy image [B, C, H, W].
            time: Timestep indices [B].
            self_cond: Unused (for GaussianDiffusion compatibility).
            class_labels: S-parameter conditioning vector [B, cond_dim].

        Returns:
            Predicted noise [B, C, H, W].
        """
        # Patchify
        x = self.patch_embed(x)  # [B, hidden_size, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_size]
        x = x + self.pos_embed

        # Conditioning: timestep + S-parameter embeddings
        t_emb = self.t_embedder(time)  # [B, hidden_size]
        if class_labels is not None:
            c_emb = self.c_embedder(class_labels)  # [B, hidden_size]
            c = t_emb + c_emb
        else:
            c = t_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer and unpatchify
        x = self.final_layer(x, c)  # [B, num_patches, patch_size^2 * C]
        x = self.unpatchify(x)  # [B, C, H, W]
        return x
