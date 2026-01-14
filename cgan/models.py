"""
Generator and Discriminator models for conditional GAN.
Uses projection-based conditioning for stable training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Generator(nn.Module):
    """
    Generator network for cGAN.
    Takes noise vector + conditioning vector, outputs single-channel image.
    
    Architecture: Fully connected -> Reshape -> Transposed Convolutions
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        cond_dim: int = 201,
        image_size: int = 128,
        ngf: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.image_size = image_size
        self.ngf = ngf
        
        # Initial projection: noise + condition -> feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, ngf * 8 * 4 * 4),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(True),
        )
        
        # Upsample + Conv blocks: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        # Using Upsample + Conv2d instead of ConvTranspose2d to avoid checkerboard artifacts
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # 64x64 -> 128x128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf, 1, 3, 1, 1, bias=False),
            nn.Tanh(),  # Output range [-1, 1]
        )
    
    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Noise tensor [B, latent_dim]
            cond: Conditioning tensor [B, cond_dim]
        
        Returns:
            Generated image [B, 1, image_size, image_size]
        """
        # Concatenate noise and condition
        x = torch.cat([z, cond], dim=1)
        
        # Project to feature space
        x = self.fc(x)
        x = x.view(-1, self.ngf * 8, 4, 4)
        
        # Generate image
        x = self.conv_blocks(x)
        
        return x


class Discriminator(nn.Module):
    """
    Discriminator network for cGAN with projection-based conditioning.
    
    Uses spectral normalization for training stability.
    Conditioning is applied via projection (inner product) instead of concatenation.
    """
    
    def __init__(
        self,
        cond_dim: int = 201,
        image_size: int = 128,
        ndf: int = 64,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.image_size = image_size
        self.ndf = ndf
        
        # Convolutional layers: 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv_blocks = nn.Sequential(
            # 128x128 -> 64x64
            nn.utils.spectral_norm(nn.Conv2d(1, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output layer (unconditional part)
        self.fc_out = nn.utils.spectral_norm(nn.Linear(ndf * 8 * 4 * 4, 1))
        
        # Projection layer for conditioning
        self.embed_cond = nn.utils.spectral_norm(nn.Linear(cond_dim, ndf * 8 * 4 * 4))
    
    def forward(self, img: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Image tensor [B, 1, image_size, image_size]
            cond: Conditioning tensor [B, cond_dim]
        
        Returns:
            Discriminator output [B, 1]
        """
        # Extract features from image
        h = self.conv_blocks(img)
        h = h.view(h.size(0), -1)  # [B, ndf * 8 * 4 * 4]
        
        # Unconditional output
        out = self.fc_out(h)
        
        # Projection-based conditioning: inner product of features and embedded condition
        embed = self.embed_cond(cond)
        proj = torch.sum(h * embed, dim=1, keepdim=True)
        
        # Combine unconditional and conditional parts
        out = out + proj
        
        return out


class Critic(nn.Module):
    """
    Critic network for WGAN-GP (Wasserstein GAN with Gradient Penalty).
    
    Key differences from Discriminator:
    - No spectral normalization (GP handles Lipschitz constraint)
    - No sigmoid output (outputs unbounded scores)
    - Uses LayerNorm instead of BatchNorm for stability
    """
    
    def __init__(
        self,
        cond_dim: int = 201,
        image_size: int = 128,
        ndf: int = 64,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.image_size = image_size
        self.ndf = ndf
        
        # Convolutional layers: 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv_blocks = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output layer (unconditional part) - no sigmoid!
        self.fc_out = nn.Linear(ndf * 8 * 4 * 4, 1)
        
        # Projection layer for conditioning
        self.embed_cond = nn.Linear(cond_dim, ndf * 8 * 4 * 4)
    
    def forward(self, img: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Image tensor [B, 1, image_size, image_size]
            cond: Conditioning tensor [B, cond_dim]
        
        Returns:
            Critic score [B, 1] (unbounded, not probability)
        """
        # Extract features from image
        h = self.conv_blocks(img)
        h = h.view(h.size(0), -1)  # [B, ndf * 8 * 4 * 4]
        
        # Unconditional output
        out = self.fc_out(h)
        
        # Projection-based conditioning
        embed = self.embed_cond(cond)
        proj = torch.sum(h * embed, dim=1, keepdim=True)
        
        # Combine unconditional and conditional parts
        out = out + proj
        
        return out


def compute_gradient_penalty(critic, real_images, fake_images, cond, device):
    """
    Compute gradient penalty for WGAN-GP.
    
    Args:
        critic: Critic network
        real_images: Real image batch [B, 1, H, W]
        fake_images: Generated image batch [B, 1, H, W]
        cond: Conditioning tensor [B, cond_dim]
        device: Torch device
    
    Returns:
        Gradient penalty loss scalar
    """
    batch_size = real_images.size(0)
    
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolated images
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)
    
    # Critic score on interpolated images
    critic_interpolated = critic(interpolated, cond)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Flatten gradients
    gradients = gradients.view(batch_size, -1)
    
    # Compute gradient norm
    gradient_norm = gradients.norm(2, dim=1)
    
    # Gradient penalty: (||grad|| - 1)^2
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


def weights_init(m):
    """Initialize network weights."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
