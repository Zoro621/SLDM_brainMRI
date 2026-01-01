"""
SBLDM Variational Autoencoder (VAE)
Compact VAE for compressing medical images to latent space.

Architecture:
- Encoder: Progressive downsampling with residual blocks
- Decoder: Progressive upsampling with residual blocks
- Latent: Gaussian distribution with learned mean and variance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm and SiLU activation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection with optional projection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class Downsample(nn.Module):
    """Spatial downsampling with strided convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling with transposed convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for feature refinement.
    Used at bottleneck resolution for global context.
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape for multi-head attention: (B, 3*C, H, W) -> (B, 3, num_heads, head_dim, H*W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        
        return x + self.proj(out)


class Encoder(nn.Module):
    """
    VAE Encoder: Maps images to latent distribution parameters.
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        latent_channels: Number of latent channels (output channels for mean/var)
        hidden_dims: List of hidden dimensions for each resolution level
        use_attention: Whether to use attention at bottleneck
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        hidden_dims: List[int] = [32, 64, 128, 256],
        use_attention: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.hidden_dims = hidden_dims
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)
        
        # Encoder blocks
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        in_dim = hidden_dims[0]
        for i, out_dim in enumerate(hidden_dims):
            # Residual blocks at this resolution
            self.down_blocks.append(nn.ModuleList([
                ResidualBlock(in_dim, out_dim),
                ResidualBlock(out_dim, out_dim)
            ]))
            
            # Downsample (except at last level)
            if i < len(hidden_dims) - 1:
                self.downsamplers.append(Downsample(out_dim))
            else:
                self.downsamplers.append(None)
            
            in_dim = out_dim
        
        # Bottleneck
        self.mid_block1 = ResidualBlock(hidden_dims[-1], hidden_dims[-1])
        self.mid_attn = AttentionBlock(hidden_dims[-1]) if use_attention else nn.Identity()
        self.mid_block2 = ResidualBlock(hidden_dims[-1], hidden_dims[-1])
        
        # Output projection to latent parameters (mean and log-variance)
        self.norm_out = nn.GroupNorm(8, hidden_dims[-1])
        self.conv_out = nn.Conv2d(hidden_dims[-1], latent_channels * 2, 3, padding=1)
        
        # Initialize output layer to produce small values (near-zero mean, small logvar)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution parameters.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            mean: Latent mean [B, latent_channels, H', W']
            logvar: Latent log-variance [B, latent_channels, H', W']
        """
        h = self.conv_in(x)
        
        # Encoder path
        for down_block, downsampler in zip(self.down_blocks, self.downsamplers):
            for resblock in down_block:
                h = resblock(h)
            if downsampler is not None:
                h = downsampler(h)
        
        # Bottleneck
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # Split into mean and log-variance
        mean, logvar = torch.chunk(h, 2, dim=1)
        
        # Clamp for numerical stability (prevent exploding gradients)
        mean = torch.clamp(mean, min=-100.0, max=100.0)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        
        return mean, logvar


class Decoder(nn.Module):
    """
    VAE Decoder: Maps latent codes back to images.
    
    Args:
        out_channels: Number of output channels (1 for grayscale)
        latent_channels: Number of latent channels
        hidden_dims: List of hidden dimensions (reversed from encoder)
        use_attention: Whether to use attention at bottleneck
    """
    
    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 4,
        hidden_dims: List[int] = [256, 128, 64, 32],
        use_attention: bool = True
    ):
        super().__init__()
        
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.hidden_dims = hidden_dims
        
        # Input projection from latent
        self.conv_in = nn.Conv2d(latent_channels, hidden_dims[0], 3, padding=1)
        
        # Bottleneck
        self.mid_block1 = ResidualBlock(hidden_dims[0], hidden_dims[0])
        self.mid_attn = AttentionBlock(hidden_dims[0]) if use_attention else nn.Identity()
        self.mid_block2 = ResidualBlock(hidden_dims[0], hidden_dims[0])
        
        # Decoder blocks
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        in_dim = hidden_dims[0]
        for i, out_dim in enumerate(hidden_dims):
            # Residual blocks at this resolution
            self.up_blocks.append(nn.ModuleList([
                ResidualBlock(in_dim, out_dim),
                ResidualBlock(out_dim, out_dim)
            ]))
            
            # Upsample (except at last level)
            if i < len(hidden_dims) - 1:
                self.upsamplers.append(Upsample(out_dim))
            else:
                self.upsamplers.append(None)
            
            in_dim = out_dim
        
        # Output projection
        self.norm_out = nn.GroupNorm(8, hidden_dims[-1])
        self.conv_out = nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to image.
        
        Args:
            z: Latent code [B, latent_channels, H', W']
            
        Returns:
            Reconstructed image [B, out_channels, H, W]
        """
        h = self.conv_in(z)
        
        # Bottleneck
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # Decoder path
        for up_block, upsampler in zip(self.up_blocks, self.upsamplers):
            for resblock in up_block:
                h = resblock(h)
            if upsampler is not None:
                h = upsampler(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class VAE(nn.Module):
    """
    Variational Autoencoder for medical image compression.
    
    Supports:
    - Configurable latent dimensions
    - KL divergence regularization
    - Reparameterization trick
    - Encoding/decoding of images and latents
    
    Args:
        in_channels: Input image channels
        latent_channels: Latent space channels
        hidden_dims: Hidden dimensions for each resolution
        use_attention: Use attention at bottleneck
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        hidden_dims: List[int] = [32, 64, 128, 256],
        use_attention: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.hidden_dims = hidden_dims
        
        # Calculate downsampling factor
        self.downsample_factor = 2 ** (len(hidden_dims) - 1)
        
        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            hidden_dims=hidden_dims,
            use_attention=use_attention
        )
        
        self.decoder = Decoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
            hidden_dims=list(reversed(hidden_dims)),
            use_attention=use_attention
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Re-initialize encoder output to produce small initial values
        # This prevents KL explosion at start of training
        nn.init.zeros_(self.encoder.conv_out.weight)
        nn.init.zeros_(self.encoder.conv_out.bias)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            mean, logvar: Distribution parameters
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to image.
        
        Args:
            z: Latent code [B, latent_channels, H', W']
            
        Returns:
            Reconstructed image [B, C, H, W]
        """
        return self.decoder(z)
    
    def reparameterize(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mean + std * epsilon
        
        Args:
            mean: Latent mean
            logvar: Latent log-variance
            
        Returns:
            Sampled latent code
        """
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # During inference, just use mean
            return mean
    
    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode, sample, decode.
        
        Args:
            x: Input image [B, C, H, W]
            sample_posterior: Whether to sample or use mean
            
        Returns:
            recon: Reconstructed image
            mean: Latent mean
            logvar: Latent log-variance
        """
        mean, logvar = self.encode(x)
        
        if sample_posterior:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        
        recon = self.decode(z)
        
        return recon, mean, logvar
    
    def get_latent(
        self,
        x: torch.Tensor,
        sample: bool = False
    ) -> torch.Tensor:
        """
        Get latent code for an image.
        
        Args:
            x: Input image
            sample: Whether to sample or use mean
            
        Returns:
            Latent code
        """
        mean, logvar = self.encode(x)
        
        if sample:
            return self.reparameterize(mean, logvar)
        return mean
    
    @staticmethod
    def kl_divergence(
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence from standard normal.
        
        KL(q(z|x) || p(z)) where p(z) = N(0, I)
        
        Args:
            mean: Latent mean
            logvar: Latent log-variance
            
        Returns:
            KL divergence (scalar, averaged over batch)
        """
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        
        # KL divergence per element, then sum over latent dims and mean over batch
        kl_per_element = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        # Sum over spatial and channel dims, mean over batch
        kl = kl_per_element.sum(dim=[1, 2, 3]).mean()
        return kl
    
    def loss_function(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1e-5
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute VAE loss = reconstruction + KL divergence.
        
        Args:
            x: Original image
            recon: Reconstructed image
            mean: Latent mean
            logvar: Latent log-variance
            kl_weight: Weight for KL term (β in β-VAE)
            
        Returns:
            loss: Total loss
            loss_dict: Dictionary of loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = self.kl_divergence(mean, logvar)
        
        # Total loss
        loss = recon_loss + kl_weight * kl_loss
        
        return loss, {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
    
    def get_num_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_vae(config: dict) -> VAE:
    """
    Create VAE from config dictionary.
    
    Args:
        config: VAE configuration
        
    Returns:
        Initialized VAE model
    """
    return VAE(
        in_channels=config.get('in_channels', 1),
        latent_channels=config.get('latent_channels', 4),
        hidden_dims=config.get('hidden_dims', [32, 64, 128, 256]),
        use_attention=config.get('use_attention', True)
    )


# Smoke test
if __name__ == '__main__':
    print("Testing VAE...")
    
    # Create model
    vae = VAE(
        in_channels=1,
        latent_channels=4,
        hidden_dims=[32, 64, 128, 256],
        use_attention=True
    )
    
    print(f"VAE parameters: {vae.get_num_params():,}")
    print(f"Downsample factor: {vae.downsample_factor}")
    
    # Test forward pass
    x = torch.randn(4, 1, 128, 128)
    print(f"\nInput shape: {x.shape}")
    
    recon, mean, logvar = vae(x)
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Mean shape: {mean.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test loss
    loss, loss_dict = vae.loss_function(x, recon, mean, logvar)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    
    # Test latent extraction
    z = vae.get_latent(x)
    print(f"\nLatent shape: {z.shape}")
    
    # Test decoding
    recon_from_z = vae.decode(z)
    print(f"Decoded shape: {recon_from_z.shape}")
    
    # Test backward pass
    loss.backward()
    print("\nBackward pass successful!")
    
    # Check memory usage
    if torch.cuda.is_available():
        vae = vae.cuda()
        x = x.cuda()
        
        torch.cuda.reset_peak_memory_stats()
        recon, mean, logvar = vae(x)
        loss, _ = vae.loss_function(x, recon, mean, logvar)
        loss.backward()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nGPU peak memory: {peak_mem:.1f} MB")
    
    print("\nVAE smoke test passed!")
