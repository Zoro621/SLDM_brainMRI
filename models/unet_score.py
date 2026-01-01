"""
SBLDM UNet Score Network
Lightweight UNet for latent diffusion score prediction.

Architecture:
- Time embedding (sinusoidal positional encoding)
- 3 down blocks with residual connections
- Bottleneck with optional attention
- 3 up blocks with skip connections
- Predicts noise ε (epsilon-prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000
) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings.
    
    From "Attention Is All You Need" (Vaswani et al., 2017).
    Modified for diffusion models.
    
    Args:
        timesteps: Tensor of timesteps [B]
        embedding_dim: Dimension of embedding
        max_period: Maximum period for sinusoids
        
    Returns:
        Embedding tensor [B, embedding_dim]
    """
    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    
    return embedding


class ResBlock(nn.Module):
    """
    Residual block with time embedding injection.
    
    Features:
    - Group normalization
    - SiLU activation
    - Optional scale-shift from time embedding
    - Dropout for regularization
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
        groups: int = 8
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        time_out_dim = out_channels * 2 if use_scale_shift_norm else out_channels
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_out_dim)
        )
        
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with time conditioning.
        
        Args:
            x: Input features [B, C, H, W]
            t_emb: Time embedding [B, time_emb_dim]
            
        Returns:
            Output features [B, out_channels, H, W]
        """
        h = x
        
        # First convolution
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time embedding
        t = self.time_emb_proj(t_emb)[:, :, None, None]
        
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(t, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift
        else:
            h = h + t
            h = self.norm2(h)
        
        # Second convolution
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for UNet.
    Applied at specified resolutions for global context.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim or channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        
        return x + self.proj(out)


class Downsample(nn.Module):
    """Strided convolution for downsampling."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Interpolation + convolution for upsampling."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNetModel(nn.Module):
    """
    Lightweight UNet for latent diffusion.
    
    Architecture:
    - Time embedding network
    - Encoder: 3 down blocks
    - Bottleneck with attention
    - Decoder: 3 up blocks with skip connections
    
    Args:
        in_channels: Input latent channels
        out_channels: Output latent channels (same as input for noise prediction)
        model_channels: Base channel count
        channel_mult: Channel multipliers for each resolution
        num_res_blocks: Number of residual blocks per resolution
        attention_resolutions: Resolutions where attention is applied
        dropout: Dropout probability
        use_scale_shift_norm: Use scale-shift normalization for time embedding
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 64,
        channel_mult: List[int] = [1, 2, 4],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [8],
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder (down path)
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        ch = model_channels
        input_block_chans = [ch]
        ds = 1  # Track spatial resolution for attention
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_emb_dim, dropout, use_scale_shift_norm)]
                ch = out_ch
                
                # Add attention if at specified resolution
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                
                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            # Downsample (except at last level)
            if level < len(channel_mult) - 1:
                self.downsamplers.append(Downsample(ch))
                input_block_chans.append(ch)
                ds *= 2
            else:
                self.downsamplers.append(None)
        
        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout, use_scale_shift_norm)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout, use_scale_shift_norm)
        
        # Decoder (up path)
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                skip_ch = input_block_chans.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout, use_scale_shift_norm)]
                ch = out_ch
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                
                self.up_blocks.append(nn.ModuleList(layers))
            
            # Upsample (except at last level)
            if level < len(channel_mult) - 1:
                self.upsamplers.append(Upsample(ch))
                ds //= 2
            else:
                self.upsamplers.append(None)
        
        # Output
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
        
        # Initialize output to zero for stable training start
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy latent [B, in_channels, H, W]
            t: Timestep [B]
            
        Returns:
            Predicted noise [B, out_channels, H, W]
        """
        # Time embedding
        t_emb = get_timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Input
        h = self.conv_in(x)
        
        # Encoder with skip connections
        skips = [h]
        
        for i, (down_block, downsampler) in enumerate(zip(self.down_blocks, self.downsamplers)):
            for layer in down_block:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            skips.append(h)
            
            if downsampler is not None:
                h = downsampler(h)
                skips.append(h)
        
        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder with skip connections
        for up_block, upsampler in zip(self.up_blocks, self.upsamplers):
            if len(skips) > 0:
                skip = skips.pop()
                # Handle potential size mismatch due to odd dimensions
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode='nearest')
                h = torch.cat([h, skip], dim=1)
            
            for layer in up_block:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            
            if upsampler is not None:
                h = upsampler(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h
    
    def get_num_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_unet(config: dict) -> UNetModel:
    """
    Create UNet from config dictionary.
    
    Args:
        config: UNet configuration
        
    Returns:
        Initialized UNet model
    """
    return UNetModel(
        in_channels=config.get('in_channels', 4),
        out_channels=config.get('out_channels', 4),
        model_channels=config.get('model_channels', 64),
        channel_mult=config.get('channel_mult', [1, 2, 4]),
        num_res_blocks=config.get('num_res_blocks', 2),
        attention_resolutions=config.get('attention_resolutions', [8]),
        dropout=config.get('dropout', 0.0),
        use_scale_shift_norm=config.get('use_scale_shift_norm', True)
    )


# Smoke test
if __name__ == '__main__':
    print("Testing UNet...")
    
    # Create model
    unet = UNetModel(
        in_channels=4,
        out_channels=4,
        model_channels=64,
        channel_mult=[1, 2, 4],
        num_res_blocks=2,
        attention_resolutions=[8],
        dropout=0.1
    )
    
    print(f"UNet parameters: {unet.get_num_params():,}")
    
    # Test forward pass
    batch_size = 4
    latent_size = 32  # 128 / 4 (VAE downsample factor)
    
    x = torch.randn(batch_size, 4, latent_size, latent_size)
    t = torch.randint(0, 1000, (batch_size,))
    
    print(f"\nInput shape: {x.shape}")
    print(f"Timesteps: {t}")
    
    noise_pred = unet(x, t)
    print(f"Output shape: {noise_pred.shape}")
    
    # Test backward pass
    loss = noise_pred.mean()
    loss.backward()
    print("\nBackward pass successful!")
    
    # Check memory usage
    if torch.cuda.is_available():
        unet = unet.cuda()
        x = x.cuda()
        t = t.cuda()
        
        torch.cuda.reset_peak_memory_stats()
        
        with torch.cuda.amp.autocast():
            noise_pred = unet(x, t)
            loss = noise_pred.mean()
        
        loss.backward()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nGPU peak memory (FP16): {peak_mem:.1f} MB")
    
    print("\nUNet smoke test passed!")
