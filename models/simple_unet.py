"""
Simplified UNet Score Model for Latent Diffusion
A clean, working implementation optimized for medical imaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: 1D tensor of timesteps [B]
        dim: Embedding dimension
        
    Returns:
        Embeddings [B, dim]
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2)
        )
        
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Time embedding with scale and shift
        t = self.time_mlp(t_emb)
        scale, shift = t.chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class SelfAttention(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape: (B, 3C, H, W) -> (B, 3, heads, head_dim, H*W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention: (B, heads, head_dim, HW) x (B, heads, HW, head_dim) -> (B, heads, HW, HW)
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply to values
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        
        return x + self.proj(out)


class Downsample(nn.Module):
    """2x downsampling."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x upsampling."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SimpleUNet(nn.Module):
    """
    Simplified UNet for latent diffusion.
    
    Clean architecture with proper skip connections.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 64,
        channel_mult: List[int] = [1, 2, 4],
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        attention_levels: List[int] = [2]  # Apply attention at these levels (0-indexed)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        time_dim = base_channels * 4
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Input projection
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        channels = [base_channels]
        ch = base_channels
        
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            
            # Res blocks at this level
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
                
                # Attention
                if use_attention and level in attention_levels:
                    block.append(SelfAttention(ch))
            
            self.down_blocks.append(block)
            channels.append(ch)
            
            # Downsample (except last level)
            if level < len(channel_mult) - 1:
                self.down_samples.append(Downsample(ch))
            else:
                self.down_samples.append(None)
        
        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = SelfAttention(ch) if use_attention else nn.Identity()
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = base_channels * mult
            
            # Res blocks with skip connection (2x channels)
            block = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop() if i == 0 else 0
                in_ch = ch + skip_ch if i == 0 else ch
                block.append(ResBlock(in_ch, out_ch, time_dim, dropout))
                ch = out_ch
                
                # Attention
                if use_attention and (len(channel_mult) - 1 - level) in attention_levels:
                    block.append(SelfAttention(ch))
            
            self.up_blocks.append(block)
            
            # Upsample (except last level)
            if level < len(channel_mult) - 1:
                self.up_samples.append(Upsample(ch))
            else:
                self.up_samples.append(None)
        
        # Output
        self.norm_out = nn.GroupNorm(min(8, ch), ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
        
        # Zero init for stable start
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy latent [B, in_channels, H, W]
            t: Timestep [B]
            
        Returns:
            Predicted noise [B, out_channels, H, W]
        """
        # Time embedding
        t_emb = get_timestep_embedding(t, self.conv_in.out_channels)
        t_emb = self.time_embed(t_emb)
        
        # Input
        h = self.conv_in(x)
        
        # Encoder
        skips = [h]
        for block, downsample in zip(self.down_blocks, self.down_samples):
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            skips.append(h)
            
            if downsample is not None:
                h = downsample(h)
        
        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        for block, upsample in zip(self.up_blocks, self.up_samples):
            first = True
            for layer in block:
                if isinstance(layer, ResBlock):
                    if first and len(skips) > 0:
                        skip = skips.pop()
                        # Handle size mismatch
                        if h.shape[2:] != skip.shape[2:]:
                            h = F.interpolate(h, size=skip.shape[2:], mode='nearest')
                        h = torch.cat([h, skip], dim=1)
                        first = False
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
            
            if upsample is not None:
                h = upsample(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h
    
    def get_num_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_unet(config: dict) -> SimpleUNet:
    """Create UNet from config dictionary."""
    return SimpleUNet(
        in_channels=config.get('in_channels', 4),
        out_channels=config.get('out_channels', 4),
        base_channels=config.get('model_channels', 64),
        channel_mult=config.get('channel_mult', [1, 2, 4]),
        num_res_blocks=config.get('num_res_blocks', 2),
        dropout=config.get('dropout', 0.1),
        use_attention=config.get('use_attention', True),
        attention_levels=config.get('attention_levels', [2])
    )


# Smoke test
if __name__ == '__main__':
    print("Testing SimpleUNet...")
    
    model = SimpleUNet(
        in_channels=4,
        out_channels=4,
        base_channels=64,
        channel_mult=[1, 2, 4],
        num_res_blocks=2,
        use_attention=True
    )
    
    print(f"Parameters: {model.get_num_params():,}")
    
    # Test forward
    x = torch.randn(2, 4, 16, 16)
    t = torch.randint(0, 1000, (2,))
    
    out = model(x, t)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    
    # Test backward
    loss = out.sum()
    loss.backward()
    print("Backward pass successful!")
    
    # Test with different sizes
    for size in [8, 16, 32]:
        x = torch.randn(2, 4, size, size)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        print(f"Size {size}: {x.shape} -> {out.shape}")
