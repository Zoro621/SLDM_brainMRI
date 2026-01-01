"""
SBLDM Diffusion Loss Functions
Standard diffusion loss + novel frequency-aware loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class DiffusionLoss(nn.Module):
    """
    Standard diffusion training loss.
    
    Predicts noise ε and computes MSE or L1 loss.
    
    L = E_{t,x_0,ε}[ ||ε - ε_θ(x_t, t)||^2 ]
    
    Args:
        loss_type: "mse" or "l1"
        reduction: "mean" or "none"
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        reduction: str = "mean"
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            noise_pred: Predicted noise [B, C, H, W]
            noise_target: Target noise [B, C, H, W]
            weights: Optional per-sample weights [B]
            
        Returns:
            Loss value
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(noise_pred, noise_target, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise_target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Reduce over spatial and channel dimensions
        loss = loss.mean(dim=[1, 2, 3])  # [B]
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
        
        # Final reduction
        if self.reduction == "mean":
            return loss.mean()
        return loss


class FrequencyAwareLoss(nn.Module):
    """
    Frequency-aware diffusion loss (Novel contribution).
    
    Adds FFT-based penalty to encourage sharp edges and
    accurate high-frequency content in medical images.
    
    L_total = L_spatial + λ * L_freq
    
    L_freq can be:
    1. Direct FFT MSE: ||FFT(ε_pred) - FFT(ε_true)||^2
    2. High-freq weighted: weight high frequencies more
    3. Magnitude-only: compare FFT magnitudes (phase-invariant)
    
    Args:
        spatial_loss_type: "mse" or "l1" for spatial loss
        freq_loss_type: "mse" or "l1" for frequency loss
        freq_weight: Weight for frequency loss term
        high_freq_weight: Extra weight for high frequencies
        use_magnitude: Use magnitude only (ignore phase)
    """
    
    def __init__(
        self,
        spatial_loss_type: str = "mse",
        freq_loss_type: str = "mse",
        freq_weight: float = 0.1,
        high_freq_weight: float = 1.0,
        use_magnitude: bool = True
    ):
        super().__init__()
        
        self.spatial_loss = DiffusionLoss(spatial_loss_type)
        self.freq_loss_type = freq_loss_type
        self.freq_weight = freq_weight
        self.high_freq_weight = high_freq_weight
        self.use_magnitude = use_magnitude
        
        # Cache for frequency weight mask
        self._freq_weight_cache = {}
    
    def _get_freq_weights(
        self,
        h: int,
        w: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create frequency weight mask.
        
        Higher weights for high frequencies (edges, details).
        """
        cache_key = (h, w, str(device))
        
        if cache_key not in self._freq_weight_cache:
            # Create frequency coordinate grid
            freq_y = torch.fft.fftfreq(h, device=device)
            freq_x = torch.fft.fftfreq(w, device=device)
            freq_y, freq_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
            
            # Distance from DC (center of frequency spectrum)
            freq_dist = torch.sqrt(freq_y**2 + freq_x**2)
            
            # Weight mask: 1 + high_freq_weight * normalized_distance
            freq_dist_norm = freq_dist / freq_dist.max()
            weights = 1.0 + self.high_freq_weight * freq_dist_norm
            
            self._freq_weight_cache[cache_key] = weights
        
        return self._freq_weight_cache[cache_key]
    
    def _compute_freq_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute frequency domain loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W]
            target: Target tensor [B, C, H, W]
            
        Returns:
            Frequency loss value
        """
        B, C, H, W = pred.shape
        
        # 2D FFT
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        if self.use_magnitude:
            # Compare magnitudes only (phase-invariant)
            pred_mag = torch.abs(pred_fft)
            target_mag = torch.abs(target_fft)
            
            # Get frequency weights
            freq_weights = self._get_freq_weights(H, W, pred.device)
            freq_weights = freq_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # Weighted difference
            diff = (pred_mag - target_mag) * freq_weights
            
        else:
            # Compare real and imaginary parts
            diff_real = pred_fft.real - target_fft.real
            diff_imag = pred_fft.imag - target_fft.imag
            diff = torch.sqrt(diff_real**2 + diff_imag**2 + 1e-8)
        
        # Compute loss
        if self.freq_loss_type == "mse":
            loss = (diff ** 2).mean()
        else:
            loss = diff.abs().mean()
        
        return loss
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined spatial + frequency loss.
        
        Args:
            noise_pred: Predicted noise [B, C, H, W]
            noise_target: Target noise [B, C, H, W]
            weights: Optional per-sample weights
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of loss components
        """
        # Spatial loss
        spatial_loss = self.spatial_loss(noise_pred, noise_target, weights)
        
        # Frequency loss
        freq_loss = self._compute_freq_loss(noise_pred, noise_target)
        
        # Combined loss
        total_loss = spatial_loss + self.freq_weight * freq_loss
        
        loss_dict = {
            'loss': total_loss.item(),
            'spatial_loss': spatial_loss.item(),
            'freq_loss': freq_loss.item()
        }
        
        return total_loss, loss_dict


class LatentCutMix:
    """
    Latent-space CutMix augmentation (Novel contribution).
    
    Randomly cuts and mixes regions between latent codes
    during diffusion training to improve diversity.
    
    For a pair of latents (z_a, z_b):
    1. Generate random box mask M
    2. z_mixed = M * z_a + (1-M) * z_b
    3. ε_mixed = M * ε_a + (1-M) * ε_b
    
    Args:
        prob: Probability of applying CutMix
        alpha: Beta distribution parameter for box size
    """
    
    def __init__(
        self,
        prob: float = 0.25,
        alpha: float = 1.0
    ):
        self.prob = prob
        self.alpha = alpha
    
    def _sample_box(
        self,
        h: int,
        w: int,
        lam: float
    ) -> Tuple[int, int, int, int]:
        """
        Sample a random box for CutMix.
        
        Args:
            h, w: Spatial dimensions
            lam: Area ratio (from Beta distribution)
            
        Returns:
            (y1, y2, x1, x2) box coordinates
        """
        cut_ratio = (1 - lam) ** 0.5
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        cy = torch.randint(0, h, (1,)).item()
        cx = torch.randint(0, w, (1,)).item()
        
        y1 = max(0, cy - cut_h // 2)
        y2 = min(h, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(w, cx + cut_w // 2)
        
        return y1, y2, x1, x2
    
    def __call__(
        self,
        z: torch.Tensor,
        noise: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CutMix to batch of latents and noise.
        
        Shuffles batch and mixes pairs.
        
        Args:
            z: Latent codes [B, C, H, W]
            noise: Noise tensors [B, C, H, W]
            
        Returns:
            z_mixed: Mixed latents
            noise_mixed: Mixed noise
        """
        if torch.rand(1).item() > self.prob:
            return z, noise
        
        B, C, H, W = z.shape
        
        # Sample mixing coefficient
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        
        # Get box
        y1, y2, x1, x2 = self._sample_box(H, W, lam)
        
        # Shuffle batch
        indices = torch.randperm(B)
        z_shuffled = z[indices]
        noise_shuffled = noise[indices]
        
        # Mix in the box region
        z_mixed = z.clone()
        noise_mixed = noise.clone()
        
        z_mixed[:, :, y1:y2, x1:x2] = z_shuffled[:, :, y1:y2, x1:x2]
        noise_mixed[:, :, y1:y2, x1:x2] = noise_shuffled[:, :, y1:y2, x1:x2]
        
        return z_mixed, noise_mixed


# Testing
if __name__ == '__main__':
    print("Testing loss functions...")
    
    # Test standard loss
    loss_fn = DiffusionLoss(loss_type="mse")
    
    pred = torch.randn(4, 4, 32, 32)
    target = torch.randn(4, 4, 32, 32)
    
    loss = loss_fn(pred, target)
    print(f"Standard MSE loss: {loss.item():.4f}")
    
    # Test frequency-aware loss
    freq_loss_fn = FrequencyAwareLoss(
        freq_weight=0.1,
        high_freq_weight=1.0
    )
    
    total_loss, loss_dict = freq_loss_fn(pred, target)
    print(f"\nFrequency-aware loss:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    # Test CutMix
    cutmix = LatentCutMix(prob=1.0)  # Always apply for testing
    
    z = torch.randn(8, 4, 32, 32)
    noise = torch.randn(8, 4, 32, 32)
    
    z_mixed, noise_mixed = cutmix(z, noise)
    
    # Check that mixing occurred
    changed = (z != z_mixed).any().item()
    print(f"\nCutMix applied: {changed}")
    
    print("\nLoss function tests passed!")
