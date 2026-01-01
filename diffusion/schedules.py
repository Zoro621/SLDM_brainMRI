"""
SBLDM Noise Schedules
Implements various beta schedules for diffusion:
- Linear schedule
- Cosine schedule  
- Gamma-rebalanced schedule (Novel contribution)
"""

import torch
import numpy as np
from typing import Optional, Tuple


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02
) -> torch.Tensor:
    """
    Linear beta schedule as in original DDPM paper.
    
    β(t) = β_start + (β_end - β_start) * t / T
    
    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
        
    Returns:
        Beta values [timesteps]
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008
) -> torch.Tensor:
    """
    Cosine beta schedule as in "Improved DDPM" paper.
    
    Provides smoother noise schedule that works better for
    images with fine details.
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent singularities
        
    Returns:
        Beta values [timesteps]
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    
    # Cumulative alpha values from cosine schedule
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    # Convert to betas
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    # Clip to prevent numerical issues
    return torch.clip(betas, 0.0001, 0.9999)


def gamma_rebalanced_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    gamma: float = 1.0
) -> torch.Tensor:
    """
    Gamma-rebalanced noise schedule (Novel contribution).
    
    β(t) = β_start + (β_end - β_start) * (t/T)^γ
    
    - γ < 1: More noise added early, gentler denoising at end
             (Hypothesis: Better for preserving fine anatomical details)
    - γ = 1: Equivalent to linear schedule
    - γ > 1: Less noise early, aggressive denoising at end
             (Hypothesis: May help with global structure)
    
    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
        gamma: Rebalancing exponent
        
    Returns:
        Beta values [timesteps]
    """
    t = torch.linspace(0, 1, timesteps)
    t_rebalanced = t ** gamma
    
    betas = beta_start + (beta_end - beta_start) * t_rebalanced
    
    return betas


def get_beta_schedule(
    schedule_type: str,
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    gamma: float = 1.0,
    cosine_s: float = 0.008
) -> torch.Tensor:
    """
    Get beta schedule by name.
    
    Args:
        schedule_type: "linear", "cosine", or "gamma_rebalanced"
        timesteps: Number of diffusion steps
        beta_start: Starting beta
        beta_end: Ending beta
        gamma: Gamma for rebalanced schedule
        cosine_s: S parameter for cosine schedule
        
    Returns:
        Beta values tensor
    """
    if schedule_type == "linear":
        return linear_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_type == "cosine":
        return cosine_beta_schedule(timesteps, cosine_s)
    elif schedule_type == "gamma_rebalanced":
        return gamma_rebalanced_schedule(timesteps, beta_start, beta_end, gamma)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def compute_alpha_schedule(
    betas: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute alpha values from betas.
    
    α_t = 1 - β_t
    ᾱ_t = ∏_{s=1}^{t} α_s (cumulative product)
    
    Args:
        betas: Beta schedule [timesteps]
        
    Returns:
        alphas: Alpha values [timesteps]
        alphas_cumprod: Cumulative product of alphas [timesteps]
        alphas_cumprod_prev: Shifted cumulative alphas [timesteps]
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    
    return alphas, alphas_cumprod, alphas_cumprod_prev


class NoiseSchedule:
    """
    Complete noise schedule with precomputed values.
    
    Provides all values needed for:
    - Forward diffusion (adding noise)
    - Training (noise prediction targets)
    - Sampling (reverse diffusion)
    
    Args:
        schedule_type: Type of schedule
        timesteps: Number of steps
        beta_start: Starting beta
        beta_end: Ending beta
        gamma: Gamma for rebalanced schedule
        device: Device to store tensors
    """
    
    def __init__(
        self,
        schedule_type: str = "cosine",
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        gamma: float = 1.0,
        device: str = "cpu"
    ):
        self.timesteps = timesteps
        self.schedule_type = schedule_type
        self.device = device
        
        # Get betas
        betas = get_beta_schedule(
            schedule_type, timesteps, beta_start, beta_end, gamma
        )
        
        # Compute alphas
        alphas, alphas_cumprod, alphas_cumprod_prev = compute_alpha_schedule(betas)
        
        # Store all precomputed values
        self.register("betas", betas)
        self.register("alphas", alphas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # For forward diffusion q(x_t | x_0)
        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.register("posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register("posterior_log_variance_clipped",
            torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        )
        self.register("posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register("posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        
        # For sampling
        self.register("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register("sqrt_recipm1_alphas_cumprod", 
            torch.sqrt(1.0 / alphas_cumprod - 1)
        )
    
    def register(self, name: str, tensor: torch.Tensor):
        """Register a buffer tensor."""
        setattr(self, name, tensor.to(self.device))
    
    def to(self, device: str):
        """Move all tensors to device."""
        self.device = device
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))
        return self
    
    def _extract(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        x_shape: Tuple
    ) -> torch.Tensor:
        """
        Extract values from a at indices t, reshape for broadcasting.
        
        Args:
            a: Tensor to index [timesteps]
            t: Timestep indices [batch_size]
            x_shape: Shape to broadcast to
            
        Returns:
            Extracted and reshaped values
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: sample x_t from q(x_t | x_0).
        
        x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        
        Args:
            x_start: Clean data x_0 [B, C, H, W]
            t: Timesteps [B]
            noise: Optional pre-sampled noise
            
        Returns:
            x_t: Noisy data
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0).
        
        Used for DDPM sampling when we have predicted x_0.
        
        Args:
            x_start: Predicted x_0
            x_t: Current noisy sample
            t: Current timestep
            
        Returns:
            posterior_mean: Mean of posterior
            posterior_variance: Variance of posterior
            posterior_log_variance: Log variance of posterior
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        
        x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
        
        Args:
            x_t: Noisy sample
            t: Timestep
            noise: Predicted noise
            
        Returns:
            Predicted x_0
        """
        return (
            self._extract(self.sqrt_recip_alphas, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get signal-to-noise ratio at timestep t.
        
        SNR(t) = ᾱ_t / (1 - ᾱ_t)
        
        Used for loss weighting and analysis.
        """
        alphas_cumprod_t = self._extract(
            self.alphas_cumprod, t, (t.shape[0], 1)
        ).squeeze()
        return alphas_cumprod_t / (1 - alphas_cumprod_t)


# Testing
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    timesteps = 1000
    
    # Compare schedules
    schedules = {
        'linear': linear_beta_schedule(timesteps),
        'cosine': cosine_beta_schedule(timesteps),
        'gamma=0.5': gamma_rebalanced_schedule(timesteps, gamma=0.5),
        'gamma=1.5': gamma_rebalanced_schedule(timesteps, gamma=1.5),
        'gamma=2.0': gamma_rebalanced_schedule(timesteps, gamma=2.0),
    }
    
    print("Beta schedule comparison:")
    for name, betas in schedules.items():
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        print(f"  {name}: β[0]={betas[0]:.6f}, β[-1]={betas[-1]:.4f}, "
              f"ᾱ[500]={alphas_cumprod[500]:.4f}")
    
    # Test NoiseSchedule class
    schedule = NoiseSchedule(
        schedule_type="cosine",
        timesteps=1000,
        device="cpu"
    )
    
    print(f"\nNoiseSchedule test:")
    print(f"  Timesteps: {schedule.timesteps}")
    print(f"  Betas shape: {schedule.betas.shape}")
    
    # Test forward diffusion
    x = torch.randn(4, 4, 32, 32)
    t = torch.tensor([0, 250, 500, 999])
    
    x_t, noise = schedule.q_sample(x, t)
    print(f"  Forward diffusion: x_t shape = {x_t.shape}")
    
    # Test SNR
    snr = schedule.get_snr(t)
    print(f"  SNR at t={t.tolist()}: {snr.tolist()}")
    
    print("\nNoise schedule tests passed!")
