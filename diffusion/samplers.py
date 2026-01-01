"""
SBLDM Samplers
DDPM and DDIM samplers for reverse diffusion + adaptive sampling.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple, List
from tqdm import tqdm

from .schedules import NoiseSchedule


class DDPMSampler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) sampler.
    
    Performs full reverse diffusion with stochastic sampling.
    Slower but produces high-quality samples.
    
    Args:
        schedule: Noise schedule object
        model: Score/noise prediction model
        clip_denoised: Whether to clip x_0 predictions
    """
    
    def __init__(
        self,
        schedule: NoiseSchedule,
        model: nn.Module,
        clip_denoised: bool = True,
        clip_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        self.schedule = schedule
        self.model = model
        self.clip_denoised = clip_denoised
        self.clip_range = clip_range
    
    @torch.no_grad()
    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance of p(x_{t-1} | x_t).
        
        Args:
            x: Current sample x_t
            t: Current timestep
            
        Returns:
            model_mean: Mean of reverse distribution
            posterior_variance: Variance
            posterior_log_variance: Log variance
        """
        # Predict noise
        noise_pred = self.model(x, t)
        
        # Predict x_0 from noise
        x_start = self.schedule.predict_start_from_noise(x, t, noise_pred)
        
        # Clip if requested
        if self.clip_denoised:
            x_start = torch.clamp(x_start, self.clip_range[0], self.clip_range[1])
        
        # Get posterior parameters
        model_mean, posterior_variance, posterior_log_variance = \
            self.schedule.q_posterior_mean_variance(x_start, x, t)
        
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from p(x_{t-1} | x_t).
        
        Args:
            x: Current sample x_t
            t: Current timestep (as tensor)
            
        Returns:
            Sample x_{t-1}
        """
        model_mean, _, posterior_log_variance = self.p_mean_variance(x, t)
        
        # Add noise (except at t=0)
        noise = torch.randn_like(x)
        nonzero_mask = (t > 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        sample = model_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise
        
        return sample
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        device: str = "cuda",
        return_intermediates: bool = False,
        progress: bool = True
    ) -> torch.Tensor:
        """
        Generate samples from noise.
        
        Args:
            shape: Output shape (B, C, H, W)
            device: Device to generate on
            return_intermediates: Whether to return intermediate samples
            progress: Show progress bar
            
        Returns:
            Generated samples
        """
        self.model.eval()
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        intermediates = [x] if return_intermediates else None
        
        # Reverse diffusion
        timesteps = list(range(self.schedule.timesteps))[::-1]
        
        if progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling")
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
            
            if return_intermediates and t % 100 == 0:
                intermediates.append(x)
        
        if return_intermediates:
            return x, intermediates
        return x


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) sampler.
    
    Deterministic or semi-stochastic sampling with fewer steps.
    Much faster than DDPM with comparable quality.
    
    Args:
        schedule: Noise schedule object
        model: Score/noise prediction model
        eta: Noise scale (0 = deterministic, 1 = DDPM-like)
        clip_denoised: Whether to clip predictions
    """
    
    def __init__(
        self,
        schedule: NoiseSchedule,
        model: nn.Module,
        eta: float = 0.0,
        clip_denoised: bool = True,
        clip_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        self.schedule = schedule
        self.model = model
        self.eta = eta
        self.clip_denoised = clip_denoised
        self.clip_range = clip_range
    
    def _get_ddim_timesteps(
        self,
        num_steps: int
    ) -> torch.Tensor:
        """
        Get DDIM timestep sequence.
        
        Evenly spaced subset of original timesteps.
        """
        c = self.schedule.timesteps // num_steps
        timesteps = torch.arange(0, self.schedule.timesteps, c)
        return timesteps.flip(0)  # Reverse order for sampling
    
    @torch.no_grad()
    def ddim_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Single DDIM sampling step.
        
        Args:
            x: Current sample x_t
            t: Current timestep
            t_prev: Previous timestep
            
        Returns:
            Sample x_{t_prev}
        """
        # Predict noise
        noise_pred = self.model(x, t)
        
        # Get schedule values
        alpha_t = self.schedule._extract(
            self.schedule.alphas_cumprod, t, x.shape
        )
        alpha_prev = self.schedule._extract(
            self.schedule.alphas_cumprod, t_prev, x.shape
        )
        
        # Predict x_0
        x_start = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        if self.clip_denoised:
            x_start = torch.clamp(x_start, self.clip_range[0], self.clip_range[1])
        
        # Compute variance
        sigma = self.eta * torch.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        )
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * noise_pred
        
        # Random noise (scaled by sigma)
        noise = torch.randn_like(x) if self.eta > 0 else 0
        
        # DDIM update
        x_prev = torch.sqrt(alpha_prev) * x_start + dir_xt + sigma * noise
        
        return x_prev
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        device: str = "cuda",
        return_intermediates: bool = False,
        progress: bool = True
    ) -> torch.Tensor:
        """
        Generate samples with DDIM.
        
        Args:
            shape: Output shape (B, C, H, W)
            num_steps: Number of sampling steps
            device: Device to generate on
            return_intermediates: Return intermediate samples
            progress: Show progress bar
            
        Returns:
            Generated samples
        """
        self.model.eval()
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Get DDIM timesteps
        timesteps = self._get_ddim_timesteps(num_steps)
        timesteps = timesteps.to(device)
        
        intermediates = [x] if return_intermediates else None
        
        if progress:
            pbar = tqdm(range(len(timesteps) - 1), desc="DDIM Sampling")
        else:
            pbar = range(len(timesteps) - 1)
        
        for i in pbar:
            t = timesteps[i].expand(shape[0])
            t_prev = timesteps[i + 1].expand(shape[0])
            
            x = self.ddim_step(x, t, t_prev)
            
            if return_intermediates:
                intermediates.append(x)
        
        # Final step to t=0
        t = timesteps[-1].expand(shape[0])
        t_prev = torch.zeros_like(t)
        x = self.ddim_step(x, t, t_prev)
        
        if return_intermediates:
            intermediates.append(x)
            return x, intermediates
        return x


class AdaptiveSampler:
    """
    Adaptive sampling with early stopping (Novel contribution).
    
    Monitors the score norm during sampling and stops early
    when the change becomes small enough.
    
    Hypothesis: For medical images, we can often stop earlier
    because the structure is simpler than natural images.
    
    Args:
        base_sampler: DDPM or DDIM sampler
        threshold: Score norm change threshold for early stop
        min_steps: Minimum steps before considering early stop
        window_size: Window for computing moving average of change
    """
    
    def __init__(
        self,
        base_sampler,
        threshold: float = 0.05,
        min_steps: int = 20,
        window_size: int = 5
    ):
        self.base_sampler = base_sampler
        self.threshold = threshold
        self.min_steps = min_steps
        self.window_size = window_size
    
    def _compute_score_norm(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> float:
        """Compute norm of predicted score/noise."""
        with torch.no_grad():
            noise_pred = model(x, t)
            return noise_pred.norm(dim=[1, 2, 3]).mean().item()
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        device: str = "cuda",
        progress: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate samples with adaptive early stopping.
        
        Args:
            shape: Output shape
            num_steps: Maximum number of steps
            device: Device
            progress: Show progress
            
        Returns:
            samples: Generated samples
            info: Dictionary with sampling statistics
        """
        model = self.base_sampler.model
        schedule = self.base_sampler.schedule
        model.eval()
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # Get timesteps (DDIM-style)
        if hasattr(self.base_sampler, '_get_ddim_timesteps'):
            timesteps = self.base_sampler._get_ddim_timesteps(num_steps).to(device)
        else:
            timesteps = torch.arange(schedule.timesteps - 1, -1, 
                                    -schedule.timesteps // num_steps, device=device)
        
        score_norms = []
        actual_steps = 0
        early_stopped = False
        
        if progress:
            pbar = tqdm(range(len(timesteps) - 1), desc="Adaptive Sampling")
        else:
            pbar = range(len(timesteps) - 1)
        
        for i in pbar:
            t = timesteps[i].expand(shape[0])
            t_prev = timesteps[min(i + 1, len(timesteps) - 1)].expand(shape[0])
            
            # Compute score norm before step
            score_norm = self._compute_score_norm(model, x, t)
            score_norms.append(score_norm)
            
            # Check early stopping condition
            if len(score_norms) >= self.window_size and actual_steps >= self.min_steps:
                recent = score_norms[-self.window_size:]
                change = abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-8)
                
                if change < self.threshold:
                    early_stopped = True
                    if progress:
                        pbar.set_postfix({'early_stop': True, 'change': f'{change:.4f}'})
                    break
            
            # Take sampling step
            if isinstance(self.base_sampler, DDIMSampler):
                x = self.base_sampler.ddim_step(x, t, t_prev)
            else:
                x = self.base_sampler.p_sample(x, t)
            
            actual_steps += 1
        
        info = {
            'actual_steps': actual_steps,
            'max_steps': num_steps,
            'early_stopped': early_stopped,
            'final_score_norm': score_norms[-1] if score_norms else 0,
            'score_norms': score_norms
        }
        
        return x, info


# Convenience function
def create_sampler(
    sampler_type: str,
    schedule: NoiseSchedule,
    model: nn.Module,
    **kwargs
):
    """
    Create a sampler by type.
    
    Args:
        sampler_type: "ddpm", "ddim", or "adaptive"
        schedule: Noise schedule
        model: Score model
        **kwargs: Additional arguments
        
    Returns:
        Sampler instance
    """
    if sampler_type == "ddpm":
        return DDPMSampler(schedule, model, **kwargs)
    elif sampler_type == "ddim":
        return DDIMSampler(
            schedule, model,
            eta=kwargs.get('eta', 0.0),
            clip_denoised=kwargs.get('clip_denoised', True)
        )
    elif sampler_type == "adaptive":
        base_sampler = DDIMSampler(
            schedule, model,
            eta=kwargs.get('eta', 0.0)
        )
        return AdaptiveSampler(
            base_sampler,
            threshold=kwargs.get('threshold', 0.05),
            min_steps=kwargs.get('min_steps', 20)
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


# Testing
if __name__ == '__main__':
    from .schedules import NoiseSchedule
    
    print("Testing samplers...")
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 3, padding=1)
        
        def forward(self, x, t):
            return self.conv(x)
    
    model = DummyModel()
    schedule = NoiseSchedule(timesteps=100)
    
    # Test DDPM sampler
    ddpm = DDPMSampler(schedule, model)
    samples = ddpm.sample((2, 4, 8, 8), device="cpu", progress=False)
    print(f"DDPM samples shape: {samples.shape}")
    
    # Test DDIM sampler  
    ddim = DDIMSampler(schedule, model, eta=0.0)
    samples = ddim.sample((2, 4, 8, 8), num_steps=10, device="cpu", progress=False)
    print(f"DDIM samples shape: {samples.shape}")
    
    # Test adaptive sampler
    adaptive = AdaptiveSampler(ddim, threshold=0.5, min_steps=5)
    samples, info = adaptive.sample((2, 4, 8, 8), num_steps=20, device="cpu", progress=False)
    print(f"Adaptive samples shape: {samples.shape}")
    print(f"Adaptive info: steps={info['actual_steps']}, early_stopped={info['early_stopped']}")
    
    print("\nSampler tests passed!")
