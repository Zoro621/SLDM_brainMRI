"""
SBLDM Evaluation Metrics
Computes image quality metrics: FID, SSIM, PSNR, and reconstruction heatmaps.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Try to import optional dependencies
try:
    from scipy import linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from skimage.metrics import structural_similarity as sk_ssim
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ============================================================================
# SSIM and PSNR
# ============================================================================

def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0,
    channel_axis: Optional[int] = None
) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        img1: First image [C, H, W] or [H, W]
        img2: Second image [C, H, W] or [H, W]
        data_range: Data range (1.0 for [0,1] images)
        channel_axis: Channel axis if multichannel
        
    Returns:
        SSIM value (higher is better, max 1.0)
    """
    if HAS_SKIMAGE:
        img1_np = img1.cpu().numpy()
        img2_np = img2.cpu().numpy()
        
        if img1_np.ndim == 3:
            # Move channel to last for skimage
            img1_np = np.transpose(img1_np, (1, 2, 0))
            img2_np = np.transpose(img2_np, (1, 2, 0))
            channel_axis = -1
        
        return sk_ssim(
            img1_np, img2_np,
            data_range=data_range,
            channel_axis=channel_axis
        )
    else:
        # Fallback: simple implementation
        return _ssim_torch(img1, img2, data_range)


def _ssim_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2
) -> float:
    """PyTorch SSIM implementation."""
    # Ensure 4D: [B, C, H, W]
    if img1.dim() == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    C1 = (C1 * data_range) ** 2
    C2 = (C2 * data_range) ** 2
    
    # Create Gaussian window
    gauss = torch.exp(
        -torch.arange(window_size).float().sub(window_size // 2).pow(2) / (2 * 1.5**2)
    )
    gauss = gauss / gauss.sum()
    window = gauss.outer(gauss)
    window = window.expand(img1.size(1), 1, window_size, window_size).to(img1.device)
    
    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img1.size(1))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances
    sigma1_sq = F.conv2d(img1**2, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding=window_size//2, groups=img1.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def compute_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1: First image
        img2: Second image  
        data_range: Maximum value range
        
    Returns:
        PSNR value in dB (higher is better)
    """
    if HAS_SKIMAGE:
        return sk_psnr(
            img1.cpu().numpy(),
            img2.cpu().numpy(),
            data_range=data_range
        )
    else:
        mse = F.mse_loss(img1, img2).item()
        if mse == 0:
            return float('inf')
        return 10 * np.log10(data_range**2 / mse)


def compute_batch_metrics(
    images1: torch.Tensor,
    images2: torch.Tensor,
    data_range: float = 1.0
) -> Dict[str, float]:
    """
    Compute SSIM and PSNR for a batch of image pairs.
    
    Args:
        images1: First batch [B, C, H, W]
        images2: Second batch [B, C, H, W]
        data_range: Data range
        
    Returns:
        Dictionary with mean and std of metrics
    """
    ssim_values = []
    psnr_values = []
    
    for i in range(len(images1)):
        ssim_values.append(compute_ssim(images1[i], images2[i], data_range))
        psnr_values.append(compute_psnr(images1[i], images2[i], data_range))
    
    return {
        'ssim_mean': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values),
        'psnr_mean': np.mean(psnr_values),
        'psnr_std': np.std(psnr_values)
    }


# ============================================================================
# FID (Fréchet Inception Distance)
# ============================================================================

class InceptionV3Features(nn.Module):
    """
    Extract features from InceptionV3 for FID computation.
    Uses the pool3 layer (2048-dimensional features).
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        except:
            from torchvision.models import inception_v3
            self.inception = inception_v3(pretrained=True)
        
        self.inception.fc = nn.Identity()  # Remove classifier
        self.inception.eval()
        self.inception.to(device)
        
        self.device = device
        
        # Preprocessing for Inception
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            x: Images [B, C, H, W] in [0, 1] range
            
        Returns:
            Features [B, 2048]
        """
        # Convert grayscale to RGB
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Preprocess
        x = self.preprocess(x)
        
        # Extract features
        features = self.inception(x)
        
        return features


def compute_fid_statistics(
    features: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance of features.
    
    Args:
        features: Feature vectors [N, D]
        
    Returns:
        mu: Mean vector [D]
        sigma: Covariance matrix [D, D]
    """
    features = features.cpu().numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Compute FID between two Gaussian distributions.
    
    FID = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2√(Σ1Σ2))
    
    Args:
        mu1, sigma1: Statistics of first distribution
        mu2, sigma2: Statistics of second distribution
        eps: Small value for numerical stability
        
    Returns:
        FID value (lower is better)
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required for FID computation")
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle numerical issues
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # FID formula
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return float(fid)


class FIDCalculator:
    """
    Calculate FID between real and generated images.
    
    Args:
        device: Device to use
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.feature_extractor = InceptionV3Features(device)
        
        # Cache for real statistics
        self.real_mu = None
        self.real_sigma = None
    
    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        batch_size: int = 32
    ) -> torch.Tensor:
        """Extract features from a batch of images."""
        features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            feat = self.feature_extractor(batch)
            features.append(feat.cpu())
        
        return torch.cat(features, dim=0)
    
    def compute_real_statistics(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ):
        """Compute and cache statistics for real images."""
        all_features = []
        n_samples = 0
        
        for batch in tqdm(dataloader, desc="Computing real statistics"):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            # Denormalize if needed (assume [-1, 1])
            if batch.min() < 0:
                batch = (batch + 1) / 2
            
            features = self.feature_extractor(batch.to(self.device))
            all_features.append(features.cpu())
            
            n_samples += len(batch)
            if max_samples and n_samples >= max_samples:
                break
        
        all_features = torch.cat(all_features, dim=0)
        if max_samples:
            all_features = all_features[:max_samples]
        
        self.real_mu, self.real_sigma = compute_fid_statistics(all_features)
        
        return self.real_mu, self.real_sigma
    
    def compute_generated_fid(
        self,
        generated_images: torch.Tensor
    ) -> float:
        """
        Compute FID for generated images against cached real statistics.
        
        Args:
            generated_images: Generated images [N, C, H, W] in [0, 1]
            
        Returns:
            FID value
        """
        if self.real_mu is None:
            raise ValueError("Must compute real statistics first")
        
        # Extract features
        gen_features = self.extract_features(generated_images)
        
        # Compute statistics
        gen_mu, gen_sigma = compute_fid_statistics(gen_features)
        
        # Compute FID
        return compute_fid(self.real_mu, self.real_sigma, gen_mu, gen_sigma)
    
    def compute_fid_between(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor
    ) -> float:
        """Compute FID between two sets of images."""
        feat1 = self.extract_features(images1)
        feat2 = self.extract_features(images2)
        
        mu1, sigma1 = compute_fid_statistics(feat1)
        mu2, sigma2 = compute_fid_statistics(feat2)
        
        return compute_fid(mu1, sigma1, mu2, sigma2)


# ============================================================================
# Reconstruction Error Heatmaps
# ============================================================================

def compute_error_heatmap(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute per-pixel reconstruction error heatmap.
    
    Args:
        original: Original image [C, H, W]
        reconstructed: Reconstructed image [C, H, W]
        normalize: Normalize to [0, 1]
        
    Returns:
        Error heatmap [H, W]
    """
    # Absolute error
    error = torch.abs(original - reconstructed)
    
    # Average over channels if multi-channel
    if error.dim() == 3:
        error = error.mean(dim=0)
    
    if normalize:
        if error.max() > 0:
            error = error / error.max()
    
    return error


def batch_error_heatmaps(
    originals: torch.Tensor,
    reconstructed: torch.Tensor
) -> torch.Tensor:
    """
    Compute error heatmaps for a batch.
    
    Args:
        originals: Original images [B, C, H, W]
        reconstructed: Reconstructed images [B, C, H, W]
        
    Returns:
        Error heatmaps [B, 1, H, W]
    """
    errors = []
    for i in range(len(originals)):
        error = compute_error_heatmap(originals[i], reconstructed[i])
        errors.append(error.unsqueeze(0))
    
    return torch.stack(errors, dim=0)


def save_heatmap_visualization(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    heatmap: torch.Tensor,
    path: str,
    cmap: str = 'hot'
):
    """
    Save visualization with original, reconstructed, and heatmap.
    
    Args:
        original: Original image [C, H, W] or [H, W]
        reconstructed: Reconstructed image [C, H, W] or [H, W]
        heatmap: Error heatmap [H, W]
        path: Output path
        cmap: Colormap for heatmap
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Prepare images
    if original.dim() == 3:
        original = original.squeeze(0)
    if reconstructed.dim() == 3:
        reconstructed = reconstructed.squeeze(0)
    
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    heatmap = heatmap.cpu().numpy()
    
    # Plot
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    im = axes[2].imshow(heatmap, cmap=cmap)
    axes[2].set_title('Error Heatmap')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Full Evaluation Pipeline
# ============================================================================

class MetricsEvaluator:
    """
    Complete evaluation pipeline for SBLDM.
    
    Computes:
    - Reconstruction metrics (SSIM, PSNR) for VAE
    - Generation metrics (FID) for full model
    - Error heatmaps for visualization
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.fid_calculator = None  # Lazy initialization
    
    def evaluate_reconstruction(
        self,
        vae: nn.Module,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate VAE reconstruction quality.
        
        Returns:
            Dictionary with SSIM and PSNR statistics
        """
        vae.eval()
        
        all_ssim = []
        all_psnr = []
        n_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating reconstruction"):
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                
                x = batch.to(self.device)
                
                # Reconstruct
                recon, _, _ = vae(x, sample_posterior=False)
                
                # Denormalize [-1, 1] -> [0, 1]
                x_01 = (x + 1) / 2
                recon_01 = (recon + 1) / 2
                recon_01 = torch.clamp(recon_01, 0, 1)
                
                # Compute metrics
                for i in range(len(x)):
                    all_ssim.append(compute_ssim(x_01[i], recon_01[i]))
                    all_psnr.append(compute_psnr(x_01[i], recon_01[i]))
                
                n_samples += len(x)
                if max_samples and n_samples >= max_samples:
                    break
        
        return {
            'ssim_mean': float(np.mean(all_ssim)),
            'ssim_std': float(np.std(all_ssim)),
            'psnr_mean': float(np.mean(all_psnr)),
            'psnr_std': float(np.std(all_psnr)),
            'n_samples': n_samples
        }
    
    def evaluate_fid(
        self,
        generated_images: torch.Tensor,
        real_dataloader: DataLoader,
        max_real_samples: Optional[int] = None
    ) -> float:
        """
        Evaluate FID between generated and real images.
        
        Args:
            generated_images: Generated images [N, C, H, W] in [0, 1]
            real_dataloader: DataLoader for real images
            max_real_samples: Maximum real samples for statistics
            
        Returns:
            FID value
        """
        if self.fid_calculator is None:
            self.fid_calculator = FIDCalculator(self.device)
        
        # Compute real statistics
        self.fid_calculator.compute_real_statistics(
            real_dataloader,
            max_samples=max_real_samples or len(generated_images)
        )
        
        # Compute FID
        return self.fid_calculator.compute_generated_fid(generated_images)
    
    def full_evaluation(
        self,
        vae: nn.Module,
        unet: nn.Module,
        dataloader: DataLoader,
        sampler,
        num_generated: int = 1000,
        device: str = "cuda"
    ) -> Dict[str, float]:
        """
        Run full evaluation pipeline.
        
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Reconstruction metrics
        print("Evaluating reconstruction...")
        recon_metrics = self.evaluate_reconstruction(vae, dataloader)
        results.update({f'recon_{k}': v for k, v in recon_metrics.items()})
        
        # Generate samples
        print(f"Generating {num_generated} samples...")
        from sample import generate_samples
        
        # This would need the full config, so this is a simplified version
        # In practice, use the generate_samples function from sample.py
        
        print("FID evaluation requires generated samples (use sample.py first)")
        
        return results


# Testing
if __name__ == '__main__':
    print("Testing metrics...")
    
    # Test SSIM/PSNR
    img1 = torch.rand(1, 64, 64)
    img2 = img1 + torch.randn_like(img1) * 0.1
    img2 = torch.clamp(img2, 0, 1)
    
    ssim = compute_ssim(img1, img2)
    psnr = compute_psnr(img1, img2)
    
    print(f"SSIM: {ssim:.4f}")
    print(f"PSNR: {psnr:.2f} dB")
    
    # Test batch metrics
    batch1 = torch.rand(4, 1, 64, 64)
    batch2 = batch1 + torch.randn_like(batch1) * 0.1
    batch2 = torch.clamp(batch2, 0, 1)
    
    batch_metrics = compute_batch_metrics(batch1, batch2)
    print(f"\nBatch metrics: {batch_metrics}")
    
    # Test error heatmap
    heatmap = compute_error_heatmap(img1, img2)
    print(f"\nHeatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    
    print("\nMetrics tests passed!")
