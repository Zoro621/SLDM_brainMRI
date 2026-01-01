"""
SBLDM Smoke Tests
Quick validation that all components work without OOM on 16GB GPU.

Usage:
    python tests/smoke_test.py --device cuda
"""

import sys
import os
import argparse
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  GPU Memory: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")


def test_data_loading():
    """Test dataset and dataloader."""
    print("\n" + "="*60)
    print("TEST: Data Loading")
    print("="*60)
    
    from data.preprocess import create_sample_dataset
    from data.dataset import get_dataloaders
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic dataset
        print("Creating synthetic dataset...")
        stats = create_sample_dataset(tmpdir, num_samples=100, target_size=(128, 128))
        print(f"  Created {stats['total_images']} images")
        
        # Test dataloader
        print("Testing dataloader...")
        train_loader, val_loader, test_loader = get_dataloaders(
            tmpdir,
            resolution=128,
            batch_size=16,
            num_workers=0  # Use 0 for testing
        )
        
        batch = next(iter(train_loader))
        print(f"  Batch shape: {batch.shape}")
        print(f"  Batch range: [{batch.min():.2f}, {batch.max():.2f}]")
        
        assert batch.shape == (16, 1, 128, 128), "Wrong batch shape"
        print("  ✓ Data loading passed")
        
        return tmpdir, train_loader, val_loader


def test_vae(device, data_dir):
    """Test VAE forward/backward pass."""
    print("\n" + "="*60)
    print("TEST: VAE")
    print("="*60)
    
    from models.vae import VAE
    from data.dataset import get_dataloaders
    
    # Create model
    vae = VAE(
        in_channels=1,
        latent_channels=4,
        hidden_dims=[32, 64, 128, 256]
    ).to(device)
    
    print(f"  VAE parameters: {vae.get_num_params():,}")
    print_gpu_memory()
    
    # Get data
    train_loader, _, _ = get_dataloaders(
        data_dir, resolution=128, batch_size=16, num_workers=0
    )
    
    batch = next(iter(train_loader)).to(device)
    
    # Forward pass
    print("Testing forward pass...")
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    with torch.cuda.amp.autocast():
        recon, mean, logvar = vae(batch)
        loss, loss_dict = vae.loss_function(batch, recon, mean, logvar)
    
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Latent shape: {mean.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print_gpu_memory()
    
    # Backward pass
    print("Testing backward pass...")
    loss.backward()
    
    # Check gradients
    grad_norm = sum(p.grad.norm().item() for p in vae.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print_gpu_memory()
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak memory: {peak_mem:.1f} MB")
        assert peak_mem < 8000, f"VAE uses too much memory: {peak_mem} MB"
    
    print("  ✓ VAE test passed")
    return vae


def test_unet(device):
    """Test UNet forward/backward pass."""
    print("\n" + "="*60)
    print("TEST: UNet")
    print("="*60)
    
    from models.unet_score import UNetModel
    
    # Create model
    unet = UNetModel(
        in_channels=4,
        out_channels=4,
        model_channels=64,
        channel_mult=[1, 2, 4],
        num_res_blocks=2,
        attention_resolutions=[8]
    ).to(device)
    
    print(f"  UNet parameters: {unet.get_num_params():,}")
    print_gpu_memory()
    
    # Create dummy input (latent size for 128x128 image with 4x downsampling)
    latent_size = 128 // 4  # = 32
    x = torch.randn(8, 4, latent_size, latent_size, device=device)
    t = torch.randint(0, 1000, (8,), device=device)
    
    # Forward pass
    print("Testing forward pass...")
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    with torch.cuda.amp.autocast():
        noise_pred = unet(x, t)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {noise_pred.shape}")
    print_gpu_memory()
    
    # Backward pass
    print("Testing backward pass...")
    loss = noise_pred.mean()
    loss.backward()
    
    grad_norm = sum(p.grad.norm().item() for p in unet.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak memory: {peak_mem:.1f} MB")
        assert peak_mem < 8000, f"UNet uses too much memory: {peak_mem} MB"
    
    print("  ✓ UNet test passed")
    return unet


def test_diffusion_step(device, unet):
    """Test one diffusion training step."""
    print("\n" + "="*60)
    print("TEST: Diffusion Training Step")
    print("="*60)
    
    from diffusion.schedules import NoiseSchedule
    from diffusion.loss import DiffusionLoss, FrequencyAwareLoss, LatentCutMix
    
    # Create schedule
    schedule = NoiseSchedule(
        schedule_type="cosine",
        timesteps=1000,
        device=device
    )
    
    # Create loss function
    loss_fn = FrequencyAwareLoss(freq_weight=0.1)
    cutmix = LatentCutMix(prob=0.5)
    
    # Dummy latents
    z = torch.randn(8, 4, 32, 32, device=device)
    
    # Sample timesteps
    t = torch.randint(0, 1000, (8,), device=device)
    
    # Sample noise
    noise = torch.randn_like(z)
    
    # Apply CutMix
    z_mixed, noise_mixed = cutmix(z, noise)
    
    # Forward diffusion
    z_noisy, noise_target = schedule.q_sample(z_mixed, t, noise_mixed)
    
    # Forward pass
    print("Testing diffusion step...")
    unet.train()
    unet.zero_grad()
    
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    with torch.cuda.amp.autocast():
        noise_pred = unet(z_noisy, t)
        loss, loss_dict = loss_fn(noise_pred, noise_target)
    
    print(f"  Noisy latent shape: {z_noisy.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss components: {loss_dict}")
    print_gpu_memory()
    
    # Backward
    loss.backward()
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak memory: {peak_mem:.1f} MB")
    
    print("  ✓ Diffusion step test passed")


def test_sampling(device, unet):
    """Test DDIM sampling."""
    print("\n" + "="*60)
    print("TEST: DDIM Sampling")
    print("="*60)
    
    from diffusion.schedules import NoiseSchedule
    from diffusion.samplers import DDIMSampler
    
    schedule = NoiseSchedule(
        schedule_type="cosine",
        timesteps=1000,
        device=device
    )
    
    sampler = DDIMSampler(schedule, unet, eta=0.0)
    
    print("Testing DDIM sampling (4 samples, 20 steps)...")
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    
    unet.eval()
    with torch.no_grad():
        samples = sampler.sample(
            shape=(4, 4, 32, 32),
            num_steps=20,
            device=device,
            progress=True
        )
    
    print(f"  Sample shape: {samples.shape}")
    print(f"  Sample range: [{samples.min():.2f}, {samples.max():.2f}]")
    
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak memory: {peak_mem:.1f} MB")
    
    print("  ✓ Sampling test passed")


def test_metrics():
    """Test evaluation metrics."""
    print("\n" + "="*60)
    print("TEST: Metrics")
    print("="*60)
    
    from eval.metrics import compute_ssim, compute_psnr, compute_error_heatmap
    
    # Create test images
    img1 = torch.rand(1, 64, 64)
    img2 = img1 + torch.randn_like(img1) * 0.1
    img2 = torch.clamp(img2, 0, 1)
    
    print("Testing SSIM/PSNR...")
    ssim = compute_ssim(img1, img2)
    psnr = compute_psnr(img1, img2)
    
    print(f"  SSIM: {ssim:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    
    assert 0 < ssim <= 1, "Invalid SSIM"
    assert psnr > 0, "Invalid PSNR"
    
    print("Testing error heatmap...")
    heatmap = compute_error_heatmap(img1, img2)
    print(f"  Heatmap shape: {heatmap.shape}")
    
    print("  ✓ Metrics test passed")


def run_all_tests(device):
    """Run all smoke tests."""
    print("\n" + "="*60)
    print("SBLDM SMOKE TESTS")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*60)
    
    # Create temp data
    from data.preprocess import create_sample_dataset
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create data
        create_sample_dataset(tmpdir, num_samples=100, target_size=(128, 128))
        
        # Run tests
        test_data_loading()
        vae = test_vae(device, tmpdir)
        unet = test_unet(device)
        test_diffusion_step(device, unet)
        test_sampling(device, unet)
        test_metrics()
    
    print("\n" + "="*60)
    print("ALL SMOKE TESTS PASSED! ✓")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='SBLDM Smoke Tests')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    run_all_tests(device)


if __name__ == '__main__':
    main()
