"""
Comprehensive Visualization Suite for SBLDM Medical Image Synthesis
Generates publication-quality figures for training analysis and evaluation.
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import seaborn as sns
from scipy import stats
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae import create_vae
from models.simple_unet import create_unet
from diffusion.schedules import NoiseSchedule
from diffusion.samplers import DDIMSampler
import yaml

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def load_training_history(log_dir):
    """Load training history from JSON files."""
    vae_path = Path(log_dir) / 'vae_history.json'
    diff_path = Path(log_dir) / 'diffusion_history.json'
    
    vae_history = None
    diff_history = None
    
    if vae_path.exists():
        with open(vae_path) as f:
            vae_history = json.load(f)
    
    if diff_path.exists():
        with open(diff_path) as f:
            diff_history = json.load(f)
    
    return vae_history, diff_history


def plot_training_curves(vae_history, diff_history, save_dir):
    """Generate training curves for VAE and Diffusion models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # VAE Training Loss
    if vae_history:
        ax = axes[0, 0]
        epochs = range(1, len(vae_history['train_loss']) + 1)
        ax.plot(epochs, vae_history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        if 'val_loss' in vae_history:
            ax.plot(epochs, vae_history['val_loss'], 'r-', linewidth=2, label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('VAE Training Curves', fontweight='bold')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add best epoch marker
        if 'val_loss' in vae_history:
            best_epoch = np.argmin(vae_history['val_loss']) + 1
            best_val = min(vae_history['val_loss'])
            ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
            ax.scatter([best_epoch], [best_val], color='green', s=100, zorder=5, marker='*')
    
    # VAE Loss Components
    if vae_history:
        ax = axes[0, 1]
        epochs = range(1, len(vae_history['train_loss']) + 1)
        
        if 'recon_loss' in vae_history:
            ax.plot(epochs, vae_history['recon_loss'], 'b-', linewidth=2, label='Reconstruction')
        if 'kl_loss' in vae_history:
            ax.plot(epochs, vae_history['kl_loss'], 'orange', linewidth=2, label='KL Divergence')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Component')
        ax.set_title('VAE Loss Components', fontweight='bold')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Diffusion Training Loss
    if diff_history:
        ax = axes[1, 0]
        steps = range(1, len(diff_history['train_loss']) + 1)
        # Smooth the loss curve
        window = min(50, len(diff_history['train_loss']) // 10)
        if window > 1:
            smoothed = np.convolve(diff_history['train_loss'], np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(diff_history['train_loss']) + 1), smoothed, 'b-', linewidth=2, label='Train Loss (smoothed)')
            ax.plot(steps, diff_history['train_loss'], 'b-', alpha=0.2, linewidth=1)
        else:
            ax.plot(steps, diff_history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        
        if 'val_loss' in diff_history and diff_history['val_loss']:
            val_steps = np.linspace(1, len(diff_history['train_loss']), len(diff_history['val_loss']))
            ax.plot(val_steps, diff_history['val_loss'], 'r-', linewidth=2, label='Val Loss', marker='o', markersize=4)
        
        ax.set_xlabel('Training Step (x100)')
        ax.set_ylabel('Loss')
        ax.set_title('Diffusion Model Training Curves', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Diffusion Loss Breakdown (if available)
    if diff_history:
        ax = axes[1, 1]
        steps = range(1, len(diff_history['train_loss']) + 1)
        
        has_components = False
        if 'mse_loss' in diff_history and diff_history['mse_loss']:
            ax.plot(steps[:len(diff_history['mse_loss'])], diff_history['mse_loss'], 
                   'b-', linewidth=2, label='MSE Loss')
            has_components = True
        if 'freq_loss' in diff_history and diff_history['freq_loss']:
            ax.plot(steps[:len(diff_history['freq_loss'])], diff_history['freq_loss'], 
                   'orange', linewidth=2, label='Frequency Loss')
            has_components = True
        
        if not has_components:
            # Show loss distribution over time
            train_loss = diff_history['train_loss']
            n_bins = 5
            bin_size = len(train_loss) // n_bins
            positions = []
            data = []
            for i in range(n_bins):
                start = i * bin_size
                end = start + bin_size if i < n_bins - 1 else len(train_loss)
                data.append(train_loss[start:end])
                positions.append((start + end) / 2)
            
            bp = ax.boxplot(data, positions=positions, widths=bin_size * 0.6)
            ax.set_xlabel('Training Step (x100)')
            ax.set_ylabel('Loss Distribution')
            ax.set_title('Loss Distribution Over Training', fontweight='bold')
        else:
            ax.set_xlabel('Training Step (x100)')
            ax.set_ylabel('Loss Component')
            ax.set_title('Diffusion Loss Components', fontweight='bold')
            ax.legend()
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png')
    plt.savefig(save_dir / 'training_curves.pdf')
    plt.close()
    print(f"Saved: training_curves.png/pdf")


def plot_sample_grid(generated_images, real_images, save_dir, n_samples=16):
    """Create side-by-side comparison of generated and real images."""
    n_cols = 8
    n_rows = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    fig.suptitle('Generated vs Real Brain MRI Samples', fontsize=16, fontweight='bold', y=1.02)
    
    # Top two rows: Generated
    for i in range(min(n_samples, len(generated_images))):
        row = i // n_cols
        col = i % n_cols
        if row < 2:
            ax = axes[row, col]
            img = generated_images[i].squeeze().cpu().numpy()
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if col == 0 and row == 0:
                ax.set_ylabel('Generated', fontsize=12, fontweight='bold')
    
    # Bottom two rows: Real
    for i in range(min(n_samples, len(real_images))):
        row = 2 + i // n_cols
        col = i % n_cols
        if row < 4:
            ax = axes[row, col]
            img = real_images[i].squeeze().cpu().numpy()
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if col == 0 and row == 2:
                ax.set_ylabel('Real', fontsize=12, fontweight='bold')
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Generated', transform=axes[0, 0].transAxes, 
                     fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)
    axes[2, 0].text(-0.1, 0.5, 'Real', transform=axes[2, 0].transAxes,
                     fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sample_comparison.png')
    plt.savefig(save_dir / 'sample_comparison.pdf')
    plt.close()
    print(f"Saved: sample_comparison.png/pdf")


def plot_reconstruction_quality(vae, real_images, device, save_dir, n_samples=8):
    """Show original, reconstructed, and error maps."""
    vae.eval()
    
    fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))
    fig.suptitle('VAE Reconstruction Quality Analysis', fontsize=14, fontweight='bold', y=1.02)
    
    with torch.no_grad():
        for i in range(min(n_samples, len(real_images))):
            # Original
            original = real_images[i:i+1].to(device)
            if original.shape[1] == 1:
                original_input = original * 2 - 1  # [0,1] -> [-1,1]
            else:
                original_input = original
            
            # Reconstruct
            recon, _, _ = vae(original_input)
            recon = torch.clamp((recon + 1) / 2, 0, 1)  # [-1,1] -> [0,1]
            
            # Error map
            error = torch.abs(real_images[i:i+1] - recon.cpu())
            
            # Plot original
            axes[0, i].imshow(real_images[i].squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Original', fontsize=12, fontweight='bold')
            
            # Plot reconstruction
            axes[1, i].imshow(recon.cpu().squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Reconstructed', fontsize=12, fontweight='bold')
            
            # Plot error map
            im = axes[2, i].imshow(error.squeeze().numpy(), cmap='hot', vmin=0, vmax=0.3)
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_ylabel('Error', fontsize=12, fontweight='bold')
    
    # Add colorbar for error maps
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.2])
    fig.colorbar(im, cax=cbar_ax, label='Absolute Error')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(save_dir / 'reconstruction_quality.png')
    plt.savefig(save_dir / 'reconstruction_quality.pdf')
    plt.close()
    print(f"Saved: reconstruction_quality.png/pdf")


def plot_denoising_process(sampler, vae, device, save_dir, latent_size=32):
    """Visualize the denoising process at different timesteps."""
    timesteps_to_show = [1000, 800, 600, 400, 200, 100, 50, 0]
    
    fig, axes = plt.subplots(2, len(timesteps_to_show), figsize=(2*len(timesteps_to_show), 5))
    fig.suptitle('Diffusion Denoising Process', fontsize=14, fontweight='bold', y=1.02)
    
    with torch.no_grad():
        # Sample at different stages
        shape = (1, vae.latent_channels, latent_size, latent_size)
        
        # Get denoising trajectory
        z = torch.randn(shape, device=device)
        
        for idx, (ax_latent, ax_image) in enumerate(zip(axes[0], axes[1])):
            t = timesteps_to_show[idx]
            
            if t == 1000:
                z_t = z.clone()
            elif t == 0:
                z_t = sampler.sample(shape, num_steps=50, device=device, progress=False)
            else:
                # Approximate intermediate state
                n_steps = int(50 * (1 - t/1000))
                z_t = sampler.sample(shape, num_steps=max(1, n_steps), device=device, progress=False)
            
            # Show latent (first channel)
            latent_vis = z_t[0, 0].cpu().numpy()
            ax_latent.imshow(latent_vis, cmap='coolwarm', vmin=-3, vmax=3)
            ax_latent.set_title(f't={t}', fontsize=10)
            ax_latent.axis('off')
            if idx == 0:
                ax_latent.set_ylabel('Latent', fontsize=11, fontweight='bold')
            
            # Decode to image
            image = vae.decode(z_t)
            image = torch.clamp((image + 1) / 2, 0, 1)
            ax_image.imshow(image[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            ax_image.axis('off')
            if idx == 0:
                ax_image.set_ylabel('Decoded', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'denoising_process.png')
    plt.savefig(save_dir / 'denoising_process.pdf')
    plt.close()
    print(f"Saved: denoising_process.png/pdf")


def plot_latent_space_analysis(vae, real_images, device, save_dir):
    """Analyze the latent space distribution."""
    vae.eval()
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    latents_mu = []
    latents_std = []
    
    with torch.no_grad():
        for i in range(min(100, len(real_images))):
            img = real_images[i:i+1].to(device)
            img = img * 2 - 1  # [0,1] -> [-1,1]
            mu, logvar = vae.encode(img)
            latents_mu.append(mu.cpu())
            latents_std.append(torch.exp(0.5 * logvar).cpu())
    
    latents_mu = torch.cat(latents_mu, dim=0)
    latents_std = torch.cat(latents_std, dim=0)
    
    # 1. Distribution of mean values
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(latents_mu.flatten().numpy(), bins=50, density=True, alpha=0.7, color='blue')
    
    # Overlay standard normal
    x = np.linspace(-4, 4, 100)
    ax1.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')
    ax1.set_xlabel('Latent Value (μ)')
    ax1.set_ylabel('Density')
    ax1.set_title('Latent Mean Distribution', fontweight='bold')
    ax1.legend()
    ax1.set_xlim(-4, 4)
    
    # 2. Distribution of std values
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(latents_std.flatten().numpy(), bins=50, density=True, alpha=0.7, color='green')
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Target σ=1')
    ax2.set_xlabel('Latent Value (σ)')
    ax2.set_ylabel('Density')
    ax2.set_title('Latent Std Distribution', fontweight='bold')
    ax2.legend()
    
    # 3. Per-channel statistics
    ax3 = fig.add_subplot(gs[0, 2])
    channel_means = latents_mu.mean(dim=(0, 2, 3)).numpy()
    channel_stds = latents_mu.std(dim=(0, 2, 3)).numpy()
    x_pos = np.arange(len(channel_means))
    ax3.bar(x_pos, channel_means, yerr=channel_stds, capsize=5, alpha=0.7)
    ax3.set_xlabel('Latent Channel')
    ax3.set_ylabel('Mean ± Std')
    ax3.set_title('Per-Channel Statistics', fontweight='bold')
    ax3.set_xticks(x_pos)
    
    # 4. 2D projection (first two channels, averaged spatially)
    ax4 = fig.add_subplot(gs[1, 0])
    z1 = latents_mu[:, 0].mean(dim=(1, 2)).numpy()
    z2 = latents_mu[:, 1].mean(dim=(1, 2)).numpy()
    ax4.scatter(z1, z2, alpha=0.5, s=20)
    ax4.set_xlabel('Latent Channel 0 (mean)')
    ax4.set_ylabel('Latent Channel 1 (mean)')
    ax4.set_title('2D Latent Projection', fontweight='bold')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # 5. Spatial variance map
    ax5 = fig.add_subplot(gs[1, 1])
    spatial_var = latents_mu.var(dim=0).mean(dim=0).numpy()
    im = ax5.imshow(spatial_var, cmap='viridis')
    ax5.set_title('Spatial Variance Map', fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im, ax=ax5, fraction=0.046)
    
    # 6. KL divergence per channel
    ax6 = fig.add_subplot(gs[1, 2])
    kl_per_channel = []
    for c in range(latents_mu.shape[1]):
        mu_c = latents_mu[:, c]
        std_c = latents_std[:, c]
        kl = 0.5 * (mu_c**2 + std_c**2 - 1 - 2*torch.log(std_c)).mean()
        kl_per_channel.append(kl.item())
    
    ax6.bar(range(len(kl_per_channel)), kl_per_channel, alpha=0.7, color='purple')
    ax6.set_xlabel('Latent Channel')
    ax6.set_ylabel('KL Divergence')
    ax6.set_title('KL Divergence per Channel', fontweight='bold')
    ax6.set_xticks(range(len(kl_per_channel)))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'latent_space_analysis.png')
    plt.savefig(save_dir / 'latent_space_analysis.pdf')
    plt.close()
    print(f"Saved: latent_space_analysis.png/pdf")


def plot_metrics_summary(metrics, save_dir):
    """Create a summary visualization of evaluation metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Evaluation Metrics Summary', fontsize=14, fontweight='bold', y=1.05)
    
    # SSIM gauge
    ax = axes[0]
    ssim_val = metrics.get('ssim', 0)
    colors = ['#ff4444', '#ffaa00', '#44aa44']
    thresholds = [0.3, 0.6, 1.0]
    
    wedges = ax.pie([ssim_val, 1-ssim_val], colors=[colors[1] if ssim_val < 0.6 else colors[2], '#eeeeee'],
                     startangle=90, counterclock=False)
    centre_circle = plt.Circle((0, 0), 0.5, fc='white')
    ax.add_patch(centre_circle)
    ax.text(0, 0, f'{ssim_val:.3f}', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.set_title('SSIM\n(higher is better)', fontweight='bold')
    
    # PSNR bar
    ax = axes[1]
    psnr_val = metrics.get('psnr', 0)
    color = '#44aa44' if psnr_val > 20 else '#ffaa00' if psnr_val > 15 else '#ff4444'
    ax.barh([0], [psnr_val], color=color, height=0.5)
    ax.barh([0], [40-psnr_val], left=psnr_val, color='#eeeeee', height=0.5)
    ax.set_xlim(0, 40)
    ax.set_yticks([])
    ax.text(psnr_val/2, 0, f'{psnr_val:.1f} dB', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.set_title('PSNR\n(higher is better)', fontweight='bold')
    ax.set_xlabel('dB')
    
    # Reference lines
    ax.axvline(x=20, color='green', linestyle='--', alpha=0.5, label='Good (20dB)')
    ax.axvline(x=30, color='blue', linestyle='--', alpha=0.5, label='Excellent (30dB)')
    ax.legend(loc='upper right', fontsize=8)
    
    # FID (lower is better)
    ax = axes[2]
    fid_val = metrics.get('fid', 0)
    # FID scale: 0-500, with lower being better
    color = '#44aa44' if fid_val < 50 else '#ffaa00' if fid_val < 150 else '#ff4444'
    ax.barh([0], [min(fid_val, 500)], color=color, height=0.5)
    ax.barh([0], [max(0, 500-fid_val)], left=min(fid_val, 500), color='#eeeeee', height=0.5)
    ax.set_xlim(0, 500)
    ax.set_yticks([])
    ax.text(min(fid_val, 400)/2, 0, f'{fid_val:.1f}', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.set_title('FID\n(lower is better)', fontweight='bold')
    ax.set_xlabel('Score')
    
    # Reference lines
    ax.axvline(x=50, color='green', linestyle='--', alpha=0.5, label='Excellent (<50)')
    ax.axvline(x=150, color='orange', linestyle='--', alpha=0.5, label='Good (<150)')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_summary.png')
    plt.savefig(save_dir / 'metrics_summary.pdf')
    plt.close()
    print(f"Saved: metrics_summary.png/pdf")


def plot_noise_schedule_comparison(save_dir):
    """Compare different noise schedules."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Noise Schedule Comparison', fontsize=14, fontweight='bold', y=1.05)
    
    timesteps = 1000
    t = np.arange(timesteps)
    
    # Linear schedule
    beta_start, beta_end = 1e-4, 0.02
    betas_linear = np.linspace(beta_start, beta_end, timesteps)
    alphas_linear = 1 - betas_linear
    alphas_cumprod_linear = np.cumprod(alphas_linear)
    
    # Cosine schedule
    s = 0.008
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod_cosine = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod_cosine = alphas_cumprod_cosine / alphas_cumprod_cosine[0]
    betas_cosine = 1 - (alphas_cumprod_cosine[1:] / alphas_cumprod_cosine[:-1])
    betas_cosine = np.clip(betas_cosine, 0, 0.999)
    alphas_cumprod_cosine = alphas_cumprod_cosine[1:]
    
    # Gamma-rebalanced (approximation)
    gamma = 0.75
    alphas_cumprod_gamma = alphas_cumprod_linear ** gamma
    
    # Plot 1: Beta schedules
    ax = axes[0]
    ax.plot(t, betas_linear, label='Linear', linewidth=2)
    ax.plot(t, betas_cosine, label='Cosine', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('β_t')
    ax.set_title('Beta Schedule', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative alpha
    ax = axes[1]
    ax.plot(t, alphas_cumprod_linear, label='Linear', linewidth=2)
    ax.plot(t, alphas_cumprod_cosine, label='Cosine', linewidth=2)
    ax.plot(t, alphas_cumprod_gamma, label=f'Gamma (γ={gamma})', linewidth=2, linestyle='--')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('ᾱ_t')
    ax.set_title('Cumulative Alpha (Signal Retention)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: SNR (Signal-to-Noise Ratio)
    ax = axes[2]
    snr_linear = alphas_cumprod_linear / (1 - alphas_cumprod_linear + 1e-8)
    snr_cosine = alphas_cumprod_cosine / (1 - alphas_cumprod_cosine + 1e-8)
    snr_gamma = alphas_cumprod_gamma / (1 - alphas_cumprod_gamma + 1e-8)
    
    ax.semilogy(t, snr_linear, label='Linear', linewidth=2)
    ax.semilogy(t, snr_cosine, label='Cosine', linewidth=2)
    ax.semilogy(t, snr_gamma, label=f'Gamma (γ={gamma})', linewidth=2, linestyle='--')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('SNR (log scale)')
    ax.set_title('Signal-to-Noise Ratio', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'noise_schedules.png')
    plt.savefig(save_dir / 'noise_schedules.pdf')
    plt.close()
    print(f"Saved: noise_schedules.png/pdf")


def plot_frequency_analysis(generated_images, real_images, save_dir):
    """Analyze frequency content of generated vs real images."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Frequency Domain Analysis', fontsize=14, fontweight='bold', y=1.02)
    
    # Average FFT magnitude
    def compute_avg_fft(images):
        ffts = []
        for img in images[:50]:
            img_np = img.squeeze().cpu().numpy()
            fft = np.fft.fft2(img_np)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            ffts.append(magnitude)
        return np.mean(ffts, axis=0)
    
    gen_fft = compute_avg_fft(generated_images)
    real_fft = compute_avg_fft(real_images)
    
    # Plot FFT magnitudes
    vmax = max(gen_fft.max(), real_fft.max()) * 0.1
    
    axes[0, 0].imshow(np.log1p(gen_fft), cmap='viridis', vmax=np.log1p(vmax))
    axes[0, 0].set_title('Generated FFT (log)', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.log1p(real_fft), cmap='viridis', vmax=np.log1p(vmax))
    axes[0, 1].set_title('Real FFT (log)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Difference
    diff_fft = np.abs(gen_fft - real_fft)
    axes[0, 2].imshow(np.log1p(diff_fft), cmap='hot')
    axes[0, 2].set_title('FFT Difference (log)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Radial average
    def radial_profile(data):
        y, x = np.indices(data.shape)
        center = np.array(data.shape) // 2
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        return tbin / (nr + 1e-8)
    
    gen_radial = radial_profile(gen_fft)
    real_radial = radial_profile(real_fft)
    
    axes[1, 0].semilogy(gen_radial[:len(gen_radial)//2], label='Generated', linewidth=2)
    axes[1, 0].semilogy(real_radial[:len(real_radial)//2], label='Real', linewidth=2)
    axes[1, 0].set_xlabel('Frequency (cycles/image)')
    axes[1, 0].set_ylabel('Power (log)')
    axes[1, 0].set_title('Radial Power Spectrum', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Low vs high frequency content
    center = np.array(gen_fft.shape) // 2
    low_freq_mask = np.zeros_like(gen_fft, dtype=bool)
    y, x = np.ogrid[:gen_fft.shape[0], :gen_fft.shape[1]]
    low_freq_mask[(x - center[1])**2 + (y - center[0])**2 <= 20**2] = True
    
    gen_low = gen_fft[low_freq_mask].sum()
    gen_high = gen_fft[~low_freq_mask].sum()
    real_low = real_fft[low_freq_mask].sum()
    real_high = real_fft[~low_freq_mask].sum()
    
    x_pos = np.arange(2)
    width = 0.35
    
    axes[1, 1].bar(x_pos - width/2, [gen_low/1e6, gen_high/1e6], width, label='Generated', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, [real_low/1e6, real_high/1e6], width, label='Real', alpha=0.7)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(['Low Freq', 'High Freq'])
    axes[1, 1].set_ylabel('Total Power (×10⁶)')
    axes[1, 1].set_title('Frequency Band Comparison', fontweight='bold')
    axes[1, 1].legend()
    
    # Ratio plot
    ratio = (gen_radial + 1e-8) / (real_radial + 1e-8)
    axes[1, 2].plot(ratio[:len(ratio)//2], linewidth=2)
    axes[1, 2].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect match')
    axes[1, 2].fill_between(range(len(ratio)//2), 0.8, 1.2, alpha=0.2, color='green', label='±20% band')
    axes[1, 2].set_xlabel('Frequency (cycles/image)')
    axes[1, 2].set_ylabel('Generated / Real')
    axes[1, 2].set_title('Power Ratio', fontweight='bold')
    axes[1, 2].set_ylim(0, 2)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'frequency_analysis.png')
    plt.savefig(save_dir / 'frequency_analysis.pdf')
    plt.close()
    print(f"Saved: frequency_analysis.png/pdf")


def plot_diversity_analysis(generated_images, save_dir):
    """Analyze diversity of generated samples."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Sample Diversity Analysis', fontsize=14, fontweight='bold', y=1.05)
    
    n_samples = min(50, len(generated_images))
    
    # Pairwise SSIM matrix
    ssim_matrix = np.zeros((n_samples, n_samples))
    
    def compute_ssim_simple(img1, img2):
        C1, C2 = 0.01**2, 0.03**2
        mu1, mu2 = img1.mean(), img2.mean()
        sigma1_sq = ((img1 - mu1)**2).mean()
        sigma2_sq = ((img2 - mu2)**2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        return ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    for i in range(n_samples):
        for j in range(n_samples):
            ssim_matrix[i, j] = compute_ssim_simple(
                generated_images[i].squeeze(),
                generated_images[j].squeeze()
            ).item()
    
    # Plot SSIM matrix
    ax = axes[0]
    im = ax.imshow(ssim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')
    ax.set_title('Pairwise SSIM Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # SSIM distribution (off-diagonal)
    ax = axes[1]
    off_diag_ssim = ssim_matrix[np.triu_indices(n_samples, k=1)]
    ax.hist(off_diag_ssim, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=off_diag_ssim.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {off_diag_ssim.mean():.3f}')
    ax.set_xlabel('SSIM')
    ax.set_ylabel('Density')
    ax.set_title('Inter-Sample SSIM Distribution', fontweight='bold')
    ax.legend()
    
    # Pixel intensity distribution
    ax = axes[2]
    all_pixels = torch.stack([g.squeeze() for g in generated_images[:n_samples]]).flatten().numpy()
    ax.hist(all_pixels, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Density')
    ax.set_title('Generated Pixel Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'diversity_analysis.png')
    plt.savefig(save_dir / 'diversity_analysis.pdf')
    plt.close()
    print(f"Saved: diversity_analysis.png/pdf")


def plot_architecture_diagram(save_dir):
    """Create a visual architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('SBLDM Architecture Overview', fontsize=16, fontweight='bold', pad=20)
    
    # Colors
    colors = {
        'input': '#a8d5ba',
        'vae': '#ffd6a5',
        'latent': '#fdffb6',
        'unet': '#caffbf',
        'diffusion': '#9bf6ff',
        'output': '#bdb2ff'
    }
    
    def draw_box(x, y, w, h, color, label, sublabel=None):
        rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.1 if sublabel else 0), label, ha='center', va='center', 
                fontsize=11, fontweight='bold')
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha='center', va='center', fontsize=9)
    
    def draw_arrow(x1, y1, x2, y2, label=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.2, label, ha='center', fontsize=9)
    
    # Training path (top)
    ax.text(8, 7.5, 'Training Pipeline', ha='center', fontsize=14, fontweight='bold', style='italic')
    
    draw_box(0.5, 5.5, 2, 1.2, colors['input'], 'Brain MRI', '128×128')
    draw_arrow(2.5, 6.1, 3.5, 6.1)
    
    draw_box(3.5, 5.5, 2.5, 1.2, colors['vae'], 'VAE Encoder', 'μ, σ')
    draw_arrow(6, 6.1, 7, 6.1)
    
    draw_box(7, 5.5, 2, 1.2, colors['latent'], 'Latent z', '32×32×4')
    draw_arrow(9, 6.1, 10, 6.1, 'Add noise')
    
    draw_box(10, 5.5, 2.5, 1.2, colors['diffusion'], 'Noisy Latent', 'z_t')
    draw_arrow(12.5, 6.1, 13.5, 6.1)
    
    draw_box(13.5, 5.5, 2, 1.2, colors['unet'], 'UNet', 'Predict ε')
    
    # Loss arrow
    ax.annotate('', xy=(14.5, 5.5), xytext=(14.5, 4.5),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(14.5, 4.2, 'Loss', ha='center', fontsize=10, color='red', fontweight='bold')
    
    # Sampling path (bottom)
    ax.text(8, 3.5, 'Sampling Pipeline', ha='center', fontsize=14, fontweight='bold', style='italic')
    
    draw_box(0.5, 1.5, 2, 1.2, colors['diffusion'], 'Noise', 'z_T ~ N(0,I)')
    draw_arrow(2.5, 2.1, 3.5, 2.1, 'DDIM')
    
    draw_box(3.5, 1.5, 2.5, 1.2, colors['unet'], 'UNet', 'Denoise')
    draw_arrow(6, 2.1, 7, 2.1)
    
    draw_box(7, 1.5, 2, 1.2, colors['latent'], 'Clean z', '32×32×4')
    draw_arrow(9, 2.1, 10, 2.1)
    
    draw_box(10, 1.5, 2.5, 1.2, colors['vae'], 'VAE Decoder', 'Reconstruct')
    draw_arrow(12.5, 2.1, 13.5, 2.1)
    
    draw_box(13.5, 1.5, 2, 1.2, colors['output'], 'Synthetic', '128×128')
    
    # Add legend
    legend_y = 0.3
    for i, (name, color) in enumerate(colors.items()):
        x = 1 + i * 2.5
        rect = Rectangle((x, legend_y), 0.3, 0.3, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 0.4, legend_y + 0.15, name.capitalize(), va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'architecture_diagram.png')
    plt.savefig(save_dir / 'architecture_diagram.pdf')
    plt.close()
    print(f"Saved: architecture_diagram.png/pdf")


def main():
    parser = argparse.ArgumentParser(description='Generate visualization figures for SBLDM')
    parser.add_argument('--config', type=str, default='configs/config_full.yaml', help='Config file')
    parser.add_argument('--log_dir', type=str, default='logs_kaggle_full_lat8x', help='Training logs directory')
    parser.add_argument('--vae_checkpoint', type=str, default='checkpoints_kaggle_full_lat8x/vae_epoch_136_best.pt')
    parser.add_argument('--diffusion_checkpoint', type=str, default='checkpoints_kaggle_full_lat8x/diffusion_step_100000.pt')
    parser.add_argument('--data_dir', type=str, default='data/processed/test', help='Test data directory')
    parser.add_argument('--output_dir', type=str, default='figures_kaggle_full_lat8x', help='Output directory for figures')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load training history
    print("\n1. Loading training history...")
    vae_history, diff_history = load_training_history(args.log_dir)
    
    # Plot training curves
    print("2. Generating training curves...")
    if vae_history or diff_history:
        plot_training_curves(vae_history, diff_history, save_dir)
    
    # Load models
    print("3. Loading models...")
    vae = create_vae(config['vae'])
    if Path(args.vae_checkpoint).exists():
        vae_state = torch.load(args.vae_checkpoint, map_location=device, weights_only=False)
        vae.load_state_dict(vae_state['model_state_dict'])
    vae = vae.to(device).eval()
    
    unet = create_unet(config['unet'])
    if Path(args.diffusion_checkpoint).exists():
        diff_state = torch.load(args.diffusion_checkpoint, map_location=device, weights_only=False)
        unet.load_state_dict(diff_state['unet_state_dict'])
    unet = unet.to(device).eval()
    
    schedule = NoiseSchedule(
        timesteps=config['diffusion']['timesteps'],
        schedule_type=config['diffusion']['beta_schedule'],
        device=device
    )
    sampler = DDIMSampler(schedule, unet, eta=0.0)
    
    # Load real images
    print("4. Loading real test images...")
    real_dir = Path(args.data_dir)  # data_dir should already point to test directory
    resolution = config['data']['resolution']
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor()
    ])
    
    real_images = []
    for p in sorted(real_dir.glob('*.png'))[:100]:
        img = Image.open(p).convert('L')
        real_images.append(transform(img))
    
    if len(real_images) == 0:
        raise RuntimeError(f"No images found in {real_dir}")
    
    real_images = torch.stack(real_images)
    print(f"   Loaded {len(real_images)} real images")
    
    # Generate samples
    print("5. Generating synthetic samples...")
    generated_images = []
    latent_size = config['data']['resolution'] // config['vae']['downsample_factor']
    
    with torch.no_grad():
        for i in range(0, 64, 8):
            bs = min(8, 64 - i)
            shape = (bs, vae.latent_channels, latent_size, latent_size)
            z = sampler.sample(shape, num_steps=50, device=device, progress=False)
            imgs = vae.decode(z)
            imgs = torch.clamp((imgs + 1) / 2, 0, 1)
            imgs = F.interpolate(imgs, size=(resolution, resolution), mode='bilinear', align_corners=False)
            generated_images.append(imgs.cpu())
    
    generated_images = torch.cat(generated_images, dim=0)
    print(f"   Generated {len(generated_images)} samples")
    
    # Generate all visualizations
    print("\n6. Generating visualizations...")
    
    print("   - Sample comparison grid...")
    plot_sample_grid(generated_images, real_images, save_dir)
    
    print("   - Reconstruction quality...")
    plot_reconstruction_quality(vae, real_images, device, save_dir)
    
    print("   - Denoising process...")
    plot_denoising_process(sampler, vae, device, save_dir, latent_size)
    
    print("   - Latent space analysis...")
    plot_latent_space_analysis(vae, real_images, device, save_dir)
    
    print("   - Noise schedule comparison...")
    plot_noise_schedule_comparison(save_dir)
    
    print("   - Frequency analysis...")
    plot_frequency_analysis(generated_images, real_images, save_dir)
    
    print("   - Diversity analysis...")
    plot_diversity_analysis(generated_images, save_dir)
    
    print("   - Architecture diagram...")
    plot_architecture_diagram(save_dir)
    
    # Metrics summary (with placeholder values - update with actual)
    print("   - Metrics summary...")
    metrics = {
        'ssim': 0.2798,
        'psnr': 12.53,
        'fid': 371.07
    }
    plot_metrics_summary(metrics, save_dir)
    
    print(f"\n✅ All figures saved to: {save_dir}/")
    print("\nGenerated files:")
    for f in sorted(save_dir.glob('*')):
        print(f"   - {f.name}")


if __name__ == '__main__':
    main()
