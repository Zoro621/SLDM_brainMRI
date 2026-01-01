"""
SBLDM Sampling Script
Generate samples from trained models.

Usage:
    python sample.py --config configs/config_full.yaml \
                     --vae_checkpoint checkpoints/vae_best.pt \
                     --diffusion_checkpoint checkpoints/diffusion_best.pt \
                     --num_samples 64 \
                     --output_dir outputs/samples
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional

import torch
from torch.cuda.amp import autocast
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from models.vae import VAE, create_vae
from models.simple_unet import SimpleUNet, create_unet
from models.ema import EMA
from diffusion.schedules import NoiseSchedule
from diffusion.samplers import DDPMSampler, DDIMSampler, AdaptiveSampler, create_sampler


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_models(
    config: dict,
    vae_checkpoint: str,
    diffusion_checkpoint: str,
    device: str = "cuda",
    use_ema: bool = True
):
    """Load trained VAE and diffusion models."""
    
    # Load VAE
    vae = create_vae(config['vae'])
    vae_state = torch.load(vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_state['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    
    # Load UNet
    unet = create_unet(config['unet'])
    diff_state = torch.load(diffusion_checkpoint, map_location=device)
    
    if use_ema and 'ema_state_dict' in diff_state:
        # Load EMA weights
        ema = EMA(unet, decay=config['diffusion'].get('ema_decay', 0.9999))
        ema.load_state_dict(diff_state['ema_state_dict'])
        ema.apply_shadow()  # Apply EMA weights to model
    else:
        unet.load_state_dict(diff_state['unet_state_dict'])
    
    unet = unet.to(device)
    unet.eval()
    
    return vae, unet


@torch.no_grad()
def generate_samples(
    vae: VAE,
    unet: SimpleUNet,
    config: dict,
    num_samples: int,
    sampler_type: str = "ddim",
    ddim_steps: int = 50,
    eta: float = 0.0,
    use_adaptive: bool = False,
    adaptive_threshold: float = 0.05,
    device: str = "cuda",
    batch_size: int = 8,
    progress: bool = True
) -> torch.Tensor:
    """
    Generate samples from the trained model.
    
    Args:
        vae: Trained VAE decoder
        unet: Trained UNet score model
        config: Configuration dict
        num_samples: Number of samples to generate
        sampler_type: "ddpm" or "ddim"
        ddim_steps: Number of DDIM steps
        eta: DDIM stochasticity (0 = deterministic)
        use_adaptive: Use adaptive sampling
        adaptive_threshold: Early stop threshold
        device: Device to use
        batch_size: Batch size for generation
        progress: Show progress bar
        
    Returns:
        Generated images [N, C, H, W] in [0, 1] range
    """
    vae.eval()
    unet.eval()
    
    # Setup noise schedule
    schedule = NoiseSchedule(
        schedule_type=config['diffusion']['beta_schedule'],
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        gamma=config['diffusion'].get('gamma', 1.0),
        device=device
    )
    
    # Create sampler
    if use_adaptive:
        base_sampler = DDIMSampler(schedule, unet, eta=eta)
        sampler = AdaptiveSampler(
            base_sampler,
            threshold=adaptive_threshold,
            min_steps=config['sampling'].get('adaptive_min_steps', 20)
        )
    else:
        sampler = create_sampler(
            sampler_type,
            schedule,
            unet,
            eta=eta
        )
    
    # Latent shape
    latent_size = config['data']['resolution'] // vae.downsample_factor
    latent_channels = config['unet']['in_channels']
    
    all_images = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    if progress:
        pbar = tqdm(range(num_batches), desc="Generating samples")
    else:
        pbar = range(num_batches)
    
    for i in pbar:
        # Current batch size (handle last batch)
        current_batch = min(batch_size, num_samples - i * batch_size)
        
        z_shape = (current_batch, latent_channels, latent_size, latent_size)
        
        # Sample latents
        if use_adaptive:
            z, info = sampler.sample(
                z_shape,
                num_steps=ddim_steps,
                device=device,
                progress=False
            )
            if progress:
                pbar.set_postfix({'steps': info['actual_steps']})
        else:
            if sampler_type == "ddim":
                z = sampler.sample(
                    z_shape,
                    num_steps=ddim_steps,
                    device=device,
                    progress=False
                )
            else:
                z = sampler.sample(
                    z_shape,
                    device=device,
                    progress=False
                )
        
        # Decode to images
        images = vae.decode(z)
        
        # Denormalize [-1, 1] -> [0, 1]
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        all_images.append(images.cpu())
    
    return torch.cat(all_images, dim=0)


@torch.no_grad()
def reconstruct_images(
    vae: VAE,
    images: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Reconstruct images through VAE.
    
    Args:
        vae: Trained VAE
        images: Input images [N, C, H, W] in [-1, 1]
        device: Device
        
    Returns:
        Reconstructed images [N, C, H, W] in [0, 1]
    """
    vae.eval()
    
    images = images.to(device)
    recon, _, _ = vae(images, sample_posterior=False)
    
    # Denormalize
    recon = (recon + 1) / 2
    recon = torch.clamp(recon, 0, 1)
    
    return recon.cpu()


def save_sample_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 8
):
    """Save images as a grid."""
    save_image(images, path, nrow=nrow, padding=2, normalize=False)
    print(f"Saved {len(images)} samples to {path}")


def save_individual_samples(
    images: torch.Tensor,
    output_dir: str,
    prefix: str = "sample"
):
    """Save each sample as individual file."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img in enumerate(images):
        # Convert to PIL
        img_np = (img.squeeze().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='L')
        
        path = os.path.join(output_dir, f"{prefix}_{i:05d}.png")
        pil_img.save(path)
    
    print(f"Saved {len(images)} individual samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate SBLDM samples')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                        help='Path to VAE checkpoint')
    parser.add_argument('--diffusion_checkpoint', type=str, required=True,
                        help='Path to diffusion checkpoint')
    parser.add_argument('--output_dir', type=str, default='./samples',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for generation')
    parser.add_argument('--sampler', type=str, default='ddim',
                        choices=['ddpm', 'ddim'],
                        help='Sampler type')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='Number of DDIM steps')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM eta (0 = deterministic)')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive sampling')
    parser.add_argument('--adaptive_threshold', type=float, default=0.05,
                        help='Adaptive sampling threshold')
    parser.add_argument('--save_grid', action='store_true',
                        help='Save as image grid')
    parser.add_argument('--save_individual', action='store_true',
                        help='Save individual images')
    parser.add_argument('--no_ema', action='store_true',
                        help='Do not use EMA weights')
    parser.add_argument('--device', type=str, default=None,
                        help='Device')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config and models
    config = load_config(args.config)
    
    print("Loading models...")
    vae, unet = load_models(
        config,
        args.vae_checkpoint,
        args.diffusion_checkpoint,
        device=device,
        use_ema=not args.no_ema
    )
    
    print(f"\nGenerating {args.num_samples} samples...")
    print(f"Sampler: {args.sampler}, Steps: {args.ddim_steps}, Eta: {args.eta}")
    if args.adaptive:
        print(f"Adaptive sampling: threshold={args.adaptive_threshold}")
    
    # Generate
    images = generate_samples(
        vae=vae,
        unet=unet,
        config=config,
        num_samples=args.num_samples,
        sampler_type=args.sampler,
        ddim_steps=args.ddim_steps,
        eta=args.eta,
        use_adaptive=args.adaptive,
        adaptive_threshold=args.adaptive_threshold,
        device=device,
        batch_size=args.batch_size
    )
    
    print(f"Generated {len(images)} images, shape: {images.shape}")
    
    # Save
    if args.save_grid or not args.save_individual:
        grid_path = os.path.join(args.output_dir, 'samples_grid.png')
        save_sample_grid(images, grid_path)
    
    if args.save_individual:
        save_individual_samples(images, args.output_dir)
    
    print("Done!")


if __name__ == '__main__':
    main()
