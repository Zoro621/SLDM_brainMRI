"""
Compute FID, SSIM, and PSNR metrics for all trained models.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import json
import yaml
from tqdm import tqdm

# Import project modules
from models.vae import VAE, create_vae
from models.simple_unet import SimpleUNet, create_unet
from diffusion.schedules import NoiseSchedule
from diffusion.samplers import DDIMSampler
from eval.metrics import compute_ssim, compute_psnr, FIDCalculator

def load_model(config_path, vae_checkpoint, unet_checkpoint, device='cuda'):
    """Load VAE and diffusion model."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load VAE
    vae = create_vae(config['vae'])
    vae_state = torch.load(vae_checkpoint, map_location=device, weights_only=False)
    vae.load_state_dict(vae_state['model_state_dict'])
    vae = vae.to(device).eval()
    
    # Load UNet
    unet = create_unet(config['unet'])
    unet_state = torch.load(unet_checkpoint, map_location=device, weights_only=False)
    unet.load_state_dict(unet_state['unet_state_dict'])
    unet = unet.to(device).eval()
    
    # Create sampler
    schedule = NoiseSchedule(
        timesteps=config['diffusion']['timesteps'],
        schedule_type=config['diffusion']['beta_schedule'],
        device=device
    )
    sampler = DDIMSampler(schedule, unet, eta=0.0)
    
    return vae, sampler, config

def load_test_images(data_dir, num_samples=100):
    """Load real test images."""
    test_dir = Path(data_dir)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    images = []
    for p in sorted(test_dir.glob('*.png'))[:num_samples]:
        img = Image.open(p).convert('L')
        images.append(transform(img))
    
    return torch.stack(images)

def compute_vae_metrics(vae, real_images, device):
    """Compute SSIM and PSNR for VAE reconstructions."""
    vae.eval()
    ssim_scores = []
    psnr_scores = []
    
    with torch.no_grad():
        for img in real_images:
            img = img.unsqueeze(0).to(device)
            recon, _, _ = vae(img)
            
            # Keep as tensors for compute_ssim/psnr
            img_tensor = img[0, 0]
            recon_tensor = recon[0, 0]
            
            ssim_val = compute_ssim(img_tensor, recon_tensor, data_range=1.0)
            psnr_val = compute_psnr(img_tensor, recon_tensor, data_range=1.0)
            
            ssim_scores.append(ssim_val)
            psnr_scores.append(psnr_val)
    
    return {
        'ssim_mean': float(np.mean(ssim_scores)),
        'ssim_std': float(np.std(ssim_scores)),
        'psnr_mean': float(np.mean(psnr_scores)),
        'psnr_std': float(np.std(psnr_scores))
    }

def compute_generation_metrics(vae, sampler, real_images, num_samples, device, fid_calculator=None):
    """Compute SSIM/PSNR for generated samples and optional FID."""
    # Generate samples
    print(f"   Generating {num_samples} samples...")
    generated = []
    batch_size = 16
    
    for i in tqdm(range(0, num_samples, batch_size), desc="   Generating"):
        n = min(batch_size, num_samples - i)
        shape = (n, vae.latent_channels, real_images.shape[-1] // vae.downsample_factor, real_images.shape[-1] // vae.downsample_factor)
        
        with torch.no_grad():
            latents = sampler.sample(shape, num_steps=50, device=device, progress=False)
            samples = vae.decode(latents)
            generated.append(samples.cpu())
    
    generated = torch.cat(generated, dim=0)
    generated = torch.clamp((generated + 1) / 2, 0, 1)
    
    # Compute metrics against random real images
    ssim_scores = []
    psnr_scores = []
    
    real_subset = torch.clamp(real_images[:num_samples], 0, 1)
    
    for gen, real in zip(generated, real_subset):
        gen_tensor = gen[0]
        real_tensor = real[0]
        
        ssim_val = compute_ssim(gen_tensor, real_tensor, data_range=1.0)
        psnr_val = compute_psnr(gen_tensor, real_tensor, data_range=1.0)
        
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)

    fid_value = None
    if fid_calculator is not None:
        real_loader = DataLoader(TensorDataset(real_subset), batch_size=32, shuffle=False)
        fid_calculator.compute_real_statistics(real_loader, max_samples=len(real_subset))
        fid_value = fid_calculator.compute_generated_fid(generated)

    return {
        'ssim_mean': float(np.mean(ssim_scores)),
        'ssim_std': float(np.std(ssim_scores)),
        'psnr_mean': float(np.mean(psnr_scores)),
        'psnr_std': float(np.std(psnr_scores)),
        'fid': fid_value
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    models = [
        {
            'name': 'FLAIR',
            'config': 'configs/config_brats.yaml',
            'vae_ckpt': 'checkpoints_brats/vae_best.pt',
            'diff_ckpt': 'checkpoints_brats/diffusion_step_100000.pt',
            'data_dir': 'data/brats_processed/test'
        },
        {
            'name': 'T1ce',
            'config': 'configs/config_brats_t1ce.yaml',
            'vae_ckpt': 'checkpoints_brats_t1ce/vae_best.pt',
            'diff_ckpt': 'checkpoints_brats_t1ce/diffusion_step_200000.pt',
            'data_dir': 'data/brats_t1ce_processed/test'
        },
        {
            'name': 'Kaggle',
            'config': 'configs/config_full.yaml',
            'vae_ckpt': 'checkpoints_kaggle_full_lat8x/vae_epoch_136_best.pt',
            'diff_ckpt': 'checkpoints_kaggle_full_lat8x/diffusion_step_100000.pt',
            'data_dir': 'data/processed/test'
        }
    ]
    
    results = {}
    
    for model_info in models:
        name = model_info['name']
        print(f"{'='*80}")
        print(f"Computing metrics for {name} model")
        print(f"{'='*80}")
        
        # Check if files exist
        if not Path(model_info['vae_ckpt']).exists():
            print(f"  VAE checkpoint not found: {model_info['vae_ckpt']}")
            continue
        if not Path(model_info['diff_ckpt']).exists():
            print(f"  Diffusion checkpoint not found: {model_info['diff_ckpt']}")
            continue
        if not Path(model_info['data_dir']).exists():
            print(f"  Data directory not found: {model_info['data_dir']}")
            continue
        
        # Load model
        print("  Loading models...")
        vae, sampler, config = load_model(
            model_info['config'],
            model_info['vae_ckpt'],
            model_info['diff_ckpt'],
            device
        )

        fid_calculator = FIDCalculator(device)
        
        # Load test images
        print("  Loading test images...")
        real_images = load_test_images(model_info['data_dir'], num_samples=200)
        print(f"  Loaded {len(real_images)} test images")
        
        # Compute VAE reconstruction metrics
        print("  Computing VAE reconstruction metrics...")
        vae_metrics = compute_vae_metrics(vae, real_images, device)
        
        # Compute generation metrics
        print("  Computing generation metrics...")
        gen_metrics = compute_generation_metrics(vae, sampler, real_images, num_samples=200, device=device, fid_calculator=fid_calculator)
        
        results[name] = {
            'vae_reconstruction': vae_metrics,
            'generation': gen_metrics
        }
        
        print(f"\n  VAE Reconstruction:")
        print(f"    SSIM: {vae_metrics['ssim_mean']:.4f} ± {vae_metrics['ssim_std']:.4f}")
        print(f"    PSNR: {vae_metrics['psnr_mean']:.2f} ± {vae_metrics['psnr_std']:.2f} dB")
        print(f"\n  Generated Samples:")
        print(f"    SSIM: {gen_metrics['ssim_mean']:.4f} ± {gen_metrics['ssim_std']:.4f}")
        print(f"    PSNR: {gen_metrics['psnr_mean']:.2f} ± {gen_metrics['psnr_std']:.2f} dB")
        if gen_metrics['fid'] is not None:
            print(f"    FID:  {gen_metrics['fid']:.2f}")
        print()
    
    # Save results
    output_file = 'quality_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print comparison table
    print("\n\nCOMPARISON TABLE:")
    print(f"{'='*80}")
    print(f"{'Model':<10} {'VAE SSIM':<12} {'VAE PSNR':<12} {'Gen SSIM':<12} {'Gen PSNR':<12} {'FID':<10}")
    print(f"{'-'*80}")
    for name, metrics in results.items():
        vae = metrics['vae_reconstruction']
        gen = metrics['generation']
        fid_str = f"{gen['fid']:.2f}" if gen['fid'] is not None else "n/a"
        print(f"{name:<10} {vae['ssim_mean']:.4f}±{vae['ssim_std']:.3f}  "
              f"{vae['psnr_mean']:.1f}±{vae['psnr_std']:.1f} dB   "
              f"{gen['ssim_mean']:.4f}±{gen['ssim_std']:.3f}  "
              f"{gen['psnr_mean']:.1f}±{gen['psnr_std']:.1f} dB   "
              f"{fid_str:<10}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
