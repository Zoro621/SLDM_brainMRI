"""
SBLDM Diffusion Training Script
Trains the latent diffusion model on pre-computed VAE latents.

Usage:
    python train_diffusion.py --config configs/config_full.yaml --vae_checkpoint checkpoints/vae_best.pt
"""

import os
import sys
import argparse
import yaml
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import get_dataloaders
from models.vae import VAE, create_vae
from models.simple_unet import SimpleUNet, create_unet
from models.ema import EMA
from diffusion.schedules import NoiseSchedule
from diffusion.loss import DiffusionLoss, FrequencyAwareLoss, LatentCutMix
from diffusion.samplers import DDIMSampler, create_sampler


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_directories(config: dict) -> dict:
    dirs = {
        'output': Path(config['experiment']['output_dir']),
        'log': Path(config['experiment']['log_dir']),
        'checkpoint': Path(config['experiment']['checkpoint_dir']),
        'samples': Path(config['experiment']['output_dir']) / 'samples'
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return {k: str(v) for k, v in dirs.items()}


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DiffusionTrainer:
    """
    Latent Diffusion Training manager.
    
    Handles:
    - Pre-computing VAE latents
    - Training diffusion UNet in latent space
    - Novel contributions: freq loss, cutmix, gamma schedule
    - EMA for stable sampling
    - Periodic sampling during training
    """
    
    def __init__(
        self,
        unet: SimpleUNet,
        vae: VAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda"
    ):
        self.unet = unet.to(device)
        self.vae = vae.to(device)
        self.vae.eval()  # VAE is frozen
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        diff_config = config['diffusion']
        
        # Noise schedule
        self.schedule = NoiseSchedule(
            schedule_type=diff_config['beta_schedule'],
            timesteps=diff_config['timesteps'],
            beta_start=diff_config['beta_start'],
            beta_end=diff_config['beta_end'],
            gamma=diff_config.get('gamma', 1.0),
            device=device
        )
        
        # Loss function
        if diff_config.get('use_freq_loss', False):
            self.loss_fn = FrequencyAwareLoss(
                spatial_loss_type=diff_config.get('loss_type', 'mse'),
                freq_loss_type=diff_config.get('freq_loss_type', 'mse'),
                freq_weight=diff_config.get('freq_loss_weight', 0.1)
            )
            self.use_freq_loss = True
        else:
            self.loss_fn = DiffusionLoss(diff_config.get('loss_type', 'mse'))
            self.use_freq_loss = False
        
        # Latent CutMix augmentation
        if diff_config.get('use_latent_cutmix', False):
            self.cutmix = LatentCutMix(
                prob=diff_config.get('cutmix_prob', 0.25),
                alpha=diff_config.get('cutmix_alpha', 1.0)
            )
        else:
            self.cutmix = None
        
        # Optimizer
        self.optimizer = optim.AdamW(
            unet.parameters(),
            lr=diff_config['learning_rate'],
            weight_decay=diff_config.get('weight_decay', 0.0)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=diff_config['training_steps'],
            eta_min=diff_config['learning_rate'] * 0.01
        )
        
        # Warmup
        self.warmup_steps = diff_config.get('warmup_steps', 1000)
        
        # EMA
        self.ema = EMA(unet, decay=diff_config.get('ema_decay', 0.9999))
        
        # Mixed precision
        self.use_amp = config['hardware'].get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient clipping
        self.grad_clip = diff_config.get('gradient_clip', 1.0)
        
        # Sampler for visualization
        self.sampler = DDIMSampler(
            self.schedule,
            self.unet,
            eta=config['sampling'].get('ddim_eta', 0.0)
        )
        
        # Tracking
        self.step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        # Pre-computed latents cache
        self.train_latents = None
        self.val_latents = None
    
    def get_lr_scale(self) -> float:
        """Get learning rate scale for warmup."""
        if self.step < self.warmup_steps:
            return self.step / self.warmup_steps
        return 1.0
    
    @torch.no_grad()
    def precompute_latents(self):
        """
        Pre-compute all VAE latents for efficient training.
        Stores latents on CPU to save GPU memory.
        """
        print("Pre-computing VAE latents...")
        
        def encode_all(loader):
            latents = []
            for batch in tqdm(loader, desc="Encoding"):
                x = batch.to(self.device)
                z = self.vae.get_latent(x, sample=False)
                latents.append(z.cpu())
            return torch.cat(latents, dim=0)
        
        self.train_latents = encode_all(self.train_loader)
        self.val_latents = encode_all(self.val_loader)
        
        print(f"Train latents: {self.train_latents.shape}")
        print(f"Val latents: {self.val_latents.shape}")
    
    def get_latent_batch(
        self,
        split: str = 'train',
        batch_size: int = 16
    ) -> torch.Tensor:
        """Get a random batch of latents."""
        if split == 'train':
            latents = self.train_latents
        else:
            latents = self.val_latents
        
        indices = torch.randint(0, len(latents), (batch_size,))
        return latents[indices].to(self.device)
    
    def train_step(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step.
        
        Args:
            z: Batch of latent codes [B, C, H, W]
            
        Returns:
            loss: Loss value
            metrics: Dictionary of metrics
        """
        self.unet.train()
        
        B = z.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.schedule.timesteps, (B,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(z)
        
        # Apply CutMix augmentation
        if self.cutmix is not None:
            z, noise = self.cutmix(z, noise)
        
        # Forward diffusion (add noise)
        z_noisy, noise = self.schedule.q_sample(z, t, noise)
        
        # Predict noise
        self.optimizer.zero_grad()
        
        with autocast(enabled=self.use_amp):
            noise_pred = self.unet(z_noisy, t)
            
            if self.use_freq_loss:
                loss, loss_dict = self.loss_fn(noise_pred, noise)
            else:
                loss = self.loss_fn(noise_pred, noise)
                loss_dict = {'loss': loss.item()}
        
        # Backward
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.unet.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.unet.parameters(), self.grad_clip)
            self.optimizer.step()
        
        # Apply warmup
        lr_scale = self.get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config['diffusion']['learning_rate'] * lr_scale
        
        # Update scheduler (after warmup)
        if self.step >= self.warmup_steps:
            self.scheduler.step()
        
        # Update EMA
        self.ema.update()
        
        self.step += 1
        
        return loss, loss_dict
    
    @torch.no_grad()
    def validate(self, num_batches: int = 10) -> Dict[str, float]:
        """Run validation."""
        self.unet.eval()
        
        total_loss = 0
        
        for i in range(num_batches):
            z = self.get_latent_batch('val', self.config['diffusion']['batch_size'])
            B = z.shape[0]
            
            t = torch.randint(0, self.schedule.timesteps, (B,), device=self.device)
            noise = torch.randn_like(z)
            z_noisy, noise = self.schedule.q_sample(z, t, noise)
            
            with self.ema.average_parameters():
                noise_pred = self.unet(z_noisy, t)
            
            if self.use_freq_loss:
                loss, _ = self.loss_fn(noise_pred, noise)
            else:
                loss = self.loss_fn(noise_pred, noise)
            
            total_loss += loss.item()
        
        return {'val_loss': total_loss / num_batches}
    
    @torch.no_grad()
    def sample_images(
        self,
        num_samples: int = 8,
        ddim_steps: int = 50
    ) -> torch.Tensor:
        """Generate sample images."""
        self.unet.eval()
        
        # Get latent shape
        z_shape = (
            num_samples,
            self.config['unet']['in_channels'],
            self.config['data']['resolution'] // self.vae.downsample_factor,
            self.config['data']['resolution'] // self.vae.downsample_factor
        )
        
        # Sample with EMA weights
        with self.ema.average_parameters():
            z = self.sampler.sample(
                z_shape,
                num_steps=ddim_steps,
                device=self.device,
                progress=False
            )
        
        # Decode latents to images
        images = self.vae.decode(z)
        
        # Denormalize from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        return images
    
    def save_samples(
        self,
        path: str,
        num_samples: int = 16
    ):
        """Generate and save sample images."""
        from torchvision.utils import save_image
        
        images = self.sample_images(num_samples)
        save_image(images, path, nrow=4, normalize=False)
    
    def save_checkpoint(
        self,
        path: str,
        is_best: bool = False
    ):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.step,
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.step = checkpoint['step']
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Resumed from step {self.step}")
    
    def train(
        self,
        total_steps: int,
        checkpoint_dir: str,
        sample_dir: str,
        log_dir: str,
        save_intermediate_checkpoints: bool = True
    ):
        """Full training loop."""
        diff_config = self.config['diffusion']
        
        print(f"\nStarting diffusion training for {total_steps} steps")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"UNet parameters: {self.unet.get_num_params():,}")
        print(f"Noise schedule: {diff_config['beta_schedule']}")
        print(f"Save intermediate checkpoints: {save_intermediate_checkpoints}")
        
        if diff_config.get('use_freq_loss'):
            print(f"Frequency-aware loss: weight={diff_config['freq_loss_weight']}")
        if diff_config.get('use_latent_cutmix'):
            print(f"Latent CutMix: prob={diff_config['cutmix_prob']}")
        
        # Pre-compute latents
        if self.train_latents is None:
            self.precompute_latents()
        
        log_interval = diff_config.get('log_interval', 100)
        sample_interval = diff_config.get('sample_interval', 2000)
        save_interval = diff_config.get('save_interval', 5000)
        
        batch_size = diff_config['batch_size']
        
        pbar = tqdm(range(self.step, total_steps), desc="Training")
        running_loss = 0
        
        for step in pbar:
            # Get batch
            z = self.get_latent_batch('train', batch_size)
            
            # Train step
            loss, loss_dict = self.train_step(z)
            running_loss += loss_dict['loss']
            
            # Logging
            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                lr = self.optimizer.param_groups[0]['lr']
                
                self.history['train_loss'].append(avg_loss)
                self.history['lr'].append(lr)
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{lr:.6f}'
                })
                
                running_loss = 0
            
            # Validation and sampling
            if (step + 1) % sample_interval == 0:
                val_metrics = self.validate()
                self.history['val_loss'].append(val_metrics['val_loss'])
                
                print(f"\nStep {step + 1}: val_loss = {val_metrics['val_loss']:.4f}")
                
                # Check for best
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                    print("New best validation loss!")
                
                # Save samples
                sample_path = os.path.join(sample_dir, f'samples_step_{step + 1}.png')
                self.save_samples(sample_path, num_samples=16)
                print(f"Saved samples to {sample_path}")
                
                # Save history to JSON periodically
                history_path = os.path.join(log_dir, 'diffusion_training_history.json')
                with open(history_path, 'w') as f:
                    json.dump({
                        'current_step': step + 1,
                        'best_val_loss': self.best_val_loss,
                        'train_loss': self.history['train_loss'],
                        'val_loss': self.history['val_loss'],
                        'lr': self.history['lr']
                    }, f, indent=2)
            
            # Checkpointing (only if enabled)
            if save_intermediate_checkpoints and (step + 1) % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_step_{step + 1}.pt')
                is_best = (step + 1) % sample_interval == 0 and val_metrics['val_loss'] < self.best_val_loss
                self.save_checkpoint(checkpoint_path, is_best=is_best)
                
                # Save latest
                latest_path = os.path.join(checkpoint_dir, 'diffusion_latest.pt')
                self.save_checkpoint(latest_path)
        
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
        
        # Save final checkpoint
        final_checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_step_{total_steps}_final.pt')
        self.save_checkpoint(final_checkpoint_path, is_best=False)
        print(f"Saved final checkpoint to {final_checkpoint_path}")
        
        # Save final history
        history_path = os.path.join(log_dir, 'diffusion_training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'total_steps': total_steps,
                'best_val_loss': self.best_val_loss,
                'train_loss': self.history['train_loss'],
                'val_loss': self.history['val_loss'],
                'lr': self.history['lr'],
                'config': self.config
            }, f, indent=2)
        print(f"Saved training history to {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Train SBLDM Diffusion')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                        help='Path to trained VAE checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to diffusion checkpoint to resume')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use')
    parser.add_argument('--no-intermediate-checkpoints', action='store_true',
                        help='Only save final checkpoint, not intermediate ones')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    dirs = setup_directories(config)
    set_seed(config['experiment']['seed'])
    
    # Device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load VAE
    print("\nLoading VAE...")
    vae = create_vae(config['vae'])
    vae_checkpoint = torch.load(args.vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print(f"Loaded VAE from {args.vae_checkpoint}")
    
    # Data
    print("\nLoading data...")
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=config['data']['data_dir'],
        resolution=config['data']['resolution'],
        batch_size=config['vae']['batch_size'],  # Use VAE batch size for encoding
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        augment=False,  # No augmentation for latent encoding
        max_samples=config['data'].get('max_samples')
    )
    
    # Create UNet
    print("\nCreating UNet...")
    unet = create_unet(config['unet'])
    print(f"UNet parameters: {unet.get_num_params():,}")
    
    # Trainer
    trainer = DiffusionTrainer(
        unet=unet,
        vae=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(
        total_steps=config['diffusion']['training_steps'],
        checkpoint_dir=dirs['checkpoint'],
        sample_dir=dirs['samples'],
        log_dir=dirs['log'],
        save_intermediate_checkpoints=not args.no_intermediate_checkpoints
    )


if __name__ == '__main__':
    main()
