"""
SBLDM VAE Training Script
Trains the VAE for latent space compression.

Usage:
    python train_vae.py --config configs/config_full.yaml
    python train_vae.py --config configs/config_debug.yaml
"""

import os
import sys
import argparse
import yaml
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import get_dataloaders
from models.vae import VAE, create_vae


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_directories(config: dict) -> dict:
    """Create output directories."""
    dirs = {
        'output': Path(config['experiment']['output_dir']),
        'log': Path(config['experiment']['log_dir']),
        'checkpoint': Path(config['experiment']['checkpoint_dir'])
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return {k: str(v) for k, v in dirs.items()}


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class VAETrainer:
    """
    VAE Training manager.
    
    Handles:
    - Training loop with progress tracking
    - Validation
    - Checkpointing
    - Mixed precision training
    - KL annealing
    """
    
    def __init__(
        self,
        model: VAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        vae_config = config['vae']
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=vae_config['learning_rate'],
            weight_decay=vae_config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        if vae_config.get('scheduler') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=vae_config['epochs'],
                eta_min=vae_config['learning_rate'] * 0.01
            )
        else:
            self.scheduler = None
        
        # Mixed precision
        self.use_amp = config['hardware'].get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # KL annealing
        self.kl_weight = vae_config.get('kl_weight', 1e-5)
        self.kl_anneal = vae_config.get('kl_anneal', False)
        self.kl_anneal_epochs = vae_config.get('kl_anneal_epochs', 20)
        
        # Gradient clipping
        self.grad_clip = vae_config.get('gradient_clip', 1.0)
        
        # Tracking
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    def get_kl_weight(self) -> float:
        """Get current KL weight with optional annealing."""
        if not self.kl_anneal:
            return self.kl_weight
        
        # Linear annealing from 0 to kl_weight
        anneal_factor = min(1.0, self.epoch / self.kl_anneal_epochs)
        return self.kl_weight * anneal_factor
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0
        
        kl_weight = self.get_kl_weight()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            x = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            with autocast(enabled=self.use_amp):
                recon, mean, logvar = self.model(x)
                loss, loss_dict = self.model.loss_function(
                    x, recon, mean, logvar, kl_weight
                )
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            # Track losses
            total_loss += loss_dict['loss']
            total_recon += loss_dict['recon_loss']
            total_kl += loss_dict['kl_loss']
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['loss']:.4f}",
                'recon': f"{loss_dict['recon_loss']:.4f}",
                'kl': f"{loss_dict['kl_loss']:.4f}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon / num_batches,
            'kl_loss': total_kl / num_batches,
            'kl_weight': kl_weight
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0
        
        for batch in self.val_loader:
            x = batch.to(self.device)
            
            with autocast(enabled=self.use_amp):
                recon, mean, logvar = self.model(x, sample_posterior=False)
                loss, loss_dict = self.model.loss_function(
                    x, recon, mean, logvar, self.kl_weight
                )
            
            total_loss += loss_dict['loss']
            total_recon += loss_dict['recon_loss']
            total_kl += loss_dict['kl_loss']
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon / num_batches,
            'kl_loss': total_kl / num_batches
        }
    
    def save_checkpoint(
        self,
        path: str,
        is_best: bool = False
    ):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
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
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Resumed from epoch {self.epoch}")
    
    def train(
        self,
        epochs: int,
        checkpoint_dir: str,
        log_dir: str
    ):
        """Full training loop."""
        print(f"\nStarting VAE training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Model parameters: {self.model.get_num_params():,}")
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Track history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start
            
            # Print summary
            print(f"\nEpoch {epoch}/{epochs-1} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Recon: {train_metrics['recon_loss']:.4f}, "
                  f"KL: {train_metrics['kl_loss']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Recon: {val_metrics['recon_loss']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}, "
                  f"KL weight: {train_metrics['kl_weight']:.6f}")
            
            # Checkpointing
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                print(f"  New best validation loss!")
            
            checkpoint_path = os.path.join(checkpoint_dir, f'vae_epoch_{epoch}.pt')
            self.save_checkpoint(checkpoint_path, is_best=is_best)
            
            # Save latest
            latest_path = os.path.join(checkpoint_dir, 'vae_latest.pt')
            self.save_checkpoint(latest_path)
        
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
        
        # Save final history
        history_path = os.path.join(log_dir, 'vae_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train SBLDM VAE')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (auto-detected if not specified)')
    
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
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config['data']['data_dir'],
        resolution=config['data']['resolution'],
        batch_size=config['vae']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        augment=True,
        max_samples=config['data'].get('max_samples')
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Model
    print("\nCreating VAE...")
    vae = create_vae(config['vae'])
    print(f"VAE parameters: {vae.get_num_params():,}")
    print(f"Downsample factor: {vae.downsample_factor}x")
    
    # Trainer
    trainer = VAETrainer(
        model=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif config['checkpoint'].get('resume'):
        trainer.load_checkpoint(config['checkpoint']['resume'])
    
    # Train
    trainer.train(
        epochs=config['vae']['epochs'],
        checkpoint_dir=dirs['checkpoint'],
        log_dir=dirs['log']
    )


if __name__ == '__main__':
    main()
