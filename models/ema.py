"""
Exponential Moving Average (EMA) for model weights.
Helps stabilize training and improve sample quality.
"""

import torch
import torch.nn as nn
from typing import Optional
import copy


class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Maintains a shadow copy of model weights that are updated
    as: shadow = decay * shadow + (1 - decay) * current
    
    Usage:
        ema = EMA(model, decay=0.9999)
        
        # During training
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            ema.update()
        
        # For evaluation
        with ema.average_parameters():
            samples = model.sample()
    
    Args:
        model: PyTorch model
        decay: EMA decay rate (0.9999 is common)
        device: Device to store EMA weights
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.decay = decay
        self.device = device
        
        # Create shadow parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device is not None:
                    self.shadow[name] = self.shadow[name].to(device)
    
    def update(self):
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def average_parameters(self):
        """Context manager for using EMA parameters."""
        class AverageContext:
            def __init__(self, ema):
                self.ema = ema
            
            def __enter__(self):
                self.ema.apply_shadow()
                return self
            
            def __exit__(self, *args):
                self.ema.restore()
        
        return AverageContext(self)
    
    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
