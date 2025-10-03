"""
Training Utilities for Enhanced Polar Registration
Includes early stopping, learning rate scheduling, and model checkpointing
"""

import torch
import torch.nn as nn
import os
import logging
from typing import Dict, Any, Optional, Union
import numpy as np


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting
    """
    
    def __init__(self, 
                 patience: int = 20,
                 min_delta: float = 1e-6,
                 restore_best_weights: bool = True,
                 mode: str = 'min'):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
            mode: 'min' for minimizing, 'max' for maximizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
        self.wait = 0
        self.best_score = None
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, score: float, model: Optional[nn.Module] = None) -> bool:
        """
        Check if training should stop early
        
        Args:
            score: Current score to monitor
            model: Model to save weights from
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.wait = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if model is not None and self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False


class LearningRateScheduler:
    """
    Advanced learning rate scheduler with multiple strategies
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 scheduler_type: str = 'cosine',
                 **kwargs):
        """
        Initialize learning rate scheduler
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: Type of scheduler ('cosine', 'step', 'plateau', 'warmup')
            **kwargs: Additional arguments for scheduler
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.kwargs = kwargs
        
        self.scheduler = self._create_scheduler()
        self.warmup_epochs = kwargs.get('warmup_epochs', 0)
        self.warmup_factor = kwargs.get('warmup_factor', 0.1)
        
        if self.warmup_epochs > 0:
            self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
    
    def _create_scheduler(self):
        """Create the appropriate scheduler"""
        if self.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.kwargs.get('T_max', 100),
                eta_min=self.kwargs.get('eta_min', 1e-6)
            )
        elif self.scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.kwargs.get('step_size', 30),
                gamma=self.kwargs.get('gamma', 0.1)
            )
        elif self.scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.kwargs.get('mode', 'min'),
                factor=self.kwargs.get('factor', 0.5),
                patience=self.kwargs.get('patience', 10),
                verbose=True
            )
        elif self.scheduler_type == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.kwargs.get('gamma', 0.95)
            )
        else:
            return None
    
    def step(self, epoch: int, metric: Optional[float] = None):
        """Step the scheduler"""
        # Warmup phase
        if epoch < self.warmup_epochs:
            warmup_factor = self.warmup_factor + (1 - self.warmup_factor) * epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
        else:
            # Normal scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if metric is not None:
                        self.scheduler.step(metric)
                else:
                    self.scheduler.step()
    
    def get_last_lr(self):
        """Get the last learning rate"""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()
        else:
            return [group['lr'] for group in self.optimizer.param_groups]


class ModelCheckpoint:
    """
    Model checkpointing utility
    """
    
    def __init__(self, 
                 save_dir: str = './checkpoints',
                 save_best: bool = True,
                 save_last: bool = True,
                 save_every: int = 10,
                 monitor: str = 'val_loss',
                 mode: str = 'min'):
        """
        Initialize model checkpointing
        
        Args:
            save_dir: Directory to save checkpoints
            save_best: Whether to save best model
            save_last: Whether to save last model
            save_every: Save every N epochs
            monitor: Metric to monitor
            mode: 'min' or 'max' for monitoring
        """
        self.save_dir = save_dir
        self.save_best = save_best
        self.save_last = save_last
        self.save_every = save_every
        self.monitor = monitor
        self.mode = mode
        
        self.best_score = None
        self.last_epoch = -1
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, 
                       epoch: int,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            metrics: Current metrics
            is_best: Whether this is the best model
        """
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_score': self.best_score
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save best model
        if self.save_best and is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
            self.best_score = metrics.get(self.monitor, float('inf'))
        
        # Save last model
        if self.save_last:
            last_path = os.path.join(self.save_dir, 'last_model.pth')
            torch.save(checkpoint, last_path)
        
        # Save periodic checkpoints
        if epoch % self.save_every == 0:
            periodic_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, periodic_path)
            self.logger.info(f"Checkpoint saved to {periodic_path}")
        
        self.last_epoch = epoch
    
    def load_checkpoint(self, 
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {checkpoint['epoch']}")
        
        return checkpoint


class GradientAccumulator:
    """
    Gradient accumulation utility for large batch training
    """
    
    def __init__(self, accumulation_steps: int = 1):
        """
        Initialize gradient accumulator
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def should_step(self) -> bool:
        """Check if optimizer should step"""
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0
    
    def reset(self):
        """Reset step counter"""
        self.step_count = 0


class MetricsTracker:
    """
    Utility for tracking and computing training metrics
    """
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def compute_averages(self) -> Dict[str, float]:
        """Compute average metrics"""
        averages = {}
        for key in self.metrics:
            averages[key] = self.metrics[key] / self.counts[key]
        return averages
    
    def reset(self):
        """Reset metrics"""
        self.metrics.clear()
        self.counts.clear()


class ModelEMA:
    """
    Exponential Moving Average of model parameters
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Initialize EMA
        
        Args:
            model: Model to create EMA for
            decay: EMA decay factor
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update EMA parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self, model: nn.Module):
        """Apply shadow parameters to model"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Restore original parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup.clear()
