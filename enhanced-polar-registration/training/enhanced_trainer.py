"""
Enhanced Training Module for Polar Coordinate Registration
Includes advanced training strategies, monitoring, and optimization techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import numpy as np
import logging
import time
import os
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from models import EnhancedUNet, PolarSpatialTransformer
from losses import EnhancedCompositeLoss
from data import EnhancedSegDataset
from training.training_utils import EarlyStopping, LearningRateScheduler, ModelCheckpoint


class EnhancedTrainer:
    """
    Enhanced trainer with advanced training strategies and monitoring
    """
    
    def __init__(self,
                 model: nn.Module,
                 spatial_transformer: nn.Module,
                 loss_fn: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 config: Dict[str, Any]):
        """
        Initialize enhanced trainer
        
        Args:
            model: Registration model
            spatial_transformer: Spatial transformer module
            loss_fn: Loss function
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.spatial_transformer = spatial_transformer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Training components
        self.scaler = GradScaler()
        self.scheduler = self._create_scheduler()
        
        # Move model to device and ensure proper dtype
        self.model = self.model.to(device)
        self.spatial_transformer = self.spatial_transformer.to(device)
        self.loss_fn = self.loss_fn.to(device)
        
        # Training utilities
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 20),
            min_delta=config.get('early_stopping_min_delta', 1e-6)
        )
        
        self.model_checkpoint = ModelCheckpoint(
            save_dir=config.get('save_dir', './checkpoints'),
            save_best=True,
            save_last=True
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Logging
        self.setup_logging()
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.init(
                project=config.get('wandb_project', 'polar-registration'),
                config=config,
                name=config.get('experiment_name', 'enhanced-polar-reg')
            )
        elif config.get('use_wandb', False) and not WANDB_AVAILABLE:
            self.logger.warning("Wandb requested but not available. Install wandb to enable experiment tracking.")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.get('save_dir', './checkpoints'), 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        
        train_pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]', leave=False)
        
        for batch_idx, batch in enumerate(train_pbar):
            # Move data to device
            moving = batch['moving'].to(self.device)
            fixed = batch['fixed'].to(self.device)
            polar_coords = batch.get('polar_coords', None)
            if polar_coords is not None:
                polar_coords = polar_coords.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            # Create input
            input_tensor = torch.cat([moving, fixed], dim=1)
            
            # Predict deformation field
            deformation_field = self.model(input_tensor, polar_coords)
            
            # Apply spatial transformation
            warped = self.spatial_transformer(moving, deformation_field, polar_coords)
            
            # Compute loss
            loss, loss_dict = self.loss_fn(warped, fixed, deformation_field, polar_coords)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.get('grad_clip_norm', 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.get('grad_clip_norm', 1.0)
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log batch metrics
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self.logger.info(f'Batch {batch_idx}: Loss = {loss.item():.6f}')
        
        # Average metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_components = {key: value / len(self.train_loader) for key, value in loss_components.items()}
        
        return {'total_loss': avg_loss, **avg_components}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        loss_components = {}
        
        val_pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]', leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                # Move data to device
                moving = batch['moving'].to(self.device)
                fixed = batch['fixed'].to(self.device)
                polar_coords = batch.get('polar_coords', None)
                if polar_coords is not None:
                    polar_coords = polar_coords.to(self.device)
                
                # Forward pass
                # Create input
                input_tensor = torch.cat([moving, fixed], dim=1)
                
                # Predict deformation field
                deformation_field = self.model(input_tensor, polar_coords)
                
                # Apply spatial transformation
                warped = self.spatial_transformer(moving, deformation_field, polar_coords)
                
                # Compute loss
                loss, loss_dict = self.loss_fn(warped, fixed, deformation_field, polar_coords)
                
                # Update metrics
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item()
                
                # Update progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_components = {key: value / len(self.val_loader) for key, value in loss_components.items()}
        
        return {'total_loss': avg_loss, **avg_components}
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop"""
        self.logger.info("Starting enhanced training...")
        self.logger.info(f"Training configuration: {self.config}")
        
        start_time = time.time()
        
        for epoch in range(self.config.get('epochs', 100)):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['total_loss'])
            
            # Validation phase
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics['total_loss'])
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.logger.info(f"Epoch {epoch+1}/{self.config.get('epochs', 100)}:")
            self.logger.info(f"  Train Loss: {train_metrics['total_loss']:.6f}")
            self.logger.info(f"  Val Loss:   {val_metrics['total_loss']:.6f}")
            self.logger.info(f"  Learning Rate: {current_lr:.2e}")
            self.logger.info(f"  Epoch Time: {epoch_time:.2f}s")
            
            # Log detailed metrics
            for key, value in train_metrics.items():
                if key != 'total_loss':
                    self.logger.info(f"  Train {key}: {value:.6f}")
            
            for key, value in val_metrics.items():
                if key != 'total_loss':
                    self.logger.info(f"  Val {key}: {value:.6f}")
            
            # Log to wandb
            if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'val_loss': val_metrics['total_loss'],
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time,
                    **{f'train_{k}': v for k, v in train_metrics.items() if k != 'total_loss'},
                    **{f'val_{k}': v for k, v in val_metrics.items() if k != 'total_loss'}
                })
            
            # Model checkpointing
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
            
            self.model_checkpoint.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=val_metrics,
                is_best=is_best
            )
            
            # Early stopping
            if self.early_stopping(val_metrics['total_loss']):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                self.logger.info(f"  GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info("Training completed!")
        self.logger.info(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Close wandb
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.finish()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0.0
        loss_components = {}
        
        self.logger.info("Evaluating model on test set...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                # Move data to device
                moving = batch['moving'].to(self.device)
                fixed = batch['fixed'].to(self.device)
                polar_coords = batch.get('polar_coords', None)
                if polar_coords is not None:
                    polar_coords = polar_coords.to(self.device)
                
                # Forward pass
                # Create input
                input_tensor = torch.cat([moving, fixed], dim=1)
                
                # Predict deformation field
                deformation_field = self.model(input_tensor, polar_coords)
                
                # Apply spatial transformation
                warped = self.spatial_transformer(moving, deformation_field, polar_coords)
                
                # Compute loss
                loss, loss_dict = self.loss_fn(warped, fixed, deformation_field, polar_coords)
                
                # Update metrics
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item()
        
        # Average metrics
        avg_loss = total_loss / len(test_loader)
        avg_components = {key: value / len(test_loader) for key, value in loss_components.items()}
        
        self.logger.info(f"Test Results:")
        self.logger.info(f"  Total Loss: {avg_loss:.6f}")
        for key, value in avg_components.items():
            self.logger.info(f"  {key}: {value:.6f}")
        
        return {'total_loss': avg_loss, **avg_components}
