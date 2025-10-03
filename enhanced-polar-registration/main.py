"""
Enhanced Polar Registration - Main Training Script
Combines all enhanced components for improved segmentation map registration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import logging
from datetime import datetime
import json
import numpy as np

# Import enhanced components
from models import EnhancedUNet, PolarSpatialTransformer
from losses import EnhancedCompositeLoss
from data import EnhancedSegDataset, MultiScaleDataset, BalancedDataset
from training import EnhancedTrainer
from utils.config import load_config
from utils.visualization import plot_training_curves, visualize_registration


def setup_logging(log_dir: str, log_level: str = 'INFO'):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f'enhanced_training_{timestamp}.log')
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("Enhanced Polar Registration Training")
    logger.info("="*80)
    logger.info(f"Log file: {log_filename}")
    
    return logger, log_filename


def create_model(config: dict, device: torch.device) -> nn.Module:
    """Create enhanced UNet model"""
    model = EnhancedUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
        num_scales=config['model']['num_scales'],
        image_size=tuple(config['data']['target_size']),
        use_polar_processing=config['model']['use_polar_processing'],
        use_attention=config['model']['use_attention']
    ).to(device)
    
    return model


def create_spatial_transformer(config: dict, device: torch.device) -> nn.Module:
    """Create enhanced spatial transformer"""
    transformer = PolarSpatialTransformer(
        size=tuple(config['data']['target_size']),
        device=device,
        use_polar_coords=config['model']['use_polar_processing'],
        interpolation_mode=config['model']['interpolation_mode']
    ).to(device)
    
    return transformer


def create_loss_function(config: dict) -> nn.Module:
    """Create enhanced loss function"""
    loss_fn = EnhancedCompositeLoss(
        dice_weight=config['loss']['dice_weight'],
        ce_weight=config['loss']['ce_weight'],
        bending_weight=config['loss']['bending_weight'],
        jacobian_weight=config['loss']['jacobian_weight'],
        polar_weight=config['loss']['polar_weight'],
        anatomical_weight=config['loss']['anatomical_weight'],
        gradient_weight=config['loss']['gradient_weight']
    )
    
    return loss_fn


def create_datasets(config: dict):
    """Create training and validation datasets"""
    # Create base dataset
    train_dataset = EnhancedSegDataset(
        data_list_file=config['data']['train_txt'],
        template_path=config['data']['template_path'],
        target_size=tuple(config['data']['target_size']),
        use_polar_coords=config['model']['use_polar_processing'],
        use_augmentation=config['data']['use_augmentation'],
        augmentation_prob=config['data']['augmentation_prob'],
        cache_data=config['data']['cache_data']
    )
    
    val_dataset = EnhancedSegDataset(
        data_list_file=config['data']['val_txt'],
        template_path=config['data']['template_path'],
        target_size=tuple(config['data']['target_size']),
        use_polar_coords=config['model']['use_polar_processing'],
        use_augmentation=False,  # No augmentation for validation
        cache_data=config['data']['cache_data']
    )
    
    # Apply dataset enhancements if specified
    if config['data'].get('use_multi_scale', False):
        train_dataset = MultiScaleDataset(train_dataset, scales=config['data']['multi_scale_levels'])
        val_dataset = MultiScaleDataset(val_dataset, scales=config['data']['multi_scale_levels'])
    
    if config['data'].get('use_balanced_sampling', False):
        train_dataset = BalancedDataset(
            config['data']['train_txt'],
            config['data']['template_path'],
            tuple(config['data']['target_size']),
            balance_classes=True
        )
    
    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, config: dict):
    """Create data loaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Create optimizer"""
    optimizer_type = config['training']['optimizer']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=config['training'].get('betas', (0.9, 0.999))
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=config['training'].get('betas', (0.9, 0.999))
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=config['training'].get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Enhanced Polar Registration Training')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to train on')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger, log_filename = setup_logging(config['training']['log_dir'], args.log_level)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Log configuration
    logger.info("Training Configuration:")
    logger.info(json.dumps(config, indent=2))
    
    # Create model components
    logger.info("Creating model components...")
    model = create_model(config, device)
    spatial_transformer = create_spatial_transformer(config, device)
    loss_fn = create_loss_function(config)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} trainable parameters")
    
    # Create datasets and data loaders
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Training batches per epoch: {len(train_loader)}")
    logger.info(f"Validation batches per epoch: {len(val_loader)}")
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    logger.info(f"Optimizer: {config['training']['optimizer']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    
    # Create trainer
    trainer = EnhancedTrainer(
        model=model,
        spatial_transformer=spatial_transformer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        config=config['training']
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = trainer.model_checkpoint.load_checkpoint(
            model, optimizer, trainer.scheduler, args.resume
        )
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint.get('best_score', float('inf'))
    
    # Start training
    logger.info("Starting training...")
    training_results = trainer.train()
    
    # Save final results
    results_path = os.path.join(config['training']['save_dir'], 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2)
    
    logger.info(f"Training results saved to: {results_path}")
    
    # Plot training curves
    if config['training'].get('plot_curves', True):
        plot_path = os.path.join(config['training']['save_dir'], 'training_curves.png')
        plot_training_curves(
            training_results['train_losses'],
            training_results['val_losses'],
            save_path=plot_path
        )
        logger.info(f"Training curves saved to: {plot_path}")
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
