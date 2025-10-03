"""
Visualization utilities for Enhanced Polar Registration
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from typing import List, Optional, Tuple
import logging


def plot_training_curves(train_losses: List[float],
                        val_losses: List[float],
                        save_path: Optional[str] = None,
                        title: str = "Training Curves"):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add best validation loss annotation
    best_epoch = np.argmin(val_losses) + 1
    best_loss = min(val_losses)
    plt.annotate(f'Best: {best_loss:.4f} at epoch {best_epoch}',
                xy=(best_epoch, best_loss),
                xytext=(best_epoch + 5, best_loss + 0.01),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Training curves saved to: {save_path}")
    
    plt.show()


def visualize_registration(moving: torch.Tensor,
                          fixed: torch.Tensor,
                          warped: torch.Tensor,
                          deformation_field: torch.Tensor,
                          save_path: Optional[str] = None,
                          slice_idx: int = 64):
    """
    Visualize registration results
    
    Args:
        moving: Moving image [C, D, H, W]
        fixed: Fixed image [C, D, H, W]
        warped: Warped image [C, D, H, W]
        deformation_field: Deformation field [3, D, H, W]
        save_path: Path to save visualization
        slice_idx: Slice index to visualize
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Convert to numpy and select slice
    moving_slice = moving[:, slice_idx, :, :].cpu().numpy()
    fixed_slice = fixed[:, slice_idx, :, :].cpu().numpy()
    warped_slice = warped[:, slice_idx, :, :].cpu().numpy()
    deformation_slice = deformation_field[:, slice_idx, :, :].cpu().numpy()
    
    # Plot moving image (first channel)
    axes[0, 0].imshow(moving_slice[0], cmap='gray')
    axes[0, 0].set_title('Moving Image')
    axes[0, 0].axis('off')
    
    # Plot fixed image (first channel)
    axes[0, 1].imshow(fixed_slice[0], cmap='gray')
    axes[0, 1].set_title('Fixed Image')
    axes[0, 1].axis('off')
    
    # Plot warped image (first channel)
    axes[0, 2].imshow(warped_slice[0], cmap='gray')
    axes[0, 2].set_title('Warped Image')
    axes[0, 2].axis('off')
    
    # Plot difference
    diff = np.abs(warped_slice[0] - fixed_slice[0])
    axes[0, 3].imshow(diff, cmap='hot')
    axes[0, 3].set_title('Difference')
    axes[0, 3].axis('off')
    
    # Plot deformation field components
    axes[1, 0].imshow(deformation_slice[0], cmap='RdBu')
    axes[1, 0].set_title('Deformation X')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(deformation_slice[1], cmap='RdBu')
    axes[1, 1].set_title('Deformation Y')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(deformation_slice[2], cmap='RdBu')
    axes[1, 2].set_title('Deformation Z')
    axes[1, 2].axis('off')
    
    # Plot deformation magnitude
    deformation_mag = np.sqrt(np.sum(deformation_slice**2, axis=0))
    axes[1, 3].imshow(deformation_mag, cmap='viridis')
    axes[1, 3].set_title('Deformation Magnitude')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Registration visualization saved to: {save_path}")
    
    plt.show()


def plot_loss_components(loss_dict: dict,
                        save_path: Optional[str] = None,
                        title: str = "Loss Components"):
    """
    Plot individual loss components
    
    Args:
        loss_dict: Dictionary of loss components
        save_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    num_components = len(loss_dict)
    cols = 3
    rows = (num_components + cols - 1) // cols
    
    for i, (name, values) in enumerate(loss_dict.items()):
        plt.subplot(rows, cols, i + 1)
        plt.plot(values, linewidth=2)
        plt.title(f'{name.replace("_", " ").title()}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Loss components plot saved to: {save_path}")
    
    plt.show()


def visualize_polar_coordinates(polar_coords: torch.Tensor,
                               save_path: Optional[str] = None,
                               slice_idx: int = 64):
    """
    Visualize polar coordinate system
    
    Args:
        polar_coords: Polar coordinates [6, D, H, W]
        save_path: Path to save visualization
        slice_idx: Slice index to visualize
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert to numpy and select slice
    coords_slice = polar_coords[:, slice_idx, :, :].cpu().numpy()
    
    # Plot Cartesian coordinates
    axes[0, 0].imshow(coords_slice[0], cmap='RdBu')
    axes[0, 0].set_title('X Coordinate')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(coords_slice[1], cmap='RdBu')
    axes[0, 1].set_title('Y Coordinate')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(coords_slice[2], cmap='RdBu')
    axes[0, 2].set_title('Z Coordinate')
    axes[0, 2].axis('off')
    
    # Plot polar coordinates
    axes[1, 0].imshow(coords_slice[3], cmap='viridis')
    axes[1, 0].set_title('Rho (Radial Distance)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(coords_slice[4], cmap='plasma')
    axes[1, 1].set_title('Theta (Polar Angle)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(coords_slice[5], cmap='twilight')
    axes[1, 2].set_title('Phi (Azimuthal Angle)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Polar coordinates visualization saved to: {save_path}")
    
    plt.show()


def create_registration_gif(moving: torch.Tensor,
                           fixed: torch.Tensor,
                           warped: torch.Tensor,
                           save_path: str,
                           axis: int = 0):
    """
    Create GIF showing registration results across slices
    
    Args:
        moving: Moving image [C, D, H, W]
        fixed: Fixed image [C, D, H, W]
        warped: Warped image [C, D, H, W]
        save_path: Path to save GIF
        axis: Axis to slice along (0, 1, or 2)
    """
    try:
        import imageio
    except ImportError:
        logging.error("imageio not available for GIF creation")
        return
    
    # Convert to numpy
    moving_np = moving[0].cpu().numpy()  # First channel
    fixed_np = fixed[0].cpu().numpy()
    warped_np = warped[0].cpu().numpy()
    
    # Normalize to [0, 1]
    moving_np = (moving_np - moving_np.min()) / (moving_np.max() - moving_np.min())
    fixed_np = (fixed_np - fixed_np.min()) / (fixed_np.max() - fixed_np.min())
    warped_np = (warped_np - warped_np.min()) / (warped_np.max() - warped_np.min())
    
    # Create frames
    frames = []
    num_slices = moving_np.shape[axis]
    
    for i in range(num_slices):
        if axis == 0:
            moving_slice = moving_np[i, :, :]
            fixed_slice = fixed_np[i, :, :]
            warped_slice = warped_np[i, :, :]
        elif axis == 1:
            moving_slice = moving_np[:, i, :]
            fixed_slice = fixed_np[:, i, :]
            warped_slice = warped_np[:, i, :]
        else:
            moving_slice = moving_np[:, :, i]
            fixed_slice = fixed_np[:, :, i]
            warped_slice = warped_np[:, :, i]
        
        # Create combined frame
        frame = np.concatenate([moving_slice, fixed_slice, warped_slice], axis=1)
        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)
    
    # Save GIF
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, duration=0.1)
    logging.info(f"Registration GIF saved to: {save_path}")


def plot_feature_maps(feature_maps: dict,
                     save_path: Optional[str] = None,
                     slice_idx: int = 64):
    """
    Plot feature maps from different layers
    
    Args:
        feature_maps: Dictionary of feature maps
        save_path: Path to save plot
        slice_idx: Slice index to visualize
    """
    num_layers = len(feature_maps)
    cols = 4
    rows = (num_layers + cols - 1) // cols
    
    plt.figure(figsize=(16, 4 * rows))
    
    for i, (name, features) in enumerate(feature_maps.items()):
        plt.subplot(rows, cols, i + 1)
        
        # Select slice and channel
        if len(features.shape) == 5:  # [B, C, D, H, W]
            feature_slice = features[0, 0, slice_idx, :, :].cpu().numpy()
        elif len(features.shape) == 4:  # [C, D, H, W]
            feature_slice = features[0, slice_idx, :, :].cpu().numpy()
        else:
            continue
        
        plt.imshow(feature_slice, cmap='viridis')
        plt.title(f'{name}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Feature maps plot saved to: {save_path}")
    
    plt.show()
