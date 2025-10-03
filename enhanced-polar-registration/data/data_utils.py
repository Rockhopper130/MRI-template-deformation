"""
Data Utilities for Enhanced Polar Registration
Includes coordinate conversion, normalization, and preprocessing functions
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union
import logging


def create_polar_coordinates(size: Tuple[int, int, int], 
                           device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Create polar coordinate grid for given size
    
    Args:
        size: (D, H, W) - depth, height, width
        device: Device to create tensor on
    
    Returns:
        polar_coords: [D, H, W, 6] - (x, y, z, rho, theta, phi)
    """
    D, H, W = size
    
    # Create coordinate grids
    z = torch.linspace(-1, 1, D, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    
    # Convert to polar coordinates
    rho = torch.sqrt(xx**2 + yy**2 + zz**2)
    theta = torch.acos(torch.clamp(zz / (rho + 1e-8), -1, 1))
    phi = torch.atan2(yy, xx)
    
    # Stack coordinates
    polar_coords = torch.stack([xx, yy, zz, rho, theta, phi], dim=-1)
    
    return polar_coords


def cartesian_to_polar(xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert Cartesian coordinates to polar coordinates
    
    Args:
        xyz: [..., 3] - Cartesian coordinates (x, y, z)
    
    Returns:
        rho, theta, phi: Polar coordinates
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    
    rho = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(torch.clamp(z / (rho + 1e-8), -1, 1))
    phi = torch.atan2(y, x)
    
    return rho, theta, phi


def polar_to_cartesian(rho: torch.Tensor, 
                      theta: torch.Tensor, 
                      phi: torch.Tensor) -> torch.Tensor:
    """
    Convert polar coordinates to Cartesian coordinates
    
    Args:
        rho: Radial distance
        theta: Polar angle
        phi: Azimuthal angle
    
    Returns:
        xyz: [..., 3] - Cartesian coordinates
    """
    x = rho * torch.sin(theta) * torch.cos(phi)
    y = rho * torch.sin(theta) * torch.sin(phi)
    z = rho * torch.cos(theta)
    
    return torch.stack([x, y, z], dim=-1)


def normalize_segmentation(segmentation: torch.Tensor) -> torch.Tensor:
    """
    Normalize segmentation to ensure proper one-hot encoding
    
    Args:
        segmentation: [C, D, H, W] - segmentation map
    
    Returns:
        normalized: [C, D, H, W] - normalized segmentation
    """
    # Ensure values are in [0, 1]
    segmentation = torch.clamp(segmentation, 0, 1)
    
    # Normalize to sum to 1 across channels
    channel_sum = torch.sum(segmentation, dim=0, keepdim=True)
    channel_sum = torch.clamp(channel_sum, min=1e-8)
    normalized = segmentation / channel_sum
    
    return normalized


def compute_center_of_mass(segmentation: torch.Tensor) -> torch.Tensor:
    """
    Compute center of mass of segmentation
    
    Args:
        segmentation: [C, D, H, W] - segmentation map
    
    Returns:
        center: [3] - center of mass (z, y, x)
    """
    C, D, H, W = segmentation.shape
    
    # Create coordinate grids
    z = torch.arange(D, device=segmentation.device, dtype=torch.float32).view(D, 1, 1)
    y = torch.arange(H, device=segmentation.device, dtype=torch.float32).view(1, H, 1)
    x = torch.arange(W, device=segmentation.device, dtype=torch.float32).view(1, 1, W)
    
    # Compute total mass
    total_mass = segmentation.sum()
    
    if total_mass <= 0:
        return torch.tensor([D/2.0, H/2.0, W/2.0], device=segmentation.device)
    
    # Compute center of mass
    cx = (segmentation * x).sum() / total_mass
    cy = (segmentation * y).sum() / total_mass
    cz = (segmentation * z).sum() / total_mass
    
    return torch.stack([cz, cy, cx], dim=0)


def compute_bounding_box(segmentation: torch.Tensor, 
                        margin: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute bounding box of segmentation
    
    Args:
        segmentation: [C, D, H, W] - segmentation map
        margin: Margin to add around bounding box
    
    Returns:
        min_coords, max_coords: [3] - minimum and maximum coordinates
    """
    # Find non-zero voxels
    mask = segmentation.sum(dim=0) > 0
    
    if not mask.any():
        # Return full volume if no segmentation
        D, H, W = segmentation.shape[1:]
        return torch.zeros(3, device=segmentation.device), torch.tensor([D, H, W], device=segmentation.device)
    
    # Find bounding box
    coords = torch.nonzero(mask, as_tuple=False).float()
    min_coords = coords.min(dim=0)[0]
    max_coords = coords.max(dim=0)[0]
    
    # Add margin
    D, H, W = segmentation.shape[1:]
    margin_size = torch.tensor([D, H, W], device=segmentation.device) * margin
    min_coords = torch.clamp(min_coords - margin_size, 0)
    max_coords = torch.clamp(max_coords + margin_size, 0, torch.tensor([D, H, W], device=segmentation.device))
    
    return min_coords, max_coords


def crop_to_bounding_box(segmentation: torch.Tensor, 
                        min_coords: torch.Tensor, 
                        max_coords: torch.Tensor) -> torch.Tensor:
    """
    Crop segmentation to bounding box
    
    Args:
        segmentation: [C, D, H, W] - segmentation map
        min_coords: [3] - minimum coordinates
        max_coords: [3] - maximum coordinates
    
    Returns:
        cropped: [C, D', H', W'] - cropped segmentation
    """
    min_coords = min_coords.long()
    max_coords = max_coords.long()
    
    cropped = segmentation[:, 
                          min_coords[0]:max_coords[0],
                          min_coords[1]:max_coords[1],
                          min_coords[2]:max_coords[2]]
    
    return cropped


def pad_to_size(segmentation: torch.Tensor, 
                target_size: Tuple[int, int, int],
                pad_value: float = 0.0) -> torch.Tensor:
    """
    Pad segmentation to target size
    
    Args:
        segmentation: [C, D, H, W] - segmentation map
        target_size: (D', H', W') - target size
        pad_value: Value to use for padding
    
    Returns:
        padded: [C, D', H', W'] - padded segmentation
    """
    C, D, H, W = segmentation.shape
    D_target, H_target, W_target = target_size
    
    # Compute padding
    pad_d = max(0, D_target - D)
    pad_h = max(0, H_target - H)
    pad_w = max(0, W_target - W)
    
    # Pad symmetrically
    pad_d_before = pad_d // 2
    pad_d_after = pad_d - pad_d_before
    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before
    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before
    
    padded = F.pad(segmentation, 
                   (pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after),
                   value=pad_value)
    
    return padded


def resize_segmentation(segmentation: torch.Tensor, 
                       target_size: Tuple[int, int, int],
                       mode: str = 'nearest') -> torch.Tensor:
    """
    Resize segmentation to target size
    
    Args:
        segmentation: [C, D, H, W] - segmentation map
        target_size: (D', H', W') - target size
        mode: Interpolation mode ('nearest', 'trilinear')
    
    Returns:
        resized: [C, D', H', W'] - resized segmentation
    """
    resized = F.interpolate(
        segmentation.unsqueeze(0),
        size=target_size,
        mode=mode,
        align_corners=False if mode == 'trilinear' else None
    ).squeeze(0)
    
    return resized


def compute_segmentation_metrics(pred: torch.Tensor, 
                                target: torch.Tensor) -> dict:
    """
    Compute segmentation metrics
    
    Args:
        pred: [C, D, H, W] - predicted segmentation
        target: [C, D, H, W] - target segmentation
    
    Returns:
        metrics: Dictionary of computed metrics
    """
    # Convert to class labels
    pred_labels = torch.argmax(pred, dim=0)
    target_labels = torch.argmax(target, dim=0)
    
    # Compute Dice coefficient for each class
    dice_scores = []
    for c in range(pred.shape[0]):
        pred_c = (pred_labels == c).float()
        target_c = (target_labels == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = 2 * intersection / union
        else:
            dice = torch.tensor(1.0 if pred_c.sum() == 0 and target_c.sum() == 0 else 0.0)
        
        dice_scores.append(dice.item())
    
    # Compute overall metrics
    metrics = {
        'dice_scores': dice_scores,
        'mean_dice': np.mean(dice_scores),
        'overall_accuracy': (pred_labels == target_labels).float().mean().item()
    }
    
    return metrics


def create_distance_transform(segmentation: torch.Tensor) -> torch.Tensor:
    """
    Create distance transform of segmentation
    
    Args:
        segmentation: [C, D, H, W] - segmentation map
    
    Returns:
        distance_transform: [C, D, H, W] - distance transform
    """
    C, D, H, W = segmentation.shape
    distance_transform = torch.zeros_like(segmentation)
    
    for c in range(C):
        # Convert to binary
        binary = (segmentation[c] > 0.5).float()
        
        # Compute distance transform (simplified version)
        # In practice, you might want to use a more efficient implementation
        coords = torch.nonzero(binary, as_tuple=False).float()
        
        if coords.numel() > 0:
            # Create coordinate grids
            z = torch.arange(D, device=segmentation.device, dtype=torch.float32).view(D, 1, 1)
            y = torch.arange(H, device=segmentation.device, dtype=torch.float32).view(1, H, 1)
            x = torch.arange(W, device=segmentation.device, dtype=torch.float32).view(1, 1, W)
            
            # Compute distances
            distances = torch.zeros(D, H, W, device=segmentation.device)
            for coord in coords:
                cz, cy, cx = coord
                dist = torch.sqrt((z - cz)**2 + (y - cy)**2 + (x - cx)**2)
                distances = torch.min(distances, dist)
            
            distance_transform[c] = distances
    
    return distance_transform


def apply_gaussian_smoothing(segmentation: torch.Tensor, 
                           sigma: float = 1.0) -> torch.Tensor:
    """
    Apply Gaussian smoothing to segmentation
    
    Args:
        segmentation: [C, D, H, W] - segmentation map
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        smoothed: [C, D, H, W] - smoothed segmentation
    """
    # Create Gaussian kernel
    kernel_size = int(2 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create 3D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32, device=segmentation.device)
    coords = coords - kernel_size // 2
    
    z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
    kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution
    smoothed = F.conv3d(
        segmentation.unsqueeze(0),
        kernel,
        padding=kernel_size//2
    ).squeeze(0)
    
    return smoothed
