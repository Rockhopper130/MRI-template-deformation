"""
Advanced Data Augmentation for Polar Coordinate Registration
Includes polar-aware augmentations and geometric transformations
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Tuple, Optional, Any
import logging


class PolarDataAugmentation:
    """
    Advanced data augmentation with polar coordinate awareness
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int, int] = (128, 128, 128),
                 augmentation_prob: float = 0.5,
                 rotation_range: Tuple[float, float] = (-15, 15),
                 translation_range: Tuple[float, float] = (-0.1, 0.1),
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 noise_std: float = 0.01,
                 elastic_alpha: float = 1.0,
                 elastic_sigma: float = 3.0):
        """
        Initialize polar data augmentation
        
        Args:
            target_size: Target size for augmentation
            augmentation_prob: Probability of applying augmentation
            rotation_range: Range for rotation angles (degrees)
            translation_range: Range for translation (fraction of image size)
            scale_range: Range for scaling
            noise_std: Standard deviation for Gaussian noise
            elastic_alpha: Alpha parameter for elastic deformation
            elastic_sigma: Sigma parameter for elastic deformation
        """
        self.target_size = target_size
        self.augmentation_prob = augmentation_prob
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        
        # Create coordinate grids for elastic deformation
        self._create_elastic_grids()
    
    def _create_elastic_grids(self):
        """Create coordinate grids for elastic deformation"""
        D, H, W = self.target_size
        
        # Create coordinate grids
        z = torch.linspace(-1, 1, D)
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        self.identity_grid = torch.stack([xx, yy, zz], dim=-1)
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply augmentation to sample"""
        if random.random() > self.augmentation_prob:
            return sample
        
        # Apply different augmentation techniques
        sample = self._apply_geometric_augmentation(sample)
        sample = self._apply_intensity_augmentation(sample)
        sample = self._apply_elastic_deformation(sample)
        sample = self._apply_polar_augmentation(sample)
        
        return sample
    
    def _apply_geometric_augmentation(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply geometric augmentations (rotation, translation, scaling)"""
        # Random rotation
        if random.random() < 0.3:
            sample = self._apply_rotation(sample)
        
        # Random translation
        if random.random() < 0.3:
            sample = self._apply_translation(sample)
        
        # Random scaling
        if random.random() < 0.3:
            sample = self._apply_scaling(sample)
        
        return sample
    
    def _apply_rotation(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply random rotation"""
        angle = random.uniform(*self.rotation_range)
        
        # Create rotation matrix (around z-axis for simplicity)
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        # Create 3D affine matrix (3x4) for 3D rotation around Z-axis
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32)
        
        # Apply rotation to moving and fixed images
        sample['moving'] = self._apply_affine_transform(sample['moving'], rotation_matrix)
        sample['fixed'] = self._apply_affine_transform(sample['fixed'], rotation_matrix)
        
        # Update polar coordinates if present
        if 'polar_coords' in sample:
            sample['polar_coords'] = self._apply_affine_transform(sample['polar_coords'], rotation_matrix)
        
        return sample
    
    def _apply_translation(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply random translation"""
        D, H, W = self.target_size
        
        # Random translation
        tx = random.uniform(*self.translation_range) * W
        ty = random.uniform(*self.translation_range) * H
        tz = random.uniform(*self.translation_range) * D
        
        translation_matrix = torch.tensor([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz]
        ], dtype=torch.float32)
        
        # Apply translation
        sample['moving'] = self._apply_affine_transform(sample['moving'], translation_matrix)
        sample['fixed'] = self._apply_affine_transform(sample['fixed'], translation_matrix)
        
        if 'polar_coords' in sample:
            sample['polar_coords'] = self._apply_affine_transform(sample['polar_coords'], translation_matrix)
        
        return sample
    
    def _apply_scaling(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply random scaling"""
        scale = random.uniform(*self.scale_range)
        
        scaling_matrix = torch.tensor([
            [scale, 0, 0, 0],
            [0, scale, 0, 0],
            [0, 0, scale, 0]
        ], dtype=torch.float32)
        
        # Apply scaling
        sample['moving'] = self._apply_affine_transform(sample['moving'], scaling_matrix)
        sample['fixed'] = self._apply_affine_transform(sample['fixed'], scaling_matrix)
        
        if 'polar_coords' in sample:
            sample['polar_coords'] = self._apply_affine_transform(sample['polar_coords'], scaling_matrix)
        
        return sample
    
    def _apply_affine_transform(self, 
                               image: torch.Tensor, 
                               transform_matrix: torch.Tensor) -> torch.Tensor:
        """Apply affine transformation to image"""
        original_shape = image.shape
        was_4d = len(image.shape) == 4
        
        if was_4d:
            image = image.unsqueeze(0)
        
        B, C, D, H, W = image.shape
        
        # Create grid
        grid = F.affine_grid(
            transform_matrix.unsqueeze(0).expand(B, -1, -1),
            image.shape,
            align_corners=False
        )
        
        # Apply transformation
        transformed = F.grid_sample(
            image,
            grid,
            mode='nearest' if C <= 5 else 'bilinear',  # Use nearest for segmentation
            padding_mode='border',
            align_corners=False
        )
        
        return transformed.squeeze(0) if was_4d else transformed
    
    def _apply_intensity_augmentation(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply intensity augmentations"""
        # Add Gaussian noise
        if random.random() < 0.2:
            noise = torch.randn_like(sample['moving']) * self.noise_std
            sample['moving'] = torch.clamp(sample['moving'] + noise, 0, 1)
            
            noise = torch.randn_like(sample['fixed']) * self.noise_std
            sample['fixed'] = torch.clamp(sample['fixed'] + noise, 0, 1)
        
        # Intensity scaling
        if random.random() < 0.2:
            scale = random.uniform(0.8, 1.2)
            sample['moving'] = torch.clamp(sample['moving'] * scale, 0, 1)
            sample['fixed'] = torch.clamp(sample['fixed'] * scale, 0, 1)
        
        return sample
    
    def _apply_elastic_deformation(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply elastic deformation"""
        if random.random() < 0.3:
            # Generate random displacement field
            displacement = self._generate_elastic_displacement()
            
            # Apply elastic deformation
            sample['moving'] = self._apply_displacement_field(sample['moving'], displacement)
            sample['fixed'] = self._apply_displacement_field(sample['fixed'], displacement)
            
            if 'polar_coords' in sample:
                sample['polar_coords'] = self._apply_displacement_field(sample['polar_coords'], displacement)
        
        return sample
    
    def _generate_elastic_displacement(self) -> torch.Tensor:
        """Generate elastic displacement field"""
        D, H, W = self.target_size
        
        # Generate random displacement
        displacement = torch.randn(3, D, H, W) * self.elastic_alpha
        
        # Smooth the displacement field
        kernel_size = int(2 * self.elastic_sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, self.elastic_sigma)
        
        # Apply smoothing to each dimension
        for i in range(3):
            displacement[i] = F.conv3d(
                displacement[i].unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=kernel_size//2
            ).squeeze()
        
        return displacement
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create 3D Gaussian kernel"""
        coords = torch.arange(size, dtype=torch.float32)
        coords = coords - size // 2
        
        # Create 3D grid
        z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
        
        # Compute Gaussian
        kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def _apply_displacement_field(self, 
                                 image: torch.Tensor, 
                                 displacement: torch.Tensor) -> torch.Tensor:
        """Apply displacement field to image"""
        was_4d = len(image.shape) == 4
        
        if was_4d:
            image = image.unsqueeze(0)
        
        B, C, D, H, W = image.shape
        
        # Create grid
        grid = self.identity_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
        
        # Add displacement
        displacement_normalized = displacement.permute(1, 2, 3, 0).unsqueeze(0)
        displacement_normalized = displacement_normalized * 2 / torch.tensor([W, H, D])  # Normalize to [-1, 1]
        
        warped_grid = grid + displacement_normalized
        
        # Apply transformation
        transformed = F.grid_sample(
            image,
            warped_grid,
            mode='nearest' if C <= 5 else 'bilinear',
            padding_mode='border',
            align_corners=False
        )
        
        return transformed.squeeze(0) if was_4d else transformed
    
    def _apply_polar_augmentation(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply polar coordinate specific augmentations"""
        if 'polar_coords' not in sample:
            return sample
        
        # Polar coordinate specific augmentations
        if random.random() < 0.2:
            # Radial scaling
            radial_scale = random.uniform(0.9, 1.1)
            sample['polar_coords'][3:4, :, :, :] *= radial_scale
        
        if random.random() < 0.2:
            # Angular rotation
            angular_rotation = random.uniform(-0.1, 0.1)
            sample['polar_coords'][5:6, :, :, :] += angular_rotation
        
        return sample


class AdvancedAugmentation:
    """
    Advanced augmentation techniques including mixup and cutmix
    """
    
    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
    
    def mixup(self, sample1: Dict[str, torch.Tensor], 
              sample2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply mixup augmentation"""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        mixed_sample = {}
        for key in sample1.keys():
            if isinstance(sample1[key], torch.Tensor):
                mixed_sample[key] = lam * sample1[key] + (1 - lam) * sample2[key]
            else:
                mixed_sample[key] = sample1[key]
        
        return mixed_sample
    
    def cutmix(self, sample1: Dict[str, torch.Tensor], 
               sample2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply cutmix augmentation"""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        # Get image dimensions
        _, D, H, W = sample1['moving'].shape
        
        # Generate random bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cut_d = int(D * cut_rat)
        
        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        cz = np.random.randint(D)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbz1 = np.clip(cz - cut_d // 2, 0, D)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        bbz2 = np.clip(cz + cut_d // 2, 0, D)
        
        # Apply cutmix
        mixed_sample = {}
        for key in sample1.keys():
            if isinstance(sample1[key], torch.Tensor) and sample1[key].shape[-3:] == (D, H, W):
                mixed_sample[key] = sample1[key].clone()
                mixed_sample[key][:, bbz1:bbz2, bby1:bby2, bbx1:bbx2] = sample2[key][:, bbz1:bbz2, bby1:bby2, bbx1:bbx2]
            else:
                mixed_sample[key] = sample1[key]
        
        return mixed_sample
