"""
Enhanced Polar Spatial Transformer
Advanced spatial transformer with polar coordinate awareness and multi-scale processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from .coordinate_utils import PolarCoordinateSystem, create_polar_coordinate_grid


class PolarSpatialTransformer(nn.Module):
    """
    Enhanced spatial transformer with polar coordinate integration
    """
    
    def __init__(self, 
                 size: Tuple[int, int, int] = (128, 128, 128),
                 device: str = 'cuda',
                 use_polar_coords: bool = True,
                 interpolation_mode: str = 'bilinear'):
        super().__init__()
        
        self.size = size
        self.device = device
        self.use_polar_coords = use_polar_coords
        self.interpolation_mode = interpolation_mode
        
        # Create identity grid
        self.register_buffer('identity_grid', self._create_identity_grid())
        
        # Polar coordinate system
        if use_polar_coords:
            self.polar_system = PolarCoordinateSystem(size, device=device)
        
        # Deformation field refinement
        self.field_refiner = nn.Sequential(
            nn.Conv3d(3, 16, 3, padding=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, 3, padding=1),
            nn.InstanceNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 3, 1)
        )
        
    def _create_identity_grid(self) -> torch.Tensor:
        """Create identity grid for spatial transformation"""
        D, H, W = self.size
        
        # Create coordinate grids
        z = torch.linspace(-1, 1, D, device=self.device)
        y = torch.linspace(-1, 1, H, device=self.device)
        x = torch.linspace(-1, 1, W, device=self.device)
        
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        
        # Stack coordinates
        grid = torch.stack([xx, yy, zz], dim=-1)
        return grid.unsqueeze(0)  # [1, D, H, W, 3]
    
    def forward(self, 
                moving: torch.Tensor, 
                deformation_field: torch.Tensor,
                polar_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply spatial transformation with polar coordinate awareness
        
        Args:
            moving: Moving image [B, C, D, H, W]
            deformation_field: Deformation field [B, 3, D, H, W]
            polar_coords: Polar coordinates [B, 6, D, H, W] (optional)
        
        Returns:
            warped: Warped moving image [B, C, D, H, W]
        """
        B, C, D, H, W = moving.shape
        
        # Refine deformation field
        refined_field = self.field_refiner(deformation_field)
        deformation_field = deformation_field + refined_field * 0.1  # Small refinement
        
        # Apply polar coordinate constraints if available
        if self.use_polar_coords and polar_coords is not None:
            deformation_field = self._apply_polar_constraints(deformation_field, polar_coords)
        
        # Reshape deformation field for grid_sample
        flow = deformation_field.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
        
        # Create warped grid
        identity_grid = self.identity_grid.expand(B, -1, -1, -1, -1)
        warped_grid = identity_grid + flow
        
        # Apply spatial transformation
        warped = F.grid_sample(
            moving, 
            warped_grid,
            mode=self.interpolation_mode,
            padding_mode='border',
            align_corners=False
        )
        
        return warped
    
    def _apply_polar_constraints(self, 
                                deformation_field: torch.Tensor,
                                polar_coords: torch.Tensor) -> torch.Tensor:
        """Apply polar coordinate constraints to deformation field"""
        B, C, D, H, W = deformation_field.shape
        
        # Extract polar coordinates
        rho = polar_coords[:, 3:4, :, :, :]  # [B, 1, D, H, W]
        theta = polar_coords[:, 4:5, :, :, :]
        phi = polar_coords[:, 5:6, :, :, :]
        
        # Convert deformation field to polar coordinates
        dx, dy, dz = deformation_field[:, 0:1], deformation_field[:, 1:2], deformation_field[:, 2:3]
        
        # Compute polar components of deformation
        dr = dx * torch.sin(theta) * torch.cos(phi) + dy * torch.sin(theta) * torch.sin(phi) + dz * torch.cos(theta)
        dtheta = (dx * torch.cos(theta) * torch.cos(phi) + dy * torch.cos(theta) * torch.sin(phi) - dz * torch.sin(theta)) / (rho + 1e-8)
        dphi = (-dx * torch.sin(phi) + dy * torch.cos(phi)) / (rho * torch.sin(theta) + 1e-8)
        
        # Apply constraints (e.g., limit radial deformation)
        dr = torch.clamp(dr, -0.1, 0.1)  # Limit radial deformation
        dtheta = torch.clamp(dtheta, -0.2, 0.2)  # Limit angular deformation
        dphi = torch.clamp(dphi, -0.2, 0.2)
        
        # Convert back to Cartesian coordinates
        new_dx = dr * torch.sin(theta) * torch.cos(phi) + rho * dtheta * torch.cos(theta) * torch.cos(phi) - rho * dphi * torch.sin(phi)
        new_dy = dr * torch.sin(theta) * torch.sin(phi) + rho * dtheta * torch.cos(theta) * torch.sin(phi) + rho * dphi * torch.cos(phi)
        new_dz = dr * torch.cos(theta) - rho * dtheta * torch.sin(theta)
        
        constrained_field = torch.cat([new_dx, new_dy, new_dz], dim=1)
        
        return constrained_field


class MultiScaleSpatialTransformer(nn.Module):
    """
    Multi-scale spatial transformer for processing at different resolutions
    """
    
    def __init__(self, 
                 size: Tuple[int, int, int] = (128, 128, 128),
                 num_scales: int = 3,
                 device: str = 'cuda'):
        super().__init__()
        
        self.size = size
        self.num_scales = num_scales
        self.device = device
        
        # Multi-scale transformers
        self.transformers = nn.ModuleList([
            PolarSpatialTransformer(
                size=(size[0] // (2**i), size[1] // (2**i), size[2] // (2**i)),
                device=device
            ) for i in range(num_scales)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(3 * num_scales, 3, 1),
            nn.InstanceNorm3d(3),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, 
                moving: torch.Tensor, 
                deformation_field: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale spatial transformation"""
        B, C, D, H, W = moving.shape
        warped_outputs = []
        
        for i, transformer in enumerate(self.transformers):
            # Downsample for this scale
            scale_factor = 2 ** i
            if i > 0:
                moving_scale = F.interpolate(moving, 
                                           size=(D // scale_factor, H // scale_factor, W // scale_factor),
                                           mode='trilinear', align_corners=False)
                field_scale = F.interpolate(deformation_field,
                                          size=(D // scale_factor, H // scale_factor, W // scale_factor),
                                          mode='trilinear', align_corners=False)
            else:
                moving_scale = moving
                field_scale = deformation_field
            
            # Apply transformation at this scale
            warped_scale = transformer(moving_scale, field_scale)
            
            # Upsample back to original size
            if i > 0:
                warped_scale = F.interpolate(warped_scale, size=(D, H, W),
                                           mode='trilinear', align_corners=False)
            
            warped_outputs.append(warped_scale)
        
        # Fuse multi-scale results
        fused_warped = torch.cat(warped_outputs, dim=1)
        final_warped = self.fusion(fused_warped)
        
        return final_warped


class AdaptiveSpatialTransformer(nn.Module):
    """
    Adaptive spatial transformer that learns optimal interpolation parameters
    """
    
    def __init__(self, 
                 size: Tuple[int, int, int] = (128, 128, 128),
                 device: str = 'cuda'):
        super().__init__()
        
        self.size = size
        self.device = device
        
        # Learnable interpolation weights
        self.interp_weights = nn.Sequential(
            nn.Conv3d(3, 16, 3, padding=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 4, 1),  # 4 interpolation modes
            nn.Softmax(dim=1)
        )
        
        # Base transformer
        self.base_transformer = PolarSpatialTransformer(size, device)
        
    def forward(self, 
                moving: torch.Tensor, 
                deformation_field: torch.Tensor) -> torch.Tensor:
        """Apply adaptive spatial transformation"""
        B, C, D, H, W = moving.shape
        
        # Compute interpolation weights
        weights = self.interp_weights(deformation_field)  # [B, 4, D, H, W]
        
        # Apply different interpolation modes
        warped_bilinear = self.base_transformer(moving, deformation_field)
        warped_nearest = F.grid_sample(
            moving,
            self.base_transformer.identity_grid.expand(B, -1, -1, -1, -1) + 
            deformation_field.permute(0, 2, 3, 4, 1),
            mode='nearest',
            padding_mode='border',
            align_corners=False
        )
        
        # Adaptive combination
        final_warped = (weights[:, 0:1] * warped_bilinear + 
                       weights[:, 1:2] * warped_nearest +
                       weights[:, 2:3] * moving +  # Identity
                       weights[:, 3:4] * warped_bilinear)  # Bilinear again
        
        return final_warped


class HierarchicalSpatialTransformer(nn.Module):
    """
    Hierarchical spatial transformer with coarse-to-fine deformation
    """
    
    def __init__(self, 
                 size: Tuple[int, int, int] = (128, 128, 128),
                 num_levels: int = 3,
                 device: str = 'cuda'):
        super().__init__()
        
        self.size = size
        self.num_levels = num_levels
        self.device = device
        
        # Hierarchical transformers
        self.level_transformers = nn.ModuleList([
            PolarSpatialTransformer(
                size=(size[0] // (2**i), size[1] // (2**i), size[2] // (2**i)),
                device=device
            ) for i in range(num_levels)
        ])
        
        # Deformation field upsamplers
        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose3d(3, 3, 2, stride=2) if i > 0 else nn.Identity()
            for i in range(num_levels)
        ])
        
    def forward(self, 
                moving: torch.Tensor, 
                deformation_field: torch.Tensor) -> torch.Tensor:
        """Apply hierarchical spatial transformation"""
        B, C, D, H, W = moving.shape
        
        # Start with the finest level
        current_moving = moving
        current_field = deformation_field
        
        # Apply transformations from coarse to fine
        for i in range(self.num_levels - 1, -1, -1):
            level = self.num_levels - 1 - i
            
            # Downsample for this level
            scale_factor = 2 ** level
            if level > 0:
                moving_level = F.interpolate(current_moving,
                                           size=(D // scale_factor, H // scale_factor, W // scale_factor),
                                           mode='trilinear', align_corners=False)
                field_level = F.interpolate(current_field,
                                          size=(D // scale_factor, H // scale_factor, W // scale_factor),
                                          mode='trilinear', align_corners=False)
            else:
                moving_level = current_moving
                field_level = current_field
            
            # Apply transformation at this level
            warped_level = self.level_transformers[i](moving_level, field_level)
            
            # Upsample for next level
            if level > 0:
                warped_level = F.interpolate(warped_level, size=(D, H, W),
                                           mode='trilinear', align_corners=False)
            
            # Update for next iteration
            current_moving = warped_level
        
        return current_moving
