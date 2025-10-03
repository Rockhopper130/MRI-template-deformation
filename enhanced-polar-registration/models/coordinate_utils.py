"""
Enhanced Polar Coordinate System Utilities
Provides multi-scale polar coordinate transformations and geometric operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class PolarCoordinateSystem(nn.Module):
    """
    Enhanced polar coordinate system with multi-scale processing
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int, int] = (128, 128, 128),
                 num_scales: int = 3,
                 device: str = 'cuda'):
        super().__init__()
        self.image_size = image_size
        self.num_scales = num_scales
        self.device = device
        
        # Create multi-scale coordinate grids
        self.coordinate_grids = self._create_coordinate_grids()
        
    def _create_coordinate_grids(self) -> list:
        """Create multi-scale polar coordinate grids"""
        D, H, W = self.image_size
        grids = []
        
        for scale in range(self.num_scales):
            scale_factor = 2 ** scale
            d_scale, h_scale, w_scale = D // scale_factor, H // scale_factor, W // scale_factor
            
            # Create coordinate grids
            z = torch.linspace(-1, 1, d_scale, device=self.device)
            y = torch.linspace(-1, 1, h_scale, device=self.device)
            x = torch.linspace(-1, 1, w_scale, device=self.device)
            
            zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
            
            # Convert to polar coordinates
            rho = torch.sqrt(xx**2 + yy**2 + zz**2)
            theta = torch.acos(torch.clamp(zz / (rho + 1e-8), -1, 1))
            phi = torch.atan2(yy, xx)
            
            # Stack coordinates
            grid = torch.stack([xx, yy, zz, rho, theta, phi], dim=-1)
            grids.append(grid)
            
        return grids  # List of tensors with different sizes
    
    def cartesian_to_polar(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert Cartesian coordinates to polar coordinates"""
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        
        rho = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.acos(torch.clamp(z / (rho + 1e-8), -1, 1))
        phi = torch.atan2(y, x)
        
        return rho, theta, phi
    
    def polar_to_cartesian(self, rho: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Convert polar coordinates to Cartesian coordinates"""
        x = rho * torch.sin(theta) * torch.cos(phi)
        y = rho * torch.sin(theta) * torch.sin(phi)
        z = rho * torch.cos(theta)
        
        return torch.stack([x, y, z], dim=-1)
    
    def get_polar_features(self, scale: int = 0) -> torch.Tensor:
        """Get polar coordinate features for a specific scale"""
        return self.coordinate_grids[scale]
    
    def compute_polar_gradients(self, field: torch.Tensor) -> torch.Tensor:
        """Compute gradients in polar coordinate system"""
        # field: [B, 3, D, H, W] - displacement field
        B, C, D, H, W = field.shape
        
        # Get polar coordinates
        rho, theta, phi = self.cartesian_to_polar(field.permute(0, 2, 3, 4, 1))
        
        # Compute gradients
        grad_rho = torch.gradient(rho, dim=(1, 2, 3))
        grad_theta = torch.gradient(theta, dim=(1, 2, 3))
        grad_phi = torch.gradient(phi, dim=(1, 2, 3))
        
        # Stack gradients
        polar_gradients = torch.stack([
            grad_rho[0], grad_rho[1], grad_rho[2],
            grad_theta[0], grad_theta[1], grad_theta[2],
            grad_phi[0], grad_phi[1], grad_phi[2]
        ], dim=1)
        
        return polar_gradients


class MultiScalePolarProcessor(nn.Module):
    """
    Multi-scale polar coordinate processor for enhanced feature extraction
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 num_scales: int = 3,
                 image_size: Tuple[int, int, int] = (128, 128, 128)):
        super().__init__()
        self.num_scales = num_scales
        self.image_size = image_size
        
        # Multi-scale processors (in_channels + 6 for polar coordinates)
        # Ensure each scale outputs enough channels so total = out_channels
        channels_per_scale = out_channels // num_scales
        remaining_channels = out_channels % num_scales
        
        self.scale_processors = nn.ModuleList()
        for i in range(num_scales):
            # Distribute remaining channels to first few scales
            scale_out_channels = channels_per_scale + (1 if i < remaining_channels else 0)
            self.scale_processors.append(nn.Sequential(
                nn.Conv3d(in_channels + 6, scale_out_channels, 3, padding=1),
                nn.InstanceNorm3d(scale_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(scale_out_channels, scale_out_channels, 3, padding=1),
                nn.InstanceNorm3d(scale_out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Polar coordinate system
        self.polar_system = PolarCoordinateSystem(image_size, num_scales)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through multi-scale polar coordinate system"""
        # Handle different tensor dimensions
        if len(x.shape) == 4:
            # Handle 4D tensor [C, D, H, W] by adding batch dimension
            x = x.unsqueeze(0)
            was_4d = True
        elif len(x.shape) == 5:
            was_4d = False
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {len(x.shape)}D tensor with shape {x.shape}")
        
        B, C, D, H, W = x.shape
        scale_features = []
        
        for scale in range(self.num_scales):
            # Downsample input
            scale_factor = 2 ** scale
            if scale > 0:
                x_scale = F.interpolate(x, 
                                      size=(D // scale_factor, H // scale_factor, W // scale_factor),
                                      mode='trilinear', align_corners=False)
            else:
                x_scale = x
            
            # Get polar coordinates for this scale
            polar_coords = self.polar_system.get_polar_features(scale)
            polar_coords = polar_coords.to(x.device)  # Move to same device as input
            # polar_coords shape: [D, H, W, 6]
            polar_coords = polar_coords.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, D, H, W, 6]
            polar_coords = polar_coords.permute(0, 4, 1, 2, 3)  # [B, 6, D, H, W]
            
            # Concatenate with input features
            x_with_polar = torch.cat([x_scale, polar_coords], dim=1)
            
            # Process through scale-specific network
            scale_feat = self.scale_processors[scale](x_with_polar)
            
            # Upsample back to original size
            if scale > 0:
                scale_feat = F.interpolate(scale_feat, size=(D, H, W),
                                         mode='trilinear', align_corners=False)
            
            scale_features.append(scale_feat)
        
        # Fuse multi-scale features
        fused_features = torch.cat(scale_features, dim=1)
        output = self.fusion(fused_features)
        
        # Handle output shape - if input was 4D, return 4D
        if was_4d:
            output = output.squeeze(0)
        
        return output


class PolarDeformationField(nn.Module):
    """
    Enhanced deformation field generator using polar coordinates
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int = 3,
                 num_polar_basis: int = 16):
        super().__init__()
        self.num_polar_basis = num_polar_basis
        
        # Polar basis functions
        self.polar_basis = nn.Parameter(torch.randn(num_polar_basis, 3))
        
        # Deformation field generator
        self.field_generator = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_polar_basis, 1),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        # Final projection to Cartesian coordinates
        self.cartesian_proj = nn.Linear(num_polar_basis, out_channels)
        
    def forward(self, x: torch.Tensor, polar_coords: torch.Tensor) -> torch.Tensor:
        """Generate deformation field using polar basis functions"""
        B, C, D, H, W = x.shape
        
        # Generate polar basis coefficients
        polar_coeffs = self.field_generator(x)  # [B, num_polar_basis, D, H, W]
        
        # Reshape for matrix multiplication
        polar_coeffs = polar_coeffs.permute(0, 2, 3, 4, 1)  # [B, D, H, W, num_polar_basis]
        polar_coeffs = polar_coeffs.reshape(-1, self.num_polar_basis)
        
        # Project to Cartesian coordinates
        cartesian_field = self.cartesian_proj(polar_coeffs)  # [B*D*H*W, 3]
        cartesian_field = cartesian_field.reshape(B, D, H, W, 3)
        cartesian_field = cartesian_field.permute(0, 4, 1, 2, 3)  # [B, 3, D, H, W]
        
        return cartesian_field


def create_polar_coordinate_grid(size: Tuple[int, int, int], 
                                device: torch.device) -> torch.Tensor:
    """Create a polar coordinate grid for the given size"""
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
    grid = torch.stack([xx, yy, zz, rho, theta, phi], dim=-1)
    
    return grid
