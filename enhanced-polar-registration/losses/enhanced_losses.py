"""
Enhanced Loss Functions for Polar Coordinate Registration
Includes advanced geometric, anatomical, and consistency losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


def dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Enhanced Dice loss with better numerical stability"""
    # Handle different input shapes
    if len(y_pred.shape) == 3:  # [D, H, W] - single channel
        y_pred = y_pred.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        y_true = y_true.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    
    # Compute Dice coefficient
    ndims = len(y_pred.shape) - 2
    vol_axes = list(range(2, ndims + 2))
    
    intersection = (y_pred * y_true).sum(dim=vol_axes)
    union = y_pred.sum(dim=vol_axes) + y_true.sum(dim=vol_axes)
    
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice_score.mean()


def cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Enhanced cross-entropy loss for segmentation"""
    if len(pred.shape) == 3:  # [D, H, W]
        pred = pred.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        target = target.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    
    # Convert target to long tensor for cross entropy
    target_long = target.squeeze(1).long()
    
    # Use BCE with logits for binary case
    return F.binary_cross_entropy_with_logits(pred.squeeze(1), target.squeeze(1))


def bending_energy_loss(flow: torch.Tensor) -> torch.Tensor:
    """Enhanced bending energy loss with better gradient computation"""
    if len(flow.shape) == 5:  # [B, 3, D, H, W]
        flow = flow.squeeze(0)  # [3, D, H, W]
    
    # Compute second-order derivatives
    d2x = flow[:, 2:] - 2*flow[:, 1:-1] + flow[:, :-2]
    d2y = flow[:, :, 2:] - 2*flow[:, :, 1:-1] + flow[:, :, :-2]
    d2z = flow[:, :, :, 2:] - 2*flow[:, :, :, 1:-1] + flow[:, :, :, :-2]
    
    return torch.mean(d2x**2) + torch.mean(d2y**2) + torch.mean(d2z**2)


def jacobian_det_loss(flow: torch.Tensor) -> torch.Tensor:
    """Enhanced Jacobian determinant loss to prevent folding"""
    if len(flow.shape) == 5:  # [B, 3, D, H, W]
        flow = flow.squeeze(0)  # [3, D, H, W]
    
    # Compute gradients along each spatial dimension
    grads = torch.gradient(flow, dim=(1, 2, 3))
    
    # Stack gradients to form the Jacobian matrix for each voxel.
    # The shape will be [3, D, H, W, 3], where the first 3 is for flow components (dx, dy, dz)
    # and the last 3 is for spatial dimensions (d/dD, d/dH, d/dW).
    J = torch.stack(grads, dim=4)

    # Permute the dimensions to get a batch of 3x3 matrices.
    # The new shape is [D, H, W, 3, 3].
    J_permuted = J.permute(1, 2, 3, 0, 4)
    
    # Compute the determinant of each 3x3 matrix.
    # Convert to float32 to avoid half precision issues
    det = torch.det(J_permuted.float())  # Resulting shape: [D, H, W]
    
    # Penalize negative determinants (folding)
    return torch.mean(F.relu(-det))


class PolarConsistencyLoss(nn.Module):
    """
    Loss to ensure consistency in polar coordinate transformations
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, deformation_field: torch.Tensor, polar_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute polar consistency loss
        
        Args:
            deformation_field: [B, 3, D, H, W] - deformation field
            polar_coords: [B, 6, D, H, W] - polar coordinates (x, y, z, rho, theta, phi)
        """
        B, C, D, H, W = deformation_field.shape
        
        # Extract polar coordinates
        rho = polar_coords[:, 3:4, :, :, :]
        theta = polar_coords[:, 4:5, :, :, :]
        phi = polar_coords[:, 5:6, :, :, :]
        
        # Convert deformation to polar components
        dx, dy, dz = deformation_field[:, 0:1], deformation_field[:, 1:2], deformation_field[:, 2:3]
        
        # Radial component
        dr = dx * torch.sin(theta) * torch.cos(phi) + dy * torch.sin(theta) * torch.sin(phi) + dz * torch.cos(theta)
        
        # Angular components
        dtheta = (dx * torch.cos(theta) * torch.cos(phi) + dy * torch.cos(theta) * torch.sin(phi) - dz * torch.sin(theta)) / (rho + 1e-8)
        dphi = (-dx * torch.sin(phi) + dy * torch.cos(phi)) / (rho * torch.sin(theta) + 1e-8)
        
        # Penalize large angular deformations (should be smooth)
        angular_penalty = torch.mean(dtheta**2) + torch.mean(dphi**2)
        
        # Penalize radial deformations that are too large
        radial_penalty = torch.mean(torch.clamp(torch.abs(dr) - 0.1, min=0)**2)
        
        return self.weight * (angular_penalty + radial_penalty)


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss that evaluates registration quality at different resolutions
    """
    
    def __init__(self, 
                 base_loss_fn,
                 scales: Tuple[int, ...] = (1, 2, 4),
                 weights: Tuple[float, ...] = (1.0, 0.5, 0.25)):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.scales = scales
        self.weights = weights
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute multi-scale loss"""
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale > 1:
                # Downsample
                pred_scale = F.interpolate(pred, scale_factor=1/scale, mode='trilinear', align_corners=False)
                target_scale = F.interpolate(target, scale_factor=1/scale, mode='trilinear', align_corners=False)
            else:
                pred_scale = pred
                target_scale = target
            
            # Compute loss at this scale
            scale_loss = self.base_loss_fn(pred_scale, target_scale, **kwargs)
            total_loss += weight * scale_loss
        
        return total_loss


class AnatomicalConsistencyLoss(nn.Module):
    """
    Loss to ensure anatomical consistency in registration
    """
    
    def __init__(self, weight: float = 0.2):
        super().__init__()
        self.weight = weight
        
        # 3D Sobel filters for edge detection
        # X-direction gradient
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], 
                               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        
        # Y-direction gradient  
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               [[-2, -4, -2], [0, 0, 0], [2, 4, 2]], 
                               [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        
        # Z-direction gradient
        sobel_z = torch.tensor([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], 
                               [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
                               [[1, 2, 1], [2, 4, 2], [1, 2, 1]]], dtype=torch.float32)
        
        # Store base filters (will be expanded for multi-channel inputs)
        self.register_buffer('sobel_x_base', sobel_x)
        self.register_buffer('sobel_y_base', sobel_y)
        self.register_buffer('sobel_z_base', sobel_z)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute anatomical consistency loss"""
        B, C, D, H, W = pred.shape
        
        device = pred.device
        
        # Create multi-channel filters and ensure they are on the correct device
        sobel_x = self.sobel_x_base.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1, -1).to(device)
        sobel_y = self.sobel_y_base.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1, -1).to(device)
        sobel_z = self.sobel_z_base.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1, -1).to(device)

        # Compute gradients
        pred_grad_x = F.conv3d(pred, sobel_x, padding=1, groups=C)
        pred_grad_y = F.conv3d(pred, sobel_y, padding=1, groups=C)
        pred_grad_z = F.conv3d(pred, sobel_z, padding=1, groups=C)
        
        target_grad_x = F.conv3d(target, sobel_x, padding=1, groups=C)
        target_grad_y = F.conv3d(target, sobel_y, padding=1, groups=C)
        target_grad_z = F.conv3d(target, sobel_z, padding=1, groups=C)
        
        # Gradient magnitude
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + pred_grad_z**2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + target_grad_z**2 + 1e-8)
        
        # Gradient direction
        pred_grad_dir = torch.stack([pred_grad_x, pred_grad_y, pred_grad_z], dim=1)
        target_grad_dir = torch.stack([target_grad_x, target_grad_y, target_grad_z], dim=1)
        
        # Normalize directions
        pred_grad_dir = F.normalize(pred_grad_dir, p=2, dim=1)
        target_grad_dir = F.normalize(target_grad_dir, p=2, dim=1)
        
        # Magnitude loss
        magnitude_loss = F.mse_loss(pred_grad_mag, target_grad_mag)
        
        # Direction loss (cosine similarity)
        direction_loss = 1 - F.cosine_similarity(pred_grad_dir, target_grad_dir, dim=1).mean()
        
        return self.weight * (magnitude_loss + direction_loss)


class GradientConsistencyLoss(nn.Module):
    """
    Loss to ensure gradient consistency in deformation fields
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, deformation_field: torch.Tensor) -> torch.Tensor:
        """Compute gradient consistency loss"""
        # Compute gradients
        grad_x = torch.gradient(deformation_field, dim=2)[0]  # [B, 3, D, H, W]
        grad_y = torch.gradient(deformation_field, dim=3)[0]
        grad_z = torch.gradient(deformation_field, dim=4)[0]
        
        # Compute divergence (trace of Jacobian)
        divergence = grad_x[:, 0:1] + grad_y[:, 1:2] + grad_z[:, 2:3]
        
        # Penalize large divergence (should be close to zero for incompressible flow)
        divergence_loss = torch.mean(divergence**2)
        
        # Compute curl (should be small for smooth deformation)
        curl_x = grad_z[:, 1:2] - grad_y[:, 2:3]
        curl_y = grad_x[:, 2:3] - grad_z[:, 0:1]
        curl_z = grad_y[:, 0:1] - grad_x[:, 1:2]
        
        curl_magnitude = torch.sqrt(curl_x**2 + curl_y**2 + curl_z**2 + 1e-8)
        curl_loss = torch.mean(curl_magnitude)
        
        return self.weight * (divergence_loss + curl_loss)


class FeatureConsistencyLoss(nn.Module):
    """
    Loss to ensure feature consistency between warped and target images
    """
    
    def __init__(self, weight: float = 0.15):
        super().__init__()
        self.weight = weight
        
        # Feature extractor (simple CNN) - will be dynamically created for multi-channel input
        self.base_feature_extractor = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1, bias=False),  # Remove bias to avoid type mismatch
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute feature consistency loss"""
        B, C, D, H, W = pred.shape
        
        # Process each channel separately and average
        pred_features_list = []
        target_features_list = []
        
        for c in range(C):
            pred_c = pred[:, c:c+1, :, :, :]  # [B, 1, D, H, W]
            target_c = target[:, c:c+1, :, :, :]  # [B, 1, D, H, W]
            
            # Extract features for this channel
            pred_feat_c = self.base_feature_extractor(pred_c)
            target_feat_c = self.base_feature_extractor(target_c)
            
            pred_features_list.append(pred_feat_c)
            target_features_list.append(target_feat_c)
        
        # Average features across channels
        pred_features = torch.stack(pred_features_list, dim=1).mean(dim=1)
        target_features = torch.stack(target_features_list, dim=1).mean(dim=1)
        
        # Compute feature similarity
        feature_loss = F.mse_loss(pred_features, target_features)
        
        return self.weight * feature_loss


class EnhancedCompositeLoss(nn.Module):
    """
    Enhanced composite loss combining multiple loss functions
    """
    
    def __init__(self, 
                 dice_weight: float = 0.4,
                 ce_weight: float = 0.2,
                 bending_weight: float = 0.1,
                 jacobian_weight: float = 0.05,
                 polar_weight: float = 0.1,
                 anatomical_weight: float = 0.1,
                 gradient_weight: float = 0.05):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.bending_weight = bending_weight
        self.jacobian_weight = jacobian_weight
        self.polar_weight = polar_weight
        self.anatomical_weight = anatomical_weight
        self.gradient_weight = gradient_weight
        
        # Initialize loss components
        self.polar_loss = PolarConsistencyLoss(polar_weight)
        self.anatomical_loss = AnatomicalConsistencyLoss(anatomical_weight)
        self.gradient_loss = GradientConsistencyLoss(gradient_weight)
        self.feature_loss = FeatureConsistencyLoss(0.1)
        
        # Multi-scale dice loss
        self.dice_loss = MultiScaleLoss(dice_loss, scales=(1, 2, 4), weights=(1.0, 0.5, 0.25))
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor, 
                deformation_field: torch.Tensor,
                polar_coords: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute enhanced composite loss
        
        Args:
            pred: Predicted warped image [B, C, D, H, W]
            target: Target image [B, C, D, H, W]
            deformation_field: Deformation field [B, 3, D, H, W]
            polar_coords: Polar coordinates [B, 6, D, H, W] (optional)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        
        # Basic losses
        loss_dict['dice'] = self.dice_loss(pred, target)
        loss_dict['ce'] = cross_entropy_loss(pred, target)
        loss_dict['bending'] = bending_energy_loss(deformation_field)
        loss_dict['jacobian'] = jacobian_det_loss(deformation_field)
        
        # Enhanced losses
        if polar_coords is not None:
            loss_dict['polar'] = self.polar_loss(deformation_field, polar_coords)
        else:
            loss_dict['polar'] = torch.tensor(0.0, device=pred.device)
        
        loss_dict['anatomical'] = self.anatomical_loss(pred, target)
        loss_dict['gradient'] = self.gradient_loss(deformation_field)
        loss_dict['feature'] = self.feature_loss(pred, target)
        
        # Compute total loss
        total_loss = (
            self.dice_weight * loss_dict['dice'] +
            self.ce_weight * loss_dict['ce'] +
            self.bending_weight * loss_dict['bending'] +
            self.jacobian_weight * loss_dict['jacobian'] +
            self.polar_weight * loss_dict['polar'] +
            self.anatomical_weight * loss_dict['anatomical'] +
            self.gradient_weight * loss_dict['gradient'] +
            0.1 * loss_dict['feature']
        )
        
        return total_loss, loss_dict
