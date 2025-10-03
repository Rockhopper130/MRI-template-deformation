"""
Advanced Attention Modules for Enhanced Polar Registration
Includes multi-scale attention, polar-aware attention, and spatial attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism for processing features at different scales
    """
    
    def __init__(self, 
                 in_channels: int,
                 num_scales: int = 3,
                 reduction_ratio: int = 16):
        super().__init__()
        self.num_scales = num_scales
        self.in_channels = in_channels
        
        # Multi-scale feature extractors
        self.scale_extractors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d((2**i, 2**i, 2**i)),
                nn.Conv3d(in_channels, in_channels // reduction_ratio, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels // reduction_ratio, in_channels, 1),
                nn.Sigmoid()
            ) for i in range(num_scales)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels * num_scales, in_channels, 1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale attention to input features"""
        B, C, D, H, W = x.shape
        scale_attentions = []
        
        for i, extractor in enumerate(self.scale_extractors):
            # Extract attention at this scale
            attention = extractor(x)
            
            # Upsample attention to original size
            attention = F.interpolate(attention, size=(D, H, W), 
                                    mode='trilinear', align_corners=False)
            
            # Apply attention
            attended_features = x * attention
            scale_attentions.append(attended_features)
        
        # Fuse multi-scale attended features
        fused_features = torch.cat(scale_attentions, dim=1)
        output = self.fusion(fused_features)
        
        return output


class PolarAttention(nn.Module):
    """
    Polar coordinate-aware attention mechanism
    """
    
    def __init__(self, 
                 in_channels: int,
                 polar_channels: int = 6,  # rho, theta, phi for 2 points
                 num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        
        # Polar coordinate projections
        self.polar_proj = nn.Linear(polar_channels, in_channels)
        
        # Output projection
        self.out_proj = nn.Linear(in_channels, in_channels)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * 4, in_channels)
        )
        
    def forward(self, x: torch.Tensor, polar_coords: torch.Tensor) -> torch.Tensor:
        """Apply polar-aware attention"""
        B, C, D, H, W = x.shape
        
        # Reshape for attention computation
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B * D * H * W, C)
        polar_flat = polar_coords.permute(0, 2, 3, 4, 1).reshape(B * D * H * W, -1)
        
        # Add polar coordinate information
        polar_features = self.polar_proj(polar_flat)
        x_with_polar = x_flat + polar_features
        
        # Self-attention
        residual = x_with_polar
        x_norm = self.norm1(x_with_polar)
        
        Q = self.q_proj(x_norm).reshape(-1, self.num_heads, self.head_dim)
        K = self.k_proj(x_norm).reshape(-1, self.num_heads, self.head_dim)
        V = self.v_proj(x_norm).reshape(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.reshape(-1, C)
        
        # Output projection
        output = self.out_proj(attended)
        output = output + residual
        
        # Feed-forward network
        residual = output
        output = self.norm2(output)
        output = self.ffn(output)
        output = output + residual
        
        # Reshape back to original format
        output = output.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        
        return output


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for focusing on important spatial regions
    """
    
    def __init__(self, in_channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention"""
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention


class ChannelAttention(nn.Module):
    """
    Channel attention mechanism for focusing on important feature channels
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention"""
        B, C, D, H, W = x.shape
        
        # Global average pooling
        avg_out = self.avg_pool(x).view(B, C)
        avg_out = self.fc(avg_out)
        
        # Global max pooling
        max_out = self.max_pool(x).view(B, C)
        max_out = self.fc(max_out)
        
        # Combine and apply attention
        attention = avg_out + max_out
        attention = self.sigmoid(attention).view(B, C, 1, 1, 1)
        
        return x * attention


class DualAttention(nn.Module):
    """
    Combined spatial and channel attention
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both channel and spatial attention"""
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing template and fixed image features
    """
    
    def __init__(self, template_channels: int, fixed_channels: int, out_channels: int, num_heads: int = 8):
        super().__init__()
        self.template_channels = template_channels
        self.fixed_channels = fixed_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        
        # Cross-attention projections
        self.q_proj = nn.Linear(template_channels, out_channels)
        self.k_proj = nn.Linear(fixed_channels, out_channels)
        self.v_proj = nn.Linear(fixed_channels, out_channels)
        
        self.out_proj = nn.Linear(out_channels, out_channels)
        
    def forward(self, template_features: torch.Tensor, fixed_features: torch.Tensor) -> torch.Tensor:
        """Apply cross-modal attention between template and fixed features"""
        B, C_t, D, H, W = template_features.shape
        _, C_f, _, _, _ = fixed_features.shape
        
        # Reshape for attention computation
        template_flat = template_features.permute(0, 2, 3, 4, 1).reshape(B * D * H * W, C_t)
        fixed_flat = fixed_features.permute(0, 2, 3, 4, 1).reshape(B * D * H * W, C_f)
        
        # Cross-attention: template queries, fixed keys and values
        Q = self.q_proj(template_flat).reshape(-1, self.num_heads, self.head_dim)
        K = self.k_proj(fixed_flat).reshape(-1, self.num_heads, self.head_dim)
        V = self.v_proj(fixed_flat).reshape(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.reshape(-1, self.out_channels)
        
        # Output projection
        output = self.out_proj(attended)
        
        # Reshape back to original format
        output = output.reshape(B, D, H, W, self.out_channels).permute(0, 4, 1, 2, 3)
        
        return output


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention mechanism that learns to combine different attention types
    """
    
    def __init__(self, in_channels: int, num_attention_types: int = 4):
        super().__init__()
        self.num_attention_types = num_attention_types
        
        # Different attention mechanisms
        self.attention_modules = nn.ModuleList([
            MultiScaleAttention(in_channels),
            PolarAttention(in_channels),
            DualAttention(in_channels),
            CrossModalAttention(in_channels)
        ])
        
        # Adaptive weighting
        self.attention_weights = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, num_attention_types, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels * num_attention_types, in_channels, 1),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor, polar_coords: Optional[torch.Tensor] = None,
                fixed_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply adaptive attention combining multiple attention types"""
        B, C, D, H, W = x.shape
        
        # Compute attention weights
        weights = self.attention_weights(x)  # [B, num_attention_types, 1, 1, 1]
        
        # Apply different attention mechanisms
        attention_outputs = []
        
        # Multi-scale attention
        ms_att = self.attention_modules[0](x)
        attention_outputs.append(ms_att)
        
        # Polar attention (if polar coordinates provided)
        if polar_coords is not None:
            polar_att = self.attention_modules[1](x, polar_coords)
            attention_outputs.append(polar_att)
        else:
            attention_outputs.append(x)  # Identity if no polar coords
        
        # Dual attention
        dual_att = self.attention_modules[2](x)
        attention_outputs.append(dual_att)
        
        # Cross-modal attention (if fixed features provided)
        if fixed_features is not None:
            cross_att = self.attention_modules[3](x, fixed_features)
            attention_outputs.append(cross_att)
        else:
            attention_outputs.append(x)  # Identity if no fixed features
        
        # Weighted combination
        weighted_outputs = []
        for i, output in enumerate(attention_outputs):
            weight = weights[:, i:i+1, :, :, :]
            weighted_outputs.append(output * weight)
        
        # Fuse weighted outputs
        fused_output = torch.cat(weighted_outputs, dim=1)
        final_output = self.fusion(fused_output)
        
        return final_output
