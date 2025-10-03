"""
Enhanced UNet Architecture with Polar Coordinate Integration
Combines traditional UNet with advanced attention mechanisms and polar coordinate processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .attention_modules import AdaptiveAttention, MultiScaleAttention, DualAttention
from .coordinate_utils import MultiScalePolarProcessor, PolarDeformationField


class EnhancedConvBlock(nn.Module):
    """
    Enhanced convolutional block with attention and residual connections
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 use_attention: bool = True,
                 use_residual: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Main convolution path
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        
        # Residual connection
        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None
        
        # Attention mechanism
        if use_attention:
            self.attention = DualAttention(out_channels)
        
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through enhanced conv block"""
        residual = x
        
        # Main convolution path
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Residual connection
        if self.use_residual:
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            out = out + residual
        
        out = self.activation(out)
        
        # Apply attention
        if self.use_attention:
            out = self.attention(out)
        
        return out


class EnhancedUNet(nn.Module):
    """
    Enhanced UNet with polar coordinate integration and advanced attention mechanisms
    """
    
    def __init__(self, 
                 in_channels: int = 10,  # 5 template + 5 fixed
                 out_channels: int = 3,  # 3D deformation field
                 base_channels: int = 32,
                 num_scales: int = 3,
                 image_size: Tuple[int, int, int] = (128, 128, 128),
                 use_polar_processing: bool = True,
                 use_attention: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_scales = num_scales
        self.image_size = image_size
        self.use_polar_processing = use_polar_processing
        self.use_attention = use_attention
        
        # Input processing
        if use_polar_processing:
            self.input_processor = MultiScalePolarProcessor(
                in_channels, base_channels, num_scales, image_size
            )
            encoder_in_channels = base_channels
        else:
            self.input_processor = None
            encoder_in_channels = in_channels
        
        # Encoder blocks
        self.enc1 = EnhancedConvBlock(encoder_in_channels, base_channels, use_attention)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = EnhancedConvBlock(base_channels, base_channels * 2, use_attention)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = EnhancedConvBlock(base_channels * 2, base_channels * 4, use_attention)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc4 = EnhancedConvBlock(base_channels * 4, base_channels * 8, use_attention)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = EnhancedConvBlock(base_channels * 8, base_channels * 16, use_attention)
        
        # Decoder blocks
        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = EnhancedConvBlock(base_channels * 16, base_channels * 8, use_attention)
        
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = EnhancedConvBlock(base_channels * 8, base_channels * 4, use_attention)
        
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = EnhancedConvBlock(base_channels * 4, base_channels * 2, use_attention)
        
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = EnhancedConvBlock(base_channels * 2, base_channels, use_attention)
        
        # Multi-scale attention for feature fusion
        if use_attention:
            self.feature_fusion = MultiScaleAttention(base_channels, num_scales)
        
        # Deformation field generation
        if use_polar_processing:
            self.deformation_generator = PolarDeformationField(base_channels, out_channels)
        else:
            self.deformation_generator = nn.Sequential(
                nn.Conv3d(base_channels, base_channels // 2, 3, padding=1),
                nn.InstanceNorm3d(base_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv3d(base_channels // 2, out_channels, 1),
                nn.Tanh()  # Normalize deformation field
            )
        
        # Initialize weights
        self._initialize_weights()
        
    # def _initialize_weights(self):
    #     """Initialize network weights"""
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.InstanceNorm3d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    # Inside the EnhancedUNet class
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Your existing Conv3d initialization logic here
                # For example:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # FIX: Add a check for m.weight and m.bias
            elif isinstance(m, nn.BatchNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, polar_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through enhanced UNet"""
        # Input processing
        if self.input_processor is not None:
            x = self.input_processor(x)
        
        # Encoder path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder path with skip connections
        up4 = self.up4(b)
        d4 = self.dec4(torch.cat((up4, e4), dim=1))
        
        up3 = self.up3(d4)
        d3 = self.dec3(torch.cat((up3, e3), dim=1))
        
        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat((up2, e2), dim=1))
        
        up1 = self.up1(d2)
        d1 = self.dec1(torch.cat((up1, e1), dim=1))
        
        # Feature fusion
        if self.use_attention:
            d1 = self.feature_fusion(d1)
        
        # Generate deformation field
        if self.use_polar_processing and polar_coords is not None:
            deformation_field = self.deformation_generator(d1, polar_coords)
        else:
            deformation_field = self.deformation_generator(d1)
        
        return deformation_field
    
    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """Extract feature maps at different levels for analysis"""
        features = {}
        
        # Input processing
        if self.input_processor is not None:
            x = self.input_processor(x)
        
        # Encoder features
        e1 = self.enc1(x)
        features['enc1'] = e1
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        features['enc2'] = e2
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        features['enc3'] = e3
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        features['enc4'] = e4
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        features['bottleneck'] = b
        
        return features


class MultiResolutionUNet(nn.Module):
    """
    Multi-resolution UNet that processes images at different scales simultaneously
    """
    
    def __init__(self, 
                 in_channels: int = 10,
                 out_channels: int = 3,
                 base_channels: int = 32,
                 num_resolutions: int = 3):
        super().__init__()
        
        self.num_resolutions = num_resolutions
        self.base_channels = base_channels
        
        # Multi-resolution UNets
        self.unets = nn.ModuleList([
            EnhancedUNet(in_channels, out_channels, base_channels // (2**i))
            for i in range(num_resolutions)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels * num_resolutions, out_channels, 1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-resolution UNet"""
        B, C, D, H, W = x.shape
        outputs = []
        
        for i, unet in enumerate(self.unets):
            # Downsample input for different resolutions
            scale_factor = 2 ** i
            if i > 0:
                x_scale = F.interpolate(x, 
                                      size=(D // scale_factor, H // scale_factor, W // scale_factor),
                                      mode='trilinear', align_corners=False)
            else:
                x_scale = x
            
            # Process through UNet
            output = unet(x_scale)
            
            # Upsample back to original size
            if i > 0:
                output = F.interpolate(output, size=(D, H, W),
                                     mode='trilinear', align_corners=False)
            
            outputs.append(output)
        
        # Fuse multi-resolution outputs
        fused_output = torch.cat(outputs, dim=1)
        final_output = self.fusion(fused_output)
        
        return final_output
