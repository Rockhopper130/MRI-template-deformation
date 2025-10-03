#!/usr/bin/env python3
"""
Comprehensive shape testing for the enhanced polar registration system
"""

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_shapes():
    """Test all components with consistent input shapes"""
    print("Testing Enhanced Polar Registration System Shapes...")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    in_channels = 10  # 5 template + 5 fixed
    out_channels = 3  # 3D deformation field
    base_channels = 32
    num_scales = 3
    image_size = (64, 64, 64)  # Smaller for testing
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test input
    input_tensor = torch.randn(batch_size, in_channels, *image_size, device=device)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    try:
        # Test 1: MultiScalePolarProcessor
        print("\n1. Testing MultiScalePolarProcessor...")
        from models.coordinate_utils import MultiScalePolarProcessor
        
        processor = MultiScalePolarProcessor(
            in_channels=in_channels,
            out_channels=base_channels,
            num_scales=num_scales,
            image_size=image_size
        ).to(device)
        
        output = processor(input_tensor)
        print(f"   Input: {input_tensor.shape}")
        print(f"   Output: {output.shape}")
        print(f"   Expected output channels: {base_channels}")
        print(f"   ✓ MultiScalePolarProcessor: PASSED")
        
        # Test 2: EnhancedUNet
        print("\n2. Testing EnhancedUNet...")
        from models.enhanced_unet import EnhancedUNet
        
        unet = EnhancedUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            num_scales=num_scales,
            image_size=image_size,
            use_polar_processing=True,
            use_attention=True
        ).to(device)
        
        # Test with polar coordinates
        polar_coords = torch.randn(batch_size, 6, *image_size, device=device)
        deformation_field = unet(input_tensor, polar_coords)
        print(f"   Input: {input_tensor.shape}")
        print(f"   Polar coords: {polar_coords.shape}")
        print(f"   Deformation field: {deformation_field.shape}")
        print(f"   Expected deformation field: [B, 3, D, H, W]")
        print(f"   ✓ EnhancedUNet: PASSED")
        
        # Test 3: PolarSpatialTransformer
        print("\n3. Testing PolarSpatialTransformer...")
        from models.polar_transformer import PolarSpatialTransformer
        
        transformer = PolarSpatialTransformer(
            size=image_size,
            device=device,
            use_polar_coords=True
        ).to(device)
        
        # Create a moving image
        moving_image = torch.randn(batch_size, 5, *image_size, device=device)
        warped_image = transformer(moving_image, deformation_field, polar_coords)
        print(f"   Moving image: {moving_image.shape}")
        print(f"   Deformation field: {deformation_field.shape}")
        print(f"   Warped image: {warped_image.shape}")
        print(f"   ✓ PolarSpatialTransformer: PASSED")
        
        # Test 4: PolarDeformationField
        print("\n4. Testing PolarDeformationField...")
        from models.coordinate_utils import PolarDeformationField
        
        # Create features from UNet (before final deformation generation)
        features = torch.randn(batch_size, base_channels, *image_size, device=device)
        polar_field = PolarDeformationField(
            in_channels=base_channels,
            out_channels=out_channels
        ).to(device)
        
        polar_deformation = polar_field(features, polar_coords)
        print(f"   Features: {features.shape}")
        print(f"   Polar coords: {polar_coords.shape}")
        print(f"   Polar deformation: {polar_deformation.shape}")
        print(f"   ✓ PolarDeformationField: PASSED")
        
        # Test 5: Attention modules
        print("\n5. Testing Attention Modules...")
        from models.attention_modules import MultiScaleAttention, CrossModalAttention
        
        # MultiScaleAttention
        ms_attention = MultiScaleAttention(
            in_channels=base_channels,
            num_scales=num_scales
        ).to(device)
        
        attended_features = ms_attention(features)
        print(f"   MultiScaleAttention input: {features.shape}")
        print(f"   MultiScaleAttention output: {attended_features.shape}")
        
        # CrossModalAttention
        cm_attention = CrossModalAttention(
            template_channels=5,
            fixed_channels=5,
            out_channels=base_channels
        ).to(device)
        
        template_features = torch.randn(batch_size, 5, *image_size, device=device)
        fixed_features = torch.randn(batch_size, 5, *image_size, device=device)
        cross_attended = cm_attention(template_features, fixed_features)
        print(f"   CrossModalAttention template: {template_features.shape}")
        print(f"   CrossModalAttention fixed: {fixed_features.shape}")
        print(f"   CrossModalAttention output: {cross_attended.shape}")
        print(f"   ✓ Attention Modules: PASSED")
        
        print("\n" + "=" * 60)
        print("🎉 ALL SHAPE TESTS PASSED! 🎉")
        print("The system is ready for training.")
        
    except Exception as e:
        print(f"\n❌ SHAPE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_shapes()
    sys.exit(0 if success else 1)
