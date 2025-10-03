"""
Test script to verify the shape mismatch fix
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_polar_coordinate_system():
    """Test PolarCoordinateSystem creation"""
    print("Testing PolarCoordinateSystem...")
    
    try:
        from models.coordinate_utils import PolarCoordinateSystem
        
        # Test with different scales
        system = PolarCoordinateSystem((128, 128, 128), num_scales=3, device='cpu')
        print("✓ PolarCoordinateSystem created successfully")
        
        # Check coordinate grids
        print(f"Number of coordinate grids: {len(system.coordinate_grids)}")
        for i, grid in enumerate(system.coordinate_grids):
            print(f"  Scale {i}: {grid.shape}")
        
        # Test getting polar features
        for scale in range(3):
            features = system.get_polar_features(scale)
            print(f"  Scale {scale} features: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ PolarCoordinateSystem test failed: {e}")
        return False

def test_multi_scale_processor():
    """Test MultiScalePolarProcessor creation"""
    print("\nTesting MultiScalePolarProcessor...")
    
    try:
        from models.coordinate_utils import MultiScalePolarProcessor
        
        # Create processor
        processor = MultiScalePolarProcessor(
            in_channels=10,
            out_channels=32,
            num_scales=3,
            image_size=(128, 128, 128)
        )
        print("✓ MultiScalePolarProcessor created successfully")
        
        # Test forward pass
        x = torch.randn(1, 10, 128, 128, 128)
        output = processor(x)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ MultiScalePolarProcessor test failed: {e}")
        return False

def test_enhanced_unet():
    """Test EnhancedUNet creation"""
    print("\nTesting EnhancedUNet...")
    
    try:
        from models import EnhancedUNet
        
        # Create model
        model = EnhancedUNet(
            in_channels=10,
            out_channels=3,
            base_channels=16,  # Smaller for testing
            num_scales=3,
            image_size=(128, 128, 128),
            use_polar_processing=True,
            use_attention=True
        )
        print("✓ EnhancedUNet created successfully")
        
        # Test forward pass
        x = torch.randn(1, 10, 128, 128, 128)
        polar_coords = torch.randn(1, 6, 128, 128, 128)
        
        with torch.no_grad():
            output = model(x, polar_coords)
            print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ EnhancedUNet test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Enhanced Polar Registration - Shape Fix Test")
    print("=" * 50)
    
    tests = [
        test_polar_coordinate_system,
        test_multi_scale_processor,
        test_enhanced_unet
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Shape mismatch issue is fixed.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
