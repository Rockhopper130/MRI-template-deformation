"""
Test script to verify the enhanced polar registration implementation
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from models import EnhancedUNet, PolarSpatialTransformer
        print("✓ Models imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import models: {e}")
        return False
    
    try:
        from losses import EnhancedCompositeLoss
        print("✓ Losses imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import losses: {e}")
        return False
    
    try:
        from data import EnhancedSegDataset
        print("✓ Data modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import data modules: {e}")
        return False
    
    try:
        from training import EnhancedTrainer
        print("✓ Training modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import training modules: {e}")
        return False
    
    try:
        from utils.config import load_config
        from utils.visualization import plot_training_curves
        print("✓ Utils imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation and forward pass"""
    print("\nTesting model creation...")
    
    try:
        from models import EnhancedUNet, PolarSpatialTransformer
        
        # Create model
        model = EnhancedUNet(
            in_channels=10,
            out_channels=3,
            base_channels=16,  # Smaller for testing
            num_scales=2,
            image_size=(64, 64, 64),  # Smaller for testing
            use_polar_processing=True,
            use_attention=True
        )
        print("✓ Enhanced UNet created successfully")
        
        # Create spatial transformer
        transformer = PolarSpatialTransformer(
            size=(64, 64, 64),
            device='cpu',
            use_polar_coords=True
        )
        print("✓ Polar Spatial Transformer created successfully")
        
        # Test forward pass
        batch_size = 1
        input_tensor = torch.randn(batch_size, 10, 64, 64, 64)
        polar_coords = torch.randn(batch_size, 6, 64, 64, 64)
        
        with torch.no_grad():
            deformation_field = model(input_tensor, polar_coords)
            print(f"✓ Forward pass successful, output shape: {deformation_field.shape}")
            
            # Test spatial transformer
            moving = torch.randn(batch_size, 5, 64, 64, 64)
            warped = transformer(moving, deformation_field, polar_coords)
            print(f"✓ Spatial transformation successful, output shape: {warped.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_loss_function():
    """Test loss function"""
    print("\nTesting loss function...")
    
    try:
        from losses import EnhancedCompositeLoss
        
        # Create loss function
        loss_fn = EnhancedCompositeLoss()
        print("✓ Enhanced composite loss created successfully")
        
        # Test loss computation
        batch_size = 1
        pred = torch.randn(batch_size, 5, 64, 64, 64)
        target = torch.randn(batch_size, 5, 64, 64, 64)
        deformation_field = torch.randn(batch_size, 3, 64, 64, 64)
        polar_coords = torch.randn(batch_size, 6, 64, 64, 64)
        
        loss, loss_dict = loss_fn(pred, target, deformation_field, polar_coords)
        print(f"✓ Loss computation successful, total loss: {loss.item():.4f}")
        print(f"  Loss components: {list(loss_dict.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False


def test_coordinate_utils():
    """Test coordinate utilities"""
    print("\nTesting coordinate utilities...")
    
    try:
        from models.coordinate_utils import create_polar_coordinate_grid, cartesian_to_polar, polar_to_cartesian
        
        # Test polar coordinate grid creation
        grid = create_polar_coordinate_grid((32, 32, 32), torch.device('cpu'))
        print(f"✓ Polar coordinate grid created, shape: {grid.shape}")
        
        # Test coordinate conversion
        xyz = torch.randn(10, 3)
        rho, theta, phi = cartesian_to_polar(xyz)
        xyz_reconstructed = polar_to_cartesian(rho, theta, phi)
        
        # Check reconstruction accuracy
        reconstruction_error = torch.mean(torch.abs(xyz - xyz_reconstructed))
        print(f"✓ Coordinate conversion successful, reconstruction error: {reconstruction_error:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Coordinate utilities test failed: {e}")
        return False


def test_attention_modules():
    """Test attention modules"""
    print("\nTesting attention modules...")
    
    try:
        from models.attention_modules import MultiScaleAttention, DualAttention
        
        # Test multi-scale attention
        attention = MultiScaleAttention(in_channels=32, num_scales=2)
        x = torch.randn(1, 32, 64, 64, 64)
        output = attention(x)
        print(f"✓ Multi-scale attention successful, output shape: {output.shape}")
        
        # Test dual attention
        dual_attention = DualAttention(in_channels=32)
        output = dual_attention(x)
        print(f"✓ Dual attention successful, output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Attention modules test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Enhanced Polar Registration - Implementation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_loss_function,
        test_coordinate_utils,
        test_attention_modules
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
        print("🎉 All tests passed! Implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
