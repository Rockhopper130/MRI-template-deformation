"""
Simple import test without PyTorch dependencies
"""

import sys
import os

def test_import_structure():
    """Test if all modules can be imported without PyTorch"""
    print("Testing import structure...")
    
    # Test basic imports
    try:
        from utils.config import load_config
        print("✓ Config module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    try:
        from utils.visualization import plot_training_curves
        print("✓ Visualization module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import visualization: {e}")
        return False
    
    # Test data module structure
    try:
        import data
        print("✓ Data module imported successfully")
        print(f"  Available classes: {data.__all__}")
    except ImportError as e:
        print(f"✗ Failed to import data module: {e}")
        return False
    
    # Test models module structure
    try:
        import models
        print("✓ Models module imported successfully")
        print(f"  Available classes: {models.__all__}")
    except ImportError as e:
        print(f"✗ Failed to import models module: {e}")
        return False
    
    # Test losses module structure
    try:
        import losses
        print("✓ Losses module imported successfully")
        print(f"  Available classes: {losses.__all__}")
    except ImportError as e:
        print(f"✗ Failed to import losses module: {e}")
        return False
    
    # Test training module structure
    try:
        import training
        print("✓ Training module imported successfully")
        print(f"  Available classes: {training.__all__}")
    except ImportError as e:
        print(f"✗ Failed to import training module: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        from utils.config import load_config
        
        # Test loading default config
        config = load_config('configs/default_config.json')
        print("✓ Default configuration loaded successfully")
        
        # Test configuration validation
        from utils.config import validate_config
        if validate_config(config):
            print("✓ Configuration validation passed")
        else:
            print("✗ Configuration validation failed")
            return False
            
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False
    
    return True

def main():
    """Run import tests"""
    print("Enhanced Polar Registration - Import Structure Test")
    print("=" * 60)
    
    tests = [
        test_import_structure,
        test_config_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Import Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All import tests passed! Module structure is correct.")
        print("\nNote: PyTorch-dependent modules will need PyTorch to be installed.")
        print("Install dependencies with: pip install -r requirements.txt")
        return True
    else:
        print("❌ Some import tests failed. Please check the module structure.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
