# Enhanced Polar Registration

An advanced segmentation map registration system that combines traditional UNet architecture with polar coordinate processing, multi-scale attention mechanisms, and enhanced loss functions for improved medical image registration performance.

## 🚀 Key Features

### **Enhanced Architecture**
- **Multi-Scale Polar Processing**: Processes images at multiple scales with polar coordinate awareness
- **Advanced Attention Mechanisms**: Multi-scale attention, polar-aware attention, and adaptive attention
- **Enhanced UNet**: Improved UNet with residual connections, attention modules, and better feature fusion
- **Polar Spatial Transformer**: Advanced spatial transformer with polar coordinate constraints

### **Advanced Loss Functions**
- **Enhanced Composite Loss**: Combines multiple loss functions with optimal weighting
- **Polar Consistency Loss**: Ensures consistency in polar coordinate transformations
- **Anatomical Consistency Loss**: Maintains anatomical structure during registration
- **Multi-Scale Loss**: Evaluates registration quality at different resolutions
- **Gradient Consistency Loss**: Ensures smooth deformation fields

### **Robust Training**
- **Advanced Data Augmentation**: Polar-aware augmentations and geometric transformations
- **Enhanced Training Loop**: Mixed precision training, gradient clipping, and advanced scheduling
- **Comprehensive Monitoring**: Detailed logging, visualization, and metrics tracking
- **Early Stopping & Checkpointing**: Prevents overfitting and saves best models

## 📁 Project Structure

```
enhanced-polar-registration/
├── models/                     # Model architectures
│   ├── enhanced_unet.py       # Enhanced UNet with attention
│   ├── polar_transformer.py   # Polar-aware spatial transformer
│   ├── attention_modules.py   # Advanced attention mechanisms
│   └── coordinate_utils.py    # Polar coordinate utilities
├── losses/                    # Loss functions
│   └── enhanced_losses.py     # Advanced loss functions
├── data/                      # Data processing
│   ├── enhanced_dataset.py    # Enhanced dataset with augmentation
│   ├── data_augmentation.py   # Advanced augmentation techniques
│   └── data_utils.py          # Data utilities
├── training/                  # Training components
│   ├── enhanced_trainer.py    # Advanced trainer
│   └── training_utils.py      # Training utilities
├── utils/                     # Utilities
│   ├── config.py              # Configuration management
│   └── visualization.py       # Visualization tools
├── configs/                   # Configuration files
│   └── default_config.json    # Default configuration
├── main.py                    # Main training script
├── inference.py               # Inference script
└── README.md                  # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU training)

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib tqdm
pip install pyyaml imageio
pip install wandb  # Optional: for experiment tracking
```

## 🚀 Quick Start

### 1. Configuration
Create or modify the configuration file (`configs/default_config.json`):

```json
{
  "model": {
    "in_channels": 10,
    "out_channels": 3,
    "base_channels": 32,
    "num_scales": 3,
    "use_polar_processing": true,
    "use_attention": true
  },
  "data": {
    "train_txt": "/path/to/train.txt",
    "val_txt": "/path/to/val.txt",
    "template_path": "/path/to/template.npy",
    "target_size": [128, 128, 128],
    "use_augmentation": true
  },
  "training": {
    "epochs": 100,
    "batch_size": 1,
    "learning_rate": 1e-3,
    "optimizer": "adamw"
  }
}
```

### 2. Training
```bash
python main.py --config configs/default_config.json --device cuda:0
```

### 3. Inference
```bash
python inference.py \
    --config configs/default_config.json \
    --checkpoint checkpoints/best_model.pth \
    --data_txt /path/to/test.txt \
    --template_path /path/to/template.npy \
    --output_dir ./results
```

## 🔧 Configuration Options

### Model Configuration
- `in_channels`: Number of input channels (template + fixed)
- `out_channels`: Number of output channels (deformation field)
- `base_channels`: Base number of channels in UNet
- `num_scales`: Number of scales for multi-scale processing
- `use_polar_processing`: Enable polar coordinate processing
- `use_attention`: Enable attention mechanisms

### Data Configuration
- `target_size`: Target image size [D, H, W]
- `use_augmentation`: Enable data augmentation
- `augmentation_prob`: Probability of applying augmentation
- `cache_data`: Cache data in memory for faster loading
- `use_multi_scale`: Enable multi-scale dataset
- `use_balanced_sampling`: Enable balanced class sampling

### Training Configuration
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate
- `optimizer`: Optimizer type (adam, adamw, sgd)
- `scheduler_type`: Learning rate scheduler (cosine, step, plateau)
- `early_stopping_patience`: Patience for early stopping
- `grad_clip_norm`: Gradient clipping norm

### Loss Configuration
- `dice_weight`: Weight for Dice loss
- `ce_weight`: Weight for cross-entropy loss
- `bending_weight`: Weight for bending energy loss
- `jacobian_weight`: Weight for Jacobian determinant loss
- `polar_weight`: Weight for polar consistency loss
- `anatomical_weight`: Weight for anatomical consistency loss
- `gradient_weight`: Weight for gradient consistency loss

## 📊 Key Improvements Over Baseline

### 1. **Polar Coordinate Integration**
- **Multi-scale polar processing** for better geometric understanding
- **Polar-aware attention mechanisms** for improved feature learning
- **Polar consistency losses** to maintain geometric constraints

### 2. **Advanced Architecture**
- **Enhanced UNet** with residual connections and attention
- **Multi-scale processing** for better feature extraction
- **Adaptive attention** that learns optimal attention strategies

### 3. **Improved Loss Functions**
- **Enhanced composite loss** with better weighting
- **Anatomical consistency loss** for structure preservation
- **Multi-scale loss** for better registration quality

### 4. **Robust Training**
- **Advanced data augmentation** with polar awareness
- **Mixed precision training** for efficiency
- **Comprehensive monitoring** and visualization

## 📈 Performance Metrics

The enhanced system provides significant improvements over the baseline:

- **Better Registration Accuracy**: Improved Dice scores and registration quality
- **Faster Convergence**: Advanced training strategies reduce training time
- **More Robust**: Better handling of challenging cases and variations
- **Better Generalization**: Enhanced augmentation and regularization

## 🔍 Visualization

The system includes comprehensive visualization tools:

- **Training Curves**: Monitor training progress
- **Registration Results**: Visualize registration quality
- **Polar Coordinates**: Understand coordinate transformations
- **Feature Maps**: Analyze learned features
- **Deformation Fields**: Visualize spatial transformations

## 🧪 Advanced Features

### Multi-Scale Processing
Process images at multiple resolutions simultaneously for better feature learning and registration quality.

### Polar Coordinate Awareness
Integrate polar coordinate information throughout the network for better geometric understanding.

### Adaptive Attention
Learn optimal attention strategies that adapt to different registration scenarios.

### Enhanced Data Augmentation
Polar-aware augmentations that maintain geometric consistency while increasing data diversity.

## 📝 Usage Examples

### Custom Configuration
```python
from utils.config import load_config, create_experiment_config

# Load base configuration
config = load_config('configs/default_config.json')

# Create custom experiment
custom_config = create_experiment_config(
    'configs/default_config.json',
    'my_experiment',
    overrides={
        'model': {'base_channels': 64},
        'training': {'learning_rate': 2e-3}
    }
)
```

### Custom Loss Function
```python
from losses import EnhancedCompositeLoss

# Create custom loss with different weights
custom_loss = EnhancedCompositeLoss(
    dice_weight=0.5,
    polar_weight=0.15,
    anatomical_weight=0.15
)
```

### Visualization
```python
from utils.visualization import visualize_registration

# Visualize registration results
visualize_registration(
    moving, fixed, warped, deformation_field,
    save_path='results/registration.png'
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built upon the original segmentation map registration baseline
- Incorporates advanced techniques from recent medical image registration research
- Uses polar coordinate systems for improved geometric understanding

## 📞 Support

For questions, issues, or contributions, please open an issue on the project repository.
