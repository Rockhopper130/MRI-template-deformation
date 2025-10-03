"""
Enhanced Polar Registration - Inference Script
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import logging
import numpy as np
from tqdm import tqdm
import json

# Import enhanced components
from models import EnhancedUNet, PolarSpatialTransformer
from data import EnhancedSegDataset
from utils.config import load_config
from utils.visualization import visualize_registration, create_registration_gif


def setup_logging(log_dir: str):
    """Setup logging for inference"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'inference.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Load trained model"""
    config = load_config(config_path)
    
    # Create model
    model = EnhancedUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        base_channels=config['model']['base_channels'],
        num_scales=config['model']['num_scales'],
        image_size=tuple(config['data']['target_size']),
        use_polar_processing=config['model']['use_polar_processing'],
        use_attention=config['model']['use_attention']
    ).to(device)
    
    # Create spatial transformer
    spatial_transformer = PolarSpatialTransformer(
        size=tuple(config['data']['target_size']),
        device=device,
        use_polar_coords=config['model']['use_polar_processing'],
        interpolation_mode=config['model']['interpolation_mode']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    spatial_transformer.eval()
    
    return model, spatial_transformer, config


def run_inference(model, spatial_transformer, dataloader, device, config, save_dir):
    """Run inference on dataset"""
    logger = logging.getLogger(__name__)
    
    results = []
    total_dice = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Running inference')):
            # Move data to device
            moving = batch['moving'].to(device)
            fixed = batch['fixed'].to(device)
            polar_coords = batch.get('polar_coords', None)
            if polar_coords is not None:
                polar_coords = polar_coords.to(device)
            
            # Forward pass
            input_tensor = torch.cat([moving, fixed], dim=1)
            deformation_field = model(input_tensor, polar_coords)
            warped = spatial_transformer(moving, deformation_field, polar_coords)
            
            # Compute Dice score
            dice_score = compute_dice_score(warped, fixed)
            total_dice += dice_score
            num_samples += 1
            
            # Store results
            result = {
                'sample_idx': batch_idx,
                'dice_score': dice_score,
                'subject_path': batch.get('subject_path', f'sample_{batch_idx}')
            }
            results.append(result)
            
            # Save visualization for first few samples
            if batch_idx < 5:
                save_path = os.path.join(save_dir, f'registration_sample_{batch_idx}.png')
                visualize_registration(
                    moving[0], fixed[0], warped[0], deformation_field[0],
                    save_path=save_path
                )
                
                # Create GIF
                gif_path = os.path.join(save_dir, f'registration_sample_{batch_idx}.gif')
                create_registration_gif(
                    moving[0], fixed[0], warped[0],
                    save_path=gif_path
                )
            
            logger.info(f"Sample {batch_idx}: Dice Score = {dice_score:.4f}")
    
    # Compute average metrics
    avg_dice = total_dice / num_samples
    std_dice = np.std([r['dice_score'] for r in results])
    
    logger.info("="*50)
    logger.info("Inference Results:")
    logger.info(f"Average Dice Score: {avg_dice:.4f} ± {std_dice:.4f}")
    logger.info(f"Number of samples: {num_samples}")
    logger.info("="*50)
    
    return results, avg_dice, std_dice


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Dice score between prediction and target"""
    # Convert to class labels
    pred_labels = torch.argmax(pred, dim=1)
    target_labels = torch.argmax(target, dim=1)
    
    # Compute Dice for each class
    dice_scores = []
    for c in range(pred.shape[1]):
        pred_c = (pred_labels == c).float()
        target_c = (target_labels == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = 2 * intersection / union
        else:
            dice = torch.tensor(1.0 if pred_c.sum() == 0 and target_c.sum() == 0 else 0.0)
        
        dice_scores.append(dice.item())
    
    return np.mean(dice_scores)


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Enhanced Polar Registration Inference')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_txt', type=str, required=True, help='Path to data list file')
    parser.add_argument('--template_path', type=str, required=True, help='Path to template')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(args.output_dir)
    
    logger.info("Starting Enhanced Polar Registration Inference")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Data: {args.data_txt}")
    
    # Load model
    logger.info("Loading model...")
    model, spatial_transformer, config = load_model(args.config, args.checkpoint, device)
    
    # Create dataset
    dataset = EnhancedSegDataset(
        data_list_file=args.data_txt,
        template_path=args.template_path,
        target_size=tuple(config['data']['target_size']),
        use_polar_coords=config['model']['use_polar_processing'],
        use_augmentation=False,
        cache_data=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Run inference
    results, avg_dice, std_dice = run_inference(
        model, spatial_transformer, dataloader, device, config, args.output_dir
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'average_dice': avg_dice,
            'std_dice': std_dice,
            'num_samples': len(results)
        }, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")
    logger.info("Inference completed successfully!")


if __name__ == '__main__':
    main()
