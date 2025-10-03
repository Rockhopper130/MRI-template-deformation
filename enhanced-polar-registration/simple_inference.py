"""
Simplified Inference Script for Enhanced Polar Registration
Similar to the original inference.py but adapted for enhanced-polar-registration
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import os
import logging
from datetime import datetime

# Import enhanced components
from models.enhanced_unet import EnhancedUNet
from models.polar_transformer import PolarSpatialTransformer
from data.enhanced_dataset import EnhancedSegDataset
from losses.enhanced_losses import dice_loss


def setup_logging(log_dir="logs"):
    """Sets up logging for the inference script."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f'enhanced_inference_log_{timestamp}.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting enhanced polar registration inference...")
    return logger


def run_inference():
    """Main function to run the inference process."""
    parser = argparse.ArgumentParser(description='Run inference and calculate Dice loss on a trained enhanced polar registration model.')
    parser.add_argument('--val_txt', type=str,
                       default='/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/val.txt',
                       help='Path to the validation file list.')
    parser.add_argument('--template_path', type=str, 
                       default='/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/OASIS_OAS1_0406_MR1/seg4_onehot.npy', 
                       help='Path to the template segmentation map.')
    parser.add_argument('--model_path', type=str, 
                       default='/shared/home/v_nishchay_nilabh/code/seg-seg-reg/Cone-Method/enhanced-polar-registration/checkpoints/best_model.pth', 
                       help='Path to the saved best_model.pth file.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference.')
    parser.add_argument('--target_size', type=int, nargs=3, default=[64, 64, 64], 
                       help='Target size for resizing (D H W).')
    parser.add_argument('--use_polar_coords', action='store_true', default=True,
                       help='Whether to use polar coordinates.')
    args = parser.parse_args()

    logger = setup_logging()
    
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # --- Data Loading ---
    logger.info("Loading validation dataset...")
    try:
        val_dataset = EnhancedSegDataset(
            data_list_file=args.val_txt,
            template_path=args.template_path,
            target_size=tuple(args.target_size),
            use_polar_coords=args.use_polar_coords,
            use_augmentation=False,
            cache_data=False
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        logger.info(f"Validation dataset loaded with {len(val_dataset)} samples.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # --- Model Initialization ---
    logger.info("Initializing enhanced model...")
    model = EnhancedUNet(
        in_channels=10,  # 5 template + 5 fixed
        out_channels=3,  # 3D deformation field
        base_channels=16,
        num_scales=2,
        image_size=tuple(args.target_size),
        use_polar_processing=args.use_polar_coords,
        use_attention=False
    ).to(device)
    
    spatial_transformer = PolarSpatialTransformer(
        size=tuple(args.target_size),
        device=device,
        use_polar_coords=args.use_polar_coords,
        interpolation_mode='bilinear'
    ).to(device)

    # --- Load Pre-trained Weights ---
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Successfully loaded model weights from '{args.model_path}' (checkpoint format)")
        else:
            model.load_state_dict(checkpoint)
            logger.info(f"Successfully loaded model weights from '{args.model_path}' (direct format)")
    except FileNotFoundError:
        logger.error(f"Model file not found at '{args.model_path}'. Please check the path.")
        return
    except Exception as e:
        logger.error(f"An error occurred while loading the model: {e}")
        return

    # --- Inference Loop ---
    model.eval()  # Set the model to evaluation mode
    spatial_transformer.eval()
    total_dice_loss = 0.0
    dice_scores = []
    
    inference_pbar = tqdm(val_loader, desc='Evaluating on Validation Set')
    
    with torch.no_grad():  # Disable gradient calculations
        for batch_idx, batch in enumerate(inference_pbar):
            moving = batch['moving'].to(device)
            fixed = batch['fixed'].to(device)
            polar_coords = batch.get('polar_coords', None)
            if polar_coords is not None:
                polar_coords = polar_coords.to(device)
            
            # Create the input tensor for the model
            input_ = torch.cat([moving, fixed], dim=1)
            
            # 1. Predict the deformation field
            if args.use_polar_coords and polar_coords is not None:
                deformation_field = model(input_, polar_coords)
            else:
                deformation_field = model(input_)
            
            # 2. Warp the moving image using the spatial transformer
            if args.use_polar_coords and polar_coords is not None:
                warped_scan = spatial_transformer(moving, deformation_field, polar_coords)
            else:
                warped_scan = spatial_transformer(moving, deformation_field)
            
            # 3. Calculate the Dice loss between the warped and fixed images
            loss = dice_loss(warped_scan, fixed)
            
            # Dice Similarity Coefficient is (1 - Dice Loss)
            dice_score = 1 - loss.item()
            dice_scores.append(dice_score)
            
            total_dice_loss += loss.item()
            
            inference_pbar.set_postfix({'Dice Score': f'{dice_score:.4f}'})

    # --- Report Results ---
    avg_dice_loss = total_dice_loss / len(val_loader)
    avg_dice_score = np.mean(dice_scores)
    std_dice_score = np.std(dice_scores)
    
    logger.info("="*50)
    logger.info("Enhanced Polar Registration Inference Complete!")
    logger.info(f"Average Dice Loss: {avg_dice_loss:.6f}")
    logger.info(f"Average Dice Score (DSC): {avg_dice_score:.4f} ± {std_dice_score:.4f}")
    logger.info(f"Number of samples: {len(dice_scores)}")
    logger.info(f"Model: Enhanced UNet with Polar Coordinates: {args.use_polar_coords}")
    logger.info(f"Target size: {args.target_size}")
    logger.info("="*50)


if __name__ == '__main__':
    run_inference()
