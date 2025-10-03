import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from datetime import datetime  # Added for timestamping log files

# --- Assume these are your custom module imports ---
# Make sure these files are in the same directory as this script.
from icosahedron_gen import IcosahedralGrid
from volumetric_surface_parameterization import VolumetricSphericalParameterization
from crsm import CRSM
from gat_stack import GATStack
from ico_blocks import IcoUNet
from s2c_head import S2CHead
from integration import ScalingAndSquaring
from spatial_transformer import SpatialTransformer
from losses import dice_loss # Specifically import dice_loss
from dataloader import OASISDataset
from torch_geometric.data import Data

# --- Helper functions and Model Definition ---
# (Copied from your training script to make this self-contained)

def dict_to_pool_tensor(pool_dict, device="cuda:4"):
    """Converts a dictionary-based pooling map to a tensor."""
    Nc = max(pool_dict.keys()) + 1
    k = max(len(v) for v in pool_dict.values())
    pool_tensor = torch.full((Nc, k), -1, dtype=torch.long, device=device)
    for c, fine_list in pool_dict.items():
        pool_tensor[c, :len(fine_list)] = torch.tensor(fine_list, dtype=torch.long, device=device)
    return pool_tensor

class SphereMorphNet(nn.Module):
    """
    Encapsulates the entire SphereMorph-Net pipeline from volumetric input
    to the final displacement field.
    """
    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        self.device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

        pool_maps_full = [dict_to_pool_tensor(m, device=self.device) for m in reversed(grid.pool_maps)]
        up_maps_full = [m.to(self.device) for m in grid.up_maps]
        channels_full = [64, 128, 256, 512]

        self.vsp = VolumetricSphericalParameterization(device=self.device)
        self.crsm = CRSM(radial_channel_indices=range(22), conical_depth_indices=[0, 5, 10, 11, 16, 21], aggregation='mean', mlp_hidden=(32,), mlp_out_dim=16).to(self.device)
        self.gat = GATStack(in_channels=16, hidden_channels=32, out_channels=64, num_layers=4, heads=4, dropout=0.1).to(self.device)
        self.ico_unet = IcoUNet(in_ch=64, channels=channels_full, pool_maps=pool_maps_full, up_maps=up_maps_full).to(self.device)
        self.s2c_head = S2CHead(in_channels=64).to(self.device)
        self.integrator = ScalingAndSquaring(max_scale=0.5).to(self.device)
        
        self.full_res_size = (128, 128, 128)
        self.low_res_size = (64, 64, 64)

    def forward(self, moving_volume, fixed_volume):
        moving_volume = moving_volume.to(self.device)
        fixed_volume = fixed_volume.to(self.device)

        grid_vertices = self.grid.vertices.to(self.device)
        grid_edge_index = self.grid.edge_index.to(self.device)

        data_moving = self.vsp(moving_volume, grid_vertices, grid_edge_index)
        data_fixed = self.vsp(fixed_volume, grid_vertices, grid_edge_index)

        x = torch.cat([data_moving.x, data_fixed.x], dim=1)
        data = Data(x=x, edge_index=grid_edge_index)

        data_crsm = self.crsm(data).to(self.device)
        x_gat = self.gat(data_crsm.x, data_crsm.edge_index)
        data_gat = data_crsm
        data_gat.x = x_gat

        x_unet_in = data_gat.x.unsqueeze(0).permute(0, 2, 1).to(self.device)
        features_spherical = self.ico_unet(x_unet_in)

        u_low_res = self.s2c_head(features_spherical.permute(0, 2, 1), grid_vertices, self.low_res_size)

        u_permuted = u_low_res.permute(3, 0, 1, 2).unsqueeze(0)
        u_full_res = F.interpolate(u_permuted, size=self.full_res_size, mode='trilinear', align_corners=False)
        u_full_res = u_full_res.squeeze(0).permute(1, 2, 3, 0)
        
        _, disp = self.integrator(u_full_res, return_displacement=True)
        return disp

# --- Main Inference Function ---
def run_inference():
    """
    Loads the best model and computes the average Dice score on the validation set.
    """
    # --- Configuration ---
    DATA_DIR = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/"
    TEMPLATE_ID = "OASIS_OAS1_0406_MR1"
    BATCH_SIZE = 1
    DEVICE = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = 'best_model.pth'

    # --- MODIFICATION: Setup Logging to File and Console ---
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/inference_log_{timestamp}.txt'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler() # Also log to console
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting inference...")
    logger.info(f"Using device: {DEVICE}")

    # --- Data Preparation ---
    try:
        with open(os.path.join(DATA_DIR, "val.txt"), "r") as f:
            val_ids = [line.strip() for line in f if line.strip()]
        if TEMPLATE_ID in val_ids: val_ids.remove(TEMPLATE_ID)
    except FileNotFoundError as e:
        logger.error(f"Could not find {e.filename}. Please ensure val.txt exists.")
        return

    template_path = os.path.join(DATA_DIR, TEMPLATE_ID, 'seg4_onehot.npy')
    val_dataset = OASISDataset(DATA_DIR, val_ids, template_path)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"Validation dataset loaded with {len(val_dataset)} samples.")

    # --- Model Initialization and Loading ---
    grid = IcosahedralGrid(subdivisions=4)
    model = SphereMorphNet(grid=grid).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at '{MODEL_PATH}'")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    logger.info(f"Successfully loaded model weights from '{MODEL_PATH}'")
    
    stn = SpatialTransformer(size=(128, 128, 128), device=DEVICE)

    # --- Evaluation Loop ---
    dice_scores = []
    val_pbar = tqdm(val_loader, desc="Evaluating on Validation Set")

    with torch.no_grad():
        for moving_vol, fixed_vol in val_pbar:
            moving_input = moving_vol.squeeze(0).to(DEVICE)
            fixed_input = fixed_vol.squeeze(0).to(DEVICE)

            disp_field = model(moving_input, fixed_input)
            disp_batch = disp_field.permute(3, 0, 1, 2).unsqueeze(0)
            
            # STN expects (B, C, D, H, W). Input is (B, D, H, W), so we add a channel dim.
            moving_batch_for_stn = moving_vol.unsqueeze(1).to(DEVICE)
            warped_scan = stn(moving=moving_batch_for_stn, flow=disp_batch)
            
            # dice_loss also expects channel dimension
            fixed_batch_for_loss = fixed_vol.unsqueeze(1).to(DEVICE)
            loss = dice_loss(warped_scan, fixed_batch_for_loss)
            
            dice_score = 1.0 - loss.item()
            dice_scores.append(dice_score)

    # --- Report Results ---
    avg_dice_score = np.mean(dice_scores)
    std_dice_score = np.std(dice_scores)
    
    logger.info("--- Inference Complete ---")
    logger.info(f"Average Dice Score on Validation Set: {avg_dice_score:.4f} ± {std_dice_score:.4f}")
    
    # --- MODIFICATION: Log the output file path ---
    logger.info(f"Log file saved to: {log_filename}")

if __name__ == '__main__':
    run_inference()