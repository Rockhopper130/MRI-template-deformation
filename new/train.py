import os
import random
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from tqdm import tqdm

# ADDED: For Automatic Mixed Precision
from torch.amp import GradScaler, autocast

# --- Assume these are your custom module imports ---
# These files should be in the same directory as this script.
from icosahedron_gen import IcosahedralGrid
from volumetric_surface_parameterization import VolumetricSphericalParameterization
from crsm import CRSM
from gat_stack import GATStack
from ico_blocks import IcoUNet
from s2c_head import S2CHead
from integration import ScalingAndSquaring
from spatial_transformer import SpatialTransformer
from losses import composite_loss
from dataloader import OASISDataset

# --- Helper functions from the notebook ---
def dict_to_pool_tensor(pool_dict, device=None):
    """Converts a dictionary-based pooling map to a tensor."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Nc = max(pool_dict.keys()) + 1
    k = max(len(v) for v in pool_dict.values())
    pool_tensor = torch.full((Nc, k), -1, dtype=torch.long, device=device)
    for c, fine_list in pool_dict.items():
        pool_tensor[c, :len(fine_list)] = torch.tensor(fine_list, dtype=torch.long, device=device)
    return pool_tensor

def dict_to_up_tensor(pool_dict, device=None):
    """Converts a dictionary-based pooling map to an upsampling tensor."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Nf = max(max(v) for v in pool_dict.values()) + 1
    up_tensor = torch.full((Nf,), -1, dtype=torch.long, device=device)
    for c, fine_list in pool_dict.items():
        for f in fine_list:
            up_tensor[f] = c
    return up_tensor

# --- Full Model Definition ---
class SphereMorphNet(nn.Module):
    """
    Encapsulates the entire SphereMorph-Net pipeline from volumetric input
    to the final displacement field.
    """
    def __init__(self, grid, device=None):
        super().__init__()
        self.grid = grid
        self.device = torch.device(device) if isinstance(device, str) else (device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # --- Pooling/Unpooling maps from grid ---
        pool_maps_full = [dict_to_pool_tensor(m, device=self.device) for m in reversed(grid.pool_maps)]
        up_maps_full = [m.to(self.device) for m in grid.up_maps]
        # FIXED: Adjusted channels to match actual pooling levels (6 levels for subdivisions=6)
        channels_full = [64, 128, 256, 512]

        # --- Model Components ---
        self.vsp = VolumetricSphericalParameterization(device=self.device)
        
        self.crsm = CRSM(
            radial_channel_indices=range(22),
            conical_depth_indices=[0, 5, 10, 11, 16, 21],
            aggregation='mean',
            mlp_hidden=(32,),
            mlp_out_dim=16
        ).to(self.device)
        
        self.gat = GATStack(
            in_channels=16,
            hidden_channels=32,
            out_channels=64,
            num_layers=4,
            heads=4,
            dropout=0.1
        ).to(self.device)
        
        self.ico_unet = model = IcoUNet(
            in_ch=64,
            channels=channels_full,
            pool_maps=pool_maps_full,
            up_maps=up_maps_full
        ).to(self.device)
        
        self.s2c_head = S2CHead(in_channels=64).to(self.device)
        self.integrator = ScalingAndSquaring(max_scale=0.5).to(self.device)
        self.full_res_size = (128, 128, 128)
        self.low_res_size = (64, 64, 64)


    def forward(self, moving_volume, fixed_volume):
        """
        Performs the full forward pass of the registration pipeline.
        """
        # 1. Volumetric to Spherical Parameterization
        moving_volume = moving_volume.to(self.device)
        fixed_volume = fixed_volume.to(self.device)

        # DataLoader with batch_size=1 yields volumes shaped [1, D, H, W].
        # VSP expects [D, H, W], so squeeze the batch dimension when present.
        if moving_volume.dim() == 4:
            if moving_volume.size(0) != 1:
                raise ValueError("Only batch_size=1 is supported for volumetric parameterization.")
            moving_volume = moving_volume.squeeze(0)
        if fixed_volume.dim() == 4:
            if fixed_volume.size(0) != 1:
                raise ValueError("Only batch_size=1 is supported for volumetric parameterization.")
            fixed_volume = fixed_volume.squeeze(0)

        grid_vertices = self.grid.vertices.to(self.device)
        grid_edge_index = self.grid.edge_index.to(self.device)

        data_moving = self.vsp(moving_volume, grid_vertices, grid_edge_index)
        data_fixed = self.vsp(fixed_volume, grid_vertices, grid_edge_index)

        x = torch.cat([data_moving.x, data_fixed.x], dim=1)
        data = Data(x=x, edge_index=grid_edge_index)

        # 2. Conical-Radial Sampling Module (CRSM)
        data_crsm = self.crsm(data).to(self.device)

        x_gat = self.gat(data_crsm.x, data_crsm.edge_index)
        data_gat = data_crsm
        data_gat.x = x_gat

        # 3. Icosahedral U-Net
        x_unet_in = data_gat.x.unsqueeze(0).permute(0, 2, 1).to(self.device)
        features_spherical = self.ico_unet(x_unet_in)

        # 4. Spherical-to-Cartesian (S2C) Head
        u_low_res = self.s2c_head(features_spherical.permute(0, 2, 1), grid_vertices, self.low_res_size)

        # 5. Upsample displacement field to full resolution
        u_permuted = u_low_res.permute(3, 0, 1, 2).unsqueeze(0)
        u_full_res = F.interpolate(u_permuted, size=self.full_res_size, mode='trilinear', align_corners=True)
        u_full_res = u_full_res.squeeze(0).permute(1, 2, 3, 0)

        # 6. Integrate velocity field to get diffeomorphic transformation
        # Integrator expects voxel-unit field; S2CHead generates voxel-space deltas already
        _, disp = self.integrator(u_full_res, return_displacement=True)

        return disp

# --- Main Training Script ---
def main():
    # --- Setup Logging ---
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/training_log_{timestamp}.txt'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also log to console for monitoring
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting training session...")
    
    # --- Configuration ---
    DATA_DIR = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/" # IMPORTANT: Update this path
    TEMPLATE_ID = "OASIS_OAS1_0406_MR1"

    # Check if data directory exists
    if not os.path.isdir(DATA_DIR):
        logger.error(f"Data directory not found at '{DATA_DIR}'")
        return

    LR = 1e-3  # CHANGED: A lower LR is often a good start with complex models
    EPOCHS = 40
    BATCH_SIZE = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Learning rate: {LR}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Template ID: {TEMPLATE_ID}")


    # --- Data Preparation ---
    try:
        with open(os.path.join(DATA_DIR, "train.txt"), "r") as f:
            train_ids = [line.strip() for line in f if line.strip()]
        with open(os.path.join(DATA_DIR, "val.txt"), "r") as f:
            val_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError as e:
        logger.error(f"Could not find {e.filename}. Please ensure train.txt and val.txt exist.")
        return

    if TEMPLATE_ID in train_ids: train_ids.remove(TEMPLATE_ID)
    if TEMPLATE_ID in val_ids: val_ids.remove(TEMPLATE_ID)

    logger.info(f"Training scans: {len(train_ids)} | Validation scans: {len(val_ids)}")

    template_path = os.path.join(DATA_DIR, TEMPLATE_ID, 'seg4_onehot.npy')

    train_dataset = OASISDataset(DATA_DIR, train_ids, template_path)
    val_dataset = OASISDataset(DATA_DIR, val_ids, template_path)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"Dataset loaded successfully")
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # --- Model Initialization ---
    grid = IcosahedralGrid(subdivisions=4)
    logger.info(f"Grid vertices shape: {grid.vertices.shape}")
    logger.info(f"Grid edge index shape: {grid.edge_index.shape}")
    model = SphereMorphNet(grid=grid, device=DEVICE).to(DEVICE)
        # ADDED: He Initialization
    def init_weights_he(m):
        """Applies He (Kaiming) initialization to Conv and Linear layers."""
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            # Use Kaiming normal initialization for weights
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # Initialize bias to zero
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    # Apply the initialization recursively to all modules
    model.apply(init_weights_he)
    logger.info("Model weights initialized with He initialization.")
    
    # Load existing best model weights if available
    loaded_best = False
    best_val_loss_loaded = float('inf')
    try:
        if os.path.exists('best_model.pth'):
            state_dict = torch.load('best_model.pth', map_location=DEVICE)
            model.load_state_dict(state_dict)
            loaded_best = True
            best_val_loss_loaded = 0.3118
            logger.info("Loaded weights from 'best_model.pth'. Resuming from best validation loss: 0.3118")
    except Exception as e:
        logger.warning(f"Could not load 'best_model.pth': {e}")
    
    stn = SpatialTransformer(size=(128, 128, 128), device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    logger.info("Model initialized successfully")
    logger.info(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ADDED: Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # CHANGED: Switched to a more proactive learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS * len(train_loader), # Total number of training steps
        eta_min=1e-5
    )

    max_grad_norm = 1.0

    # --- Training Loop ---
    best_val_loss = best_val_loss_loaded if loaded_best else float('inf')
    logger.info("Starting training loop...")
    logger.info(f"Training for {EPOCHS} epochs with {len(train_loader)} batches per epoch")
    
    # Track training time
    import time
    start_time = time.time()

    for epoch in range(EPOCHS):
        # -- Training Phase --
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for moving_vol, fixed_vol, moving_onehot, fixed_onehot in train_pbar:
            # moving_*: [C,D,H,W]; *_intensity: [D,H,W]
            optimizer.zero_grad(set_to_none=True) # More efficient

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16):
                disp_field = model(moving_vol, fixed_vol)  # [D,H,W,3] voxel units

            # REMOVED: Problematic manual displacement field normalization
            
            # Reshape for STN and Loss
            disp_batch_vox = disp_field.permute(3, 0, 1, 2).unsqueeze(0)  # [B,3,D,H,W]
            # Convert voxel offsets to normalized flow expected by grid_sample with align_corners=True
            D, H, W = disp_batch_vox.shape[2:5]
            scale = torch.tensor([2.0/(W-1), 2.0/(H-1), 2.0/(D-1)], device=disp_batch_vox.device, dtype=disp_batch_vox.dtype).view(1,3,1,1,1)
            flow_norm = disp_batch_vox * scale
            moving_batch = moving_onehot.float().unsqueeze(0).to(DEVICE)  # [B,C,D,H,W]
            fixed_batch = fixed_onehot.float().unsqueeze(0).to(DEVICE)    # [B,C,D,H,W]

            warped_scan = stn(moving=moving_batch, flow=flow_norm)
            # Compute losses on probabilities vs targets one-hot
            loss = composite_loss(probs=torch.clamp(warped_scan, 1e-6, 1.0), target_onehot=fixed_batch, flow=disp_batch_vox)

            # CHANGED: Scaler for mixed precision backpropagation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            # CHANGED: Scheduler is updated at each step
            scheduler.step()
            
            train_loss += float(loss.detach().cpu().item())
            train_pbar.set_postfix({'loss': loss.item()})


        avg_train_loss = train_loss / len(train_loader)

        # -- Validation Phase --
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for moving_vol, fixed_vol, moving_onehot, fixed_onehot in val_pbar:
                disp_field = model(moving_vol, fixed_vol)
                
                # REMOVED: Problematic manual displacement field normalization

                disp_batch_vox = disp_field.permute(3, 0, 1, 2).unsqueeze(0)
                D, H, W = disp_batch_vox.shape[2:5]
                scale = torch.tensor([2.0/(W-1), 2.0/(H-1), 2.0/(D-1)], device=disp_batch_vox.device, dtype=disp_batch_vox.dtype).view(1,3,1,1,1)
                flow_norm = disp_batch_vox * scale
                moving_batch = moving_onehot.float().unsqueeze(0).to(DEVICE)
                fixed_batch = fixed_onehot.float().unsqueeze(0).to(DEVICE)

                warped_scan = stn(moving=moving_batch, flow=flow_norm)
                loss = composite_loss(probs=torch.clamp(warped_scan, 1e-6, 1.0), target_onehot=fixed_batch, flow=disp_batch_vox)
            
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})

        avg_val_loss = val_loss / len(val_loader)

        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Log learning rate
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Current learning rate: {current_lr:.2e}")
        
        # Log memory usage if on CUDA
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(DEVICE) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(DEVICE) / 1024**3    # GB
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")

        # -- Save Best Model --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f"New best model saved with val loss: {best_val_loss:.4f}")

    logger.info("-> Training complete.")
    logger.info(f"Best validation loss achieved: {best_val_loss:.4f}")
    logger.info(f"Training completed successfully in {EPOCHS} epochs")
    logger.info(f"Log file saved to: {log_filename}")

    # Calculate and log training time
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Total training time: {training_time:.2f} seconds")


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    main()