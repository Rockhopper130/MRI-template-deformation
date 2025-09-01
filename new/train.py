import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from tqdm import tqdm

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
def dict_to_pool_tensor(pool_dict, device="cuda"):
    """Converts a dictionary-based pooling map to a tensor."""
    Nc = max(pool_dict.keys()) + 1
    k = max(len(v) for v in pool_dict.values())
    pool_tensor = torch.full((Nc, k), -1, dtype=torch.long, device=device)
    for c, fine_list in pool_dict.items():
        pool_tensor[c, :len(fine_list)] = torch.tensor(fine_list, dtype=torch.long, device=device)
    return pool_tensor

def dict_to_up_tensor(pool_dict, device="cuda"):
    """Converts a dictionary-based pooling map to an upsampling tensor."""
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
    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # --- Pooling/Unpooling maps from grid ---
        pool_map_162_to_42 = self.grid.pool_maps[1]
        pool_map_42_to_12 = self.grid.pool_maps[0]
        
        pool_tensor_lvl0 = dict_to_pool_tensor(pool_map_162_to_42, device=self.device)
        pool_tensor_lvl1 = dict_to_pool_tensor(pool_map_42_to_12, device=self.device)
        up_tensor_lvl0 = dict_to_up_tensor(pool_map_162_to_42, device=self.device)
        up_tensor_lvl1 = dict_to_up_tensor(pool_map_42_to_12, device=self.device)
        
        pool_maps = [pool_tensor_lvl0, pool_tensor_lvl1]
        up_maps = [up_tensor_lvl0, up_tensor_lvl1]

        # --- Model Components ---
        self.vsp = VolumetricSphericalParameterization()
        self.crsm = CRSM(
            radial_channel_indices=range(22), 
            conical_depth_indices=[0, 5, 10, 11, 16, 21], 
            aggregation='mean', 
            mlp_hidden=(32,), 
            mlp_out_dim=16
        )
        self.gat = GATStack(
            in_channels=16, 
            hidden_channels=32,
            out_channels=64,
            num_layers=4, 
            heads=4, 
            dropout=0.1
        ).to(self.device)
        self.ico_unet = IcoUNet(
            in_ch=64,
            channels=[32, 16],
            pool_maps=pool_maps,
            up_maps=up_maps
        )
        self.s2c_head = S2CHead(in_channels=16)
        self.integrator = ScalingAndSquaring(max_scale=0.5)
        self.full_res_size = (128, 128, 128) # Using smaller res for faster training
        self.low_res_size = (64, 64, 64)


    def forward(self, moving_volume, fixed_volume):
        """
        Performs the full forward pass of the registration pipeline.
        """
        # 1. Volumetric to Spherical Parameterization
        moving_volume = moving_volume.to(self.device)
        fixed_volume = fixed_volume.to(self.device)
        
        # Ensure grid vertices and edges are on the correct device
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
        # Reshape for IcoUNet: [Batch, Features, Num_Vertices]
        x_unet_in = data_gat.x.unsqueeze(0).permute(0, 2, 1).to(self.device)
        
        features_spherical = self.ico_unet(x_unet_in)

        # 4. Spherical-to-Cartesian (S2C) Head
        u_low_res = self.s2c_head(features_spherical.permute(0, 2, 1), grid_vertices, self.low_res_size)

        # 5. Upsample displacement field to full resolution
        u_permuted = u_low_res.permute(3, 0, 1, 2).unsqueeze(0)
        u_full_res = F.interpolate(u_permuted, size=self.full_res_size, mode='trilinear', align_corners=False)
        u_full_res = u_full_res.squeeze(0).permute(1, 2, 3, 0)

        # 6. Integrate velocity field to get diffeomorphic transformation
        _, disp = self.integrator(u_full_res, return_displacement=True)
        
        return disp


# --- Main Training Script ---
def main():
    # --- Configuration ---
    DATA_DIR = "/shared/scratch/0/home/v_nishchay_nilabh/oasis_data/scans/" # IMPORTANT: Update this path
    TEMPLATE_ID = "OASIS_OAS1_0406_MR1"
    
    # Check if data directory exists
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory not found at '{DATA_DIR}'")
        print("Please update the DATA_DIR variable in the script.")
        return

    LR = 1e-3
    EPOCHS = 100
    BATCH_SIZE = 1 # Model process one pair at a time
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # --- Data Preparation ---
    try:
        with open(os.path.join(DATA_DIR, "train.txt"), "r") as f:
            train_ids = [line.strip() for line in f if line.strip()]
        with open(os.path.join(DATA_DIR, "val.txt"), "r") as f:
            val_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError as e:
        print(f"Error: Could not find {e.filename}. Please ensure train.txt and val.txt exist.")
        return

    # Exclude the template from the lists of moving scans just in case
    if TEMPLATE_ID in train_ids:
        train_ids.remove(TEMPLATE_ID)
    if TEMPLATE_ID in val_ids:
        val_ids.remove(TEMPLATE_ID)

    print(f"Total scans: {len(train_ids) + len(val_ids)+1} | Training scans: {len(train_ids)} | Validation scans: {len(val_ids)}")

    template_path = os.path.join(DATA_DIR, TEMPLATE_ID, 'seg4_onehot.npy')

    train_dataset = OASISDataset(DATA_DIR, train_ids, template_path)
    val_dataset = OASISDataset(DATA_DIR, val_ids, template_path)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Model Initialization ---
    grid = IcosahedralGrid(subdivisions=2)
    model = SphereMorphNet(grid=grid).to(DEVICE)
    stn = SpatialTransformer(size=(128, 128, 128), device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- Training Loop ---
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        # -- Training Phase --
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for moving_vol, fixed_vol in train_pbar:
            # Dataloader batching adds an extra dimension, remove it
            moving_vol = moving_vol.squeeze(0)
            fixed_vol = fixed_vol.squeeze(0)

            optimizer.zero_grad()
            
            # Forward pass
            disp_field = model(moving_vol, fixed_vol) # -> [D,H,W,3]
            
            # Reshape for STN and Loss
            disp_batch = disp_field.permute(3, 0, 1, 2).unsqueeze(0) # -> [1,3,D,H,W]
            moving_batch = moving_vol.float().unsqueeze(0).unsqueeze(0).to(DEVICE) # -> [1,1,D,H,W]
            fixed_batch = fixed_vol.float().to(DEVICE) # -> [D,H,W]
            
            # print("\n[DEBUG] Shapes and Devices")
            # print(" moving_vol :", moving_vol.shape, moving_vol.device, moving_vol.dtype)
            # print(" fixed_vol  :", fixed_vol.shape, fixed_vol.device, fixed_vol.dtype)
            # print(" disp_field :", disp_field.shape, disp_field.device, disp_field.dtype)
            # print(" moving_batch:", moving_batch.shape, moving_batch.device)
            # print(" disp_batch  :", disp_batch.shape, disp_batch.device)
            # print(" fixed_batch :", fixed_batch.shape, fixed_batch.device)
            
            # print("[DEBUG] Value Ranges")
            # print(" moving_vol min/max:", moving_vol.min().item(), moving_vol.max().item())
            # print(" fixed_vol  min/max:", fixed_vol.min().item(), fixed_vol.max().item())
            # print(" disp_field min/max:", disp_field.min().item(), disp_field.max().item())

            warped_scan = stn(moving=moving_batch, flow=disp_batch) # -> [1,1,D,H,W]
            
            # print("[DEBUG] warped_scan :", warped_scan.shape, warped_scan.device)
            # print(" warped_scan min/max:", warped_scan.min().item(), warped_scan.max().item())
            # print(" fixed_batch vs warped_scan:", fixed_batch.shape, warped_scan.squeeze(0).squeeze(0).shape)
            
            # Loss calculation
            loss = composite_loss(fixed_batch, warped_scan.squeeze(0).squeeze(0), disp_batch)
            # print("[DEBUG] loss:", loss.item())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # -- Validation Phase --
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for moving_vol, fixed_vol in val_pbar:
                moving_vol = moving_vol.squeeze(0)
                fixed_vol = fixed_vol.squeeze(0)
                
                disp_field = model(moving_vol, fixed_vol)
                disp_batch = disp_field.permute(3, 0, 1, 2).unsqueeze(0)
                moving_batch = moving_vol.float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                fixed_batch = fixed_vol.float().to(DEVICE)
                
                warped_scan = stn(moving=moving_batch, flow=disp_batch)
                loss = composite_loss(fixed_batch, warped_scan.squeeze(0).squeeze(0), disp_batch)
                
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # -- Save Best Model --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"-> New best model saved with val loss: {best_val_loss:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    print("-> Loss plot saved to loss_plot.png")


if __name__ == '__main__':
    # Set a seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    main()
