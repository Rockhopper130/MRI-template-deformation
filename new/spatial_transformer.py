
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """
    A true dense STN for 3D one-hot maps, with an identity grid buffer
    and nearest‐neighbor sampling for crisp labels.
    """
    def __init__(self, size, device='cpu'):
        """
        Args:
            size: tuple of ints (D, H, W)
            device: tensor device
        """
        super().__init__()
        D, H, W = size
 
        lin_z = torch.linspace(-1, 1, D, device=device)
        lin_y = torch.linspace(-1, 1, H, device=device)
        lin_x = torch.linspace(-1, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(lin_z, lin_y, lin_x, indexing='ij')

        id_grid = torch.stack((xx, yy, zz), dim=-1)
        self.register_buffer('id_grid', id_grid.unsqueeze(0))

    def forward(self, moving, flow):
        """
        Args:
            moving: (B, C, D, H, W) one‐hot template
            flow:   (B, 3, D, H, W) displacement in normalized coords
        Returns:
            warped: (B, C, D, H, W) one‐hot warped template
        """
        B, C, D, H, W = moving.shape
      
        flow = flow.permute(0, 2, 3, 4, 1)
   
        grid = self.id_grid.expand(B, -1, -1, -1, -1)
    
        warped_grid = grid + flow
      
        warped = F.grid_sample(
            moving, warped_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        return warped
