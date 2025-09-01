import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _make_meshgrid(D, H, W, device, dtype):
    z = torch.arange(D, device=device, dtype=dtype)
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack([xx, yy, zz], dim=-1)
    return grid


def _vox_to_norm(grid_vox, D, H, W):
    x = grid_vox[..., 0]
    y = grid_vox[..., 1]
    z = grid_vox[..., 2]
    nx = 2.0 * x / (W - 1) - 1.0
    ny = 2.0 * y / (H - 1) - 1.0
    nz = 2.0 * z / (D - 1) - 1.0
    return torch.stack([nx, ny, nz], dim=-1)


def _warp_field(field, pos_vox, padding_mode='border', align_corners=True):
    # field: [B, D, H, W, 3]
    # pos_vox: [B, D, H, W, 3] (sampling positions in voxel coordinates)
    B, D, H, W, _ = field.shape
    field_perm = field.permute(0, 4, 1, 2, 3)  # [B, 3, D, H, W]
    grid_norm = _vox_to_norm(pos_vox, D, H, W)  # [B, D, H, W, 3]
    sampled = F.grid_sample(field_perm, grid_norm, mode='bilinear', padding_mode=padding_mode, align_corners=align_corners)
    sampled = sampled.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
    return sampled


class ScalingAndSquaring(nn.Module):
    """
    Integrates a stationary velocity field using the scaling-and-squaring method to produce
    a diffeomorphic transformation.

    Input
    -----
    v : torch.Tensor
        Stationary velocity field (or displacement field) in **voxel units**.
        Shape: [D, H, W, 3] or [B, D, H, W, 3]. The last dim order is (x, y, z) corresponding
        to width, height, depth axes respectively.

    Parameters
    ----------
    max_scale : float
        Target maximum magnitude (in voxels) after scaling. Default 0.5.
        The algorithm chooses n such that max(|v|)/2^n <= max_scale.
    padding_mode : str
        Padding mode for sampling (passed to grid_sample). Default 'border'.
    align_corners : bool
        Passed to grid_sample for coordinate normalization. Default True.

    Returns
    -------
    phi_map : torch.Tensor
        The integrated transformation map in **voxel coordinates** of shape [D, H, W, 3]
        (or [B, D, H, W, 3] for batched input). Each element gives the absolute coordinate
        phi(x) where x is the voxel index (i.e. phi(x) = x + displacement).

    Notes
    -----
    - This implementation uses trilinear interpolation via torch.nn.functional.grid_sample
      and is fully differentiable (works with autograd).
    - For large volumes you may want to increase memory by reducing batch size or using
      mixed precision.
    """

    def __init__(self, max_scale: float = 0.5, padding_mode: str = 'border', align_corners: bool = True):
        super().__init__()
        self.max_scale = float(max_scale)
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, v, n_steps: int = None, return_displacement: bool = False):
        # Accept both [D,H,W,3] and [B,D,H,W,3]
        batched = True
        if v.dim() == 4:
            v = v.unsqueeze(0)
            batched = False
        if v.dim() != 5 or v.shape[-1] != 3:
            raise ValueError('v must have shape [D,H,W,3] or [B,D,H,W,3] with last dim=3 (x,y,z)')

        B, D, H, W, _ = v.shape
        device = v.device
        dtype = v.dtype

        # compute maximum magnitude of the velocity (in voxels)
        max_mag = float(torch.norm(v, dim=-1).max().detach().cpu().item())

        if n_steps is None:
            if max_mag <= self.max_scale or max_mag == 0.0:
                n = 0
            else:
                n = int(math.ceil(math.log2(max_mag / self.max_scale)))
        else:
            n = int(n_steps)
            if n < 0:
                raise ValueError('n_steps must be non-negative')

        # scaling
        scale = 2 ** n
        phi = v / float(scale)

        # identity grid (voxel coordinates)
        base_grid = _make_meshgrid(D, H, W, device=device, dtype=dtype)  # [D,H,W,3]
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B,D,H,W,3]

        # iterative squaring (composition)
        for _ in range(n):
            pos = base_grid + phi  # sample locations in voxel coords
            sampled = _warp_field(phi, pos, padding_mode=self.padding_mode, align_corners=self.align_corners)
            phi = phi + sampled

        # final mapped coordinates phi_map = x + phi (absolute voxel coords)
        phi_map = base_grid + phi

        if not batched:
            phi_map = phi_map.squeeze(0)
            phi = phi.squeeze(0)

        return (phi_map, phi) if return_displacement else phi_map