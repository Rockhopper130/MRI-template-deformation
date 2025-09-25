import torch
import torch.nn as nn
import math

def cartesian_to_spherical(xyz):
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    rho = torch.sqrt(x * x + y * y + z * z)
    rho_clamped = torch.clamp(rho, min=1e-12)
    theta = torch.acos(torch.clamp(z / rho_clamped, -1.0, 1.0))
    phi = torch.atan2(y, x)
    return rho, theta, phi


def spherical_unit_vectors(theta, phi):
    st = torch.sin(theta)
    ct = torch.cos(theta)
    cp = torch.cos(phi)
    sp = torch.sin(phi)
    e_r = torch.stack([st * cp, st * sp, ct], dim=-1)
    e_theta = torch.stack([ct * cp, ct * sp, -st], dim=-1)
    e_phi = torch.stack([-sp, cp, torch.zeros_like(sp)], dim=-1)
    return e_r, e_theta, e_phi


class S2CHead(nn.Module):
    def __init__(self, in_channels, k=4, chunk_voxels=200000, scale_factor=0.1):
        super().__init__()
        self.proj = nn.Linear(in_channels, 3)
        self.k = k
        self.chunk_voxels = chunk_voxels
        self.scale_factor = scale_factor  # Scale factor to keep displacement field in reasonable range

    def forward(self, features, vertex_pos_cartesian, out_size, voxel_origin=None, voxel_spacing=None):
        device = features.device
        if features.dim() == 3:
            batch = True
            b, V, C = features.shape
        else:
            batch = False
            V, C = features.shape
        d_sph = self.proj(features.view(-1, features.shape[-1]))
        if batch:
            d_sph = d_sph.view(b, V, 3)
        else:
            d_sph = d_sph.view(V, 3)
        verts = vertex_pos_cartesian.to(device).view(-1, 3)
        vert_rho, vert_theta, vert_phi = cartesian_to_spherical(verts)
        vert_unit = spherical_unit_vectors(vert_theta, vert_phi)[0]
        max_rho = float(vert_rho.max().item())
        D, H, W = out_size
        if voxel_origin is None or voxel_spacing is None:
            # Assume voxel index grid centered at zero in normalized sense; we build voxel coordinates
            xs = torch.arange(W, device=device, dtype=verts.dtype)
            ys = torch.arange(H, device=device, dtype=verts.dtype)
            zs = torch.arange(D, device=device, dtype=verts.dtype)
        else:
            xs = voxel_origin[0] + torch.arange(W, device=device, dtype=verts.dtype) * voxel_spacing[0]
            ys = voxel_origin[1] + torch.arange(H, device=device, dtype=verts.dtype) * voxel_spacing[1]
            zs = voxel_origin[2] + torch.arange(D, device=device, dtype=verts.dtype) * voxel_spacing[2]
        # Use ij indexing to match (z,y,x) order, then build (x,y,z) voxel coords and center them
        grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing='ij')
        vox_coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], dim=1)
        center_xyz = torch.tensor([(W - 1) * 0.5, (H - 1) * 0.5, (D - 1) * 0.5], device=device, dtype=verts.dtype)
        vox_coords = vox_coords - center_xyz
        N = vox_coords.shape[0]
        rho_v, theta_v, phi_v = cartesian_to_spherical(vox_coords)
        e_r_v, e_theta_v, e_phi_v = spherical_unit_vectors(theta_v, phi_v)
        vox_unit = e_r_v
        out = torch.empty((N, 3), device=device)
        vert_unit_t = vert_unit.t()
        neighbor_k = min(self.k, V)
        start = 0
        eps = 1e-8
        while start < N:
            end = min(start + self.chunk_voxels, N)
            chunk = vox_unit[start:end]
            sim = torch.matmul(chunk, vert_unit_t)
            sim = torch.clamp(sim, -1.0, 1.0)
            if neighbor_k == V:
                idx = torch.arange(V, device=device).unsqueeze(0).expand(chunk.shape[0], V)
                dots = sim
            else:
                dots, idx = torch.topk(sim, neighbor_k, dim=1)
            ang_dist = 1.0 - dots
            weights = 1.0 / (ang_dist + eps)
            weights = weights / weights.sum(dim=1, keepdim=True)
            if batch:
                # d_sph: [b, V, 3], idx: [M, k]
                dchunk = torch.sum(weights.unsqueeze(0).unsqueeze(-1) * d_sph[:, idx, :], dim=2)
            else:
                dchunk = torch.sum(weights.unsqueeze(-1) * d_sph[idx], dim=1)
            d_r = dchunk[..., 0]
            d_theta = dchunk[..., 1]
            d_phi = dchunk[..., 2]
            rho_local = rho_v[start:end]
            sin_theta = torch.sin(theta_v[start:end])
            delta = (
                d_r.unsqueeze(-1) * e_r_v[start:end]
                + rho_local.unsqueeze(-1) * d_theta.unsqueeze(-1) * e_theta_v[start:end]
                + rho_local.unsqueeze(-1) * sin_theta.unsqueeze(-1) * d_phi.unsqueeze(-1) * e_phi_v[start:end]
            )
            out[start:end] = delta
            start = end
        out = out.view(D, H, W, 3)
        # Apply scaling factor to keep displacement field in reasonable range
        out = out * self.scale_factor
        return out