import torch
from torch_geometric.data import Data

class VolumetricSphericalParameterization:
    def __init__(
        self,
        steps=200,
        fractions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99),
        mask_threshold=0.01,
        device=None
    ):
        self.steps = steps
        self.fractions = fractions
        self.mask_threshold = mask_threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, volume, vertices, edge_index, brain_mask=None):
        return self.run(volume, vertices, edge_index, brain_mask)

    def run(self, volume, vertices, edge_index, brain_mask=None):
        volume = volume.to(self.device).float()
        if brain_mask is None:
            brain_mask = (volume > 0).to(volume.dtype)
        else:
            brain_mask = brain_mask.to(self.device).float()

        D, H, W = volume.shape
        corners = torch.tensor(
            [[0,0,0],[0,0,W-1],[0,H-1,0],[0,H-1,W-1],
             [D-1,0,0],[D-1,0,W-1],[D-1,H-1,0],[D-1,H-1,W-1]],
            device=self.device, dtype=torch.float32
        )

        center_vox = self.compute_center_of_mass(brain_mask)
        distances = torch.norm(corners - center_vox.unsqueeze(0), dim=1)
        max_radius_val = float(distances.max().item()) * 1.05

        vertices = vertices.to(self.device).float()
        dirs = vertices / torch.norm(vertices, dim=1, keepdim=True).clamp(min=1e-6)

        center = self.compute_center_of_mass(brain_mask).to(self.device)
        t = torch.linspace(0.0, max_radius_val, self.steps, device=self.device, dtype=torch.float32).view(1, self.steps, 1)
        dirs_exp = dirs.view(-1, 1, 3)
        sample_points = center.view(1, 1, 3) + dirs_exp * t

        # mask sampling
        mask_vals = self.trilinear_interpolate(brain_mask, sample_points.view(-1, 3), D, H, W)
        mask_vals = mask_vals.view(-1, self.steps)
        inside = (mask_vals > self.mask_threshold).float()
        crossing = (inside[:, :-1] - inside[:, 1:] == 1.0)

        rho = torch.full((dirs.shape[0],), max_radius_val, device=self.device, dtype=torch.float32)
        idx_first = torch.argmax(crossing.to(torch.int64), dim=1)
        has_cross_idx = crossing.any(dim=1).nonzero(as_tuple=False).squeeze(1)

        if has_cross_idx.numel() > 0:
            rows = has_cross_idx
            cols = idx_first[rows]
            mask_i = mask_vals[rows, cols]
            mask_prev = mask_vals[rows, cols+1]
            denom = (mask_i - mask_prev).clamp(min=1e-6)
            frac = ((mask_i - self.mask_threshold) / denom).clamp(0.0, 1.0)
            rho_vals = (t.view(self.steps)[cols] * (1.0 - frac) + t.view(self.steps)[cols+1] * frac)
            rho[rows] = rho_vals

        rho = rho.clamp(max=max_radius_val)

        # texture sampling
        sample_fractions = torch.tensor(self.fractions, device=self.device, dtype=torch.float32).view(1, 1, -1)
        sample_t = rho.view(-1, 1, 1) * sample_fractions
        
        # center : [3], dirs_exp : [N, 1, 3], sample_t: [N, 1, 10] | N = #icosahedron_subdivisio_points
        # (N, 1, 3) * (N, 1, 10, 1) -> (N, 1, 10, 3)
        sample_points_tex = center.view(1, 1, 1, 3) + dirs_exp.unsqueeze(2) * sample_t.unsqueeze(-1)

        # drop the singleton dim -> (N, 10, 3)
        sample_points_tex = sample_points_tex.squeeze(1)

        # reshape to (N*10, 3) for interpolation
        intensities = self.trilinear_interpolate(volume, sample_points_tex.reshape(-1, 3), D, H, W)

        # back to (N, 10)
        intensities = intensities.view(dirs.shape[0], -1)

        features = torch.cat([rho.view(-1, 1), intensities], dim=1)
        return Data(x=features, edge_index=edge_index.to(self.device))

    @staticmethod
    def compute_center_of_mass(mask):
        device = mask.device
        D, H, W = mask.shape
        z = torch.arange(0, D, device=device, dtype=torch.float32).view(D, 1, 1)
        y = torch.arange(0, H, device=device, dtype=torch.float32).view(1, H, 1)
        x = torch.arange(0, W, device=device, dtype=torch.float32).view(1, 1, W)
        m = mask.sum()
        if m <= 0:
            return torch.tensor([D/2.0, H/2.0, W/2.0], device=device, dtype=torch.float32)
        cx = (mask * x).sum() / m
        cy = (mask * y).sum() / m
        cz = (mask * z).sum() / m
        return torch.stack([cz, cy, cx], dim=0)

    @staticmethod
    def trilinear_interpolate(vol, coords, D, H, W):
        coords = coords.clone()
        coords[:, 0] = coords[:, 0].clamp(0.0, D-1.0)
        coords[:, 1] = coords[:, 1].clamp(0.0, H-1.0)
        coords[:, 2] = coords[:, 2].clamp(0.0, W-1.0)

        z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
        z0, y0, x0 = z.floor().long(), y.floor().long(), x.floor().long()
        z1, y1, x1 = (z0 + 1).clamp(max=D-1), (y0 + 1).clamp(max=H-1), (x0 + 1).clamp(max=W-1)
        wz, wy, wx = (z - z0).unsqueeze(1), (y - y0).unsqueeze(1), (x - x0).unsqueeze(1)

        c000 = vol[z0, y0, x0]
        c001 = vol[z0, y0, x1]
        c010 = vol[z0, y1, x0]
        c011 = vol[z0, y1, x1]
        c100 = vol[z1, y0, x0]
        c101 = vol[z1, y0, x1]
        c110 = vol[z1, y1, x0]
        c111 = vol[z1, y1, x1]

        c00 = c000 * (1-wx).squeeze(1) + c001 * wx.squeeze(1)
        c01 = c010 * (1-wx).squeeze(1) + c011 * wx.squeeze(1)
        c10 = c100 * (1-wx).squeeze(1) + c101 * wx.squeeze(1)
        c11 = c110 * (1-wx).squeeze(1) + c111 * wx.squeeze(1)
        c0 = c00 * (1-wy).squeeze(1) + c01 * wy.squeeze(1)
        c1 = c10 * (1-wy).squeeze(1) + c11 * wy.squeeze(1)
        return c0 * (1-wz).squeeze(1) + c1 * wz.squeeze(1)
