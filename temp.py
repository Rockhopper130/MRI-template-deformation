"""
SCoRe-Net: Spherical Cone-Equivariant Registration Network
Target runtime: PyTorch 2.8.0+cu128, CUDA Toolkit 12.8

This is a reference implementation that avoids deprecated or version-fragile APIs.
It uses:
  - torch (2.8)
  - e3nn (tested with >=0.6; only stable, widely available modules are used)
  - torch_geometric (>=2.4 recommended, but code does not depend on brittle ops)

Main modules:
  - IcoSphere: deterministic icosahedral tessellation + hierarchy
  - SphericalPreprocessor: center → spherical sample → graph construction
  - EquivariantEdgeConv: SO(3)-equivariant message passing via e3nn TensorProduct
  - EquivariantUNet: Siamese encoder/decoder on spherical graphs
  - SirenDecoder: continuous deformation field with FiLM conditioning
  - ScoreNet: end-to-end registration model
  - Losses: LNCC (local NCC), smoothness via ∥∇u∥², Jacobian folding penalty

Notes
-----
* All functions/classes are self-contained and use standard, available APIs from
  torch, e3nn, and (optionally) torch_geometric. If torch_geometric is not
  installed, the code falls back to an internal kNN graph builder.
* The e3nn parts use o3.Irreps, o3.spherical_harmonics, and o3.TensorProduct
  combined with a small MLP for weights (a stable, well-supported pattern).
* Pooling/upsampling on the sphere is implemented using the deterministic
  icosahedral hierarchy provided by IcoSphere, avoiding deprecated pyg ops.
* You can train directly on point clouds (x,y,z,intensity). For voxel MRI, sample
  a surface point cloud (e.g., cortical mesh or isosurface) before using this model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    _HAVE_PYG = True
except Exception:
    _HAVE_PYG = False
    Data = object  # Minimal stub so type hints still work

from e3nn import o3
from e3nn.nn import FullyConnectedNet  # stable MLP helper in e3nn


# -----------------------------
# Geometry: Icosahedral sphere
# -----------------------------

class IcoSphere:
    """Generates an icosahedral sphere and hierarchical vertex mappings.

    Provides (V_l, F_l) for levels l=0..L, plus a parent map from level l to l-1
    to enable pooling and unpooling without relying on external graph libs.
    """
    def __init__(self, levels: int = 3, radius: float = 1.0, device: str | torch.device = "cpu"):
        assert levels >= 0
        self.levels = levels
        self.radius = radius
        self.device = torch.device(device)
        self.vertices: List[torch.Tensor] = []  # list of (N_l,3)
        self.faces: List[torch.Tensor] = []     # list of (M_l,3) (indices)
        self.parent: List[Optional[torch.Tensor]] = []  # parent idx of each v at level l in level l-1
        self._build()

    @staticmethod
    def _base_icosahedron(device):
        t = (1.0 + math.sqrt(5.0)) / 2.0
        verts = torch.tensor([
            (-1,  t,  0), ( 1,  t,  0), (-1, -t,  0), ( 1, -t,  0),
            ( 0, -1,  t), ( 0,  1,  t), ( 0, -1, -t), ( 0,  1, -t),
            ( t,  0, -1), ( t,  0,  1), (-t,  0, -1), (-t,  0,  1)
        ], dtype=torch.float32, device=device)
        faces = torch.tensor([
            (0,11,5), (0,5,1), (0,1,7), (0,7,10), (0,10,11),
            (1,5,9), (5,11,4), (11,10,2), (10,7,6), (7,1,8),
            (3,9,4), (3,4,2), (3,2,6), (3,6,8), (3,8,9),
            (4,9,5), (2,4,11), (6,2,10), (8,6,7), (9,8,1)
        ], dtype=torch.long, device=device)
        verts = IcoSphere._normalize(verts)
        return verts, faces

    @staticmethod
    def _normalize(v):
        v = F.normalize(v, dim=-1)
        return v

    def _subdivide(self, verts, faces):
        """Loop subdivision: split each triangle, project midpoints to sphere."""
        # midpoint cache to avoid duplicates
        cache: Dict[Tuple[int,int], int] = {}
        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in cache:
                return cache[key]
            m = F.normalize((verts[i] + verts[j]) * 0.5, dim=0)
            verts.append(m)
            idx = len(verts) - 1
            cache[key] = idx
            return idx

        verts = [v for v in verts]  # list of tensors
        new_faces = []
        for tri in faces.tolist():
            i, j, k = tri
            a = midpoint(i, j)
            b = midpoint(j, k)
            c = midpoint(k, i)
            new_faces += [
                (i, a, c), (a, j, b), (c, b, k), (a, b, c)
            ]
        verts = torch.stack(verts, dim=0)
        faces = torch.tensor(new_faces, dtype=torch.long, device=verts.device)
        return verts, faces

    def _build(self):
        v0, f0 = self._base_icosahedron(self.device)
        self.vertices = [v0 * self.radius]
        self.faces = [f0]
        self.parent = [None]
        for l in range(1, self.levels + 1):
            v_prev = self.vertices[-1]
            f_prev = self.faces[-1]
            v, f = self._subdivide(v_prev, f_prev)
            self.vertices.append(v * self.radius)
            self.faces.append(f)
            # parent mapping via nearest neighbor on previous level
            with torch.no_grad():
                d = torch.cdist(v, v_prev)
                parent = d.argmin(dim=1)
            self.parent.append(parent)

    def neighbors_knn(self, level: int, k: int = 6) -> torch.Tensor:
        """Return undirected edge index (2,E) via kNN on the sphere."""
        pos = self.vertices[level]
        # k+1 because self is nearest; we will drop self edges
        d = torch.cdist(pos, pos)
        knn = d.topk(k=k+1, largest=False).indices[:, 1:]
        src = torch.arange(pos.size(0), device=pos.device).unsqueeze(1).expand_as(knn)
        edge_index = torch.stack([src.reshape(-1), knn.reshape(-1)], dim=0)
        # make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return edge_index


# -------------------------------------
# Graph container independent of PyG
# -------------------------------------
@dataclass
class SphereGraph:
    pos: torch.Tensor            # (N,3) cart positions on sphere
    x: torch.Tensor              # node features shaped to match irreps
    edge_index: torch.Tensor     # (2,E)
    level: int                   # icosphere level
    parent: Optional[torch.Tensor] = None  # mapping to parent level (N,)


# --------------------------------------------
# Utilities: irreps feature packing/unpacking
# --------------------------------------------

def make_initial_irreps() -> o3.Irreps:
    # 1 scalar (intensity) + 1 vector (xyz from center)
    return o3.Irreps("1x0e + 1x1o")


def pack_initial_features(intensity: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Pack intensity + position into a flat (N, C) matching Irreps '1x0e + 1x1o'."""
    assert intensity.ndim == 2 and intensity.size(1) == 1
    assert pos.ndim == 2 and pos.size(1) == 3
    return torch.cat([intensity, pos], dim=1)


# -------------------------------------------------
# Spherical harmonics + equivariant edge conv block
# -------------------------------------------------
class EquivariantEdgeConv(nn.Module):
    def __init__(self,
                 irreps_in: o3.Irreps,
                 irreps_hidden: o3.Irreps,
                 sh_lmax: int = 2,
                 radial_mlp_hidden: int = 64):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=sh_lmax)

        # this handles instructions internally
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False,
        )

        # Radial MLP to produce weights for TP
        self.radial_mlp = FullyConnectedNet([1, radial_mlp_hidden, self.tp.weight_numel], act=F.silu)

        # Gated nonlinearity setup
        self.scalar_irreps = o3.Irreps([ir for ir in self.irreps_out if ir.ir.l == 0])
        self.nonscalar_irreps = o3.Irreps([ir for ir in self.irreps_out if ir.ir.l != 0])

        self.lin_scalar = o3.Linear(self.irreps_out, self.scalar_irreps)
        self.lin_nonscalar = o3.Linear(self.irreps_out, self.nonscalar_irreps) if self.nonscalar_irreps.dim > 0 else None
        self.lin_gates = o3.Linear(self.irreps_out, self.scalar_irreps) if self.nonscalar_irreps.dim > 0 else None
        self.act_scalar = nn.SiLU()
        self.act_gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        edge_vec = pos[dst] - pos[src]
        edge_len = edge_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        edge_dir = edge_vec / edge_len

        # spherical harmonics on unit directions
        Y = o3.spherical_harmonics(list(range(self.irreps_sh.lmax + 1)), edge_dir, normalize=True)

        # radial weights
        w = self.radial_mlp(edge_len)

        # tensor product messages
        m = self.tp(x[src], Y, w)

        # aggregate at destinations
        out = torch.zeros(
            x.size(0),
            m.size(1),   # use the output feature dimension, not x’s
            device=x.device,
            dtype=x.dtype
        )
        out = out.index_add(0, dst, m)

        # gated activation
        if self.nonscalar_irreps.dim == 0:
            return self.act_scalar(self.lin_scalar(out))
        s = self.act_scalar(self.lin_scalar(out))
        ns = self.lin_nonscalar(out)
        g = self.act_gate(self.lin_gates(out))
        return s + (g * ns)

# ---------------------------------------
# Encoder/Decoder on spherical hierarchy
# ---------------------------------------
class EquivariantEncoder(nn.Module):
    def __init__(self, levels: int, width: int = 32, sh_lmax: int = 2):
        super().__init__()
        self.levels = levels
        self.ir_in = make_initial_irreps()  # 1x0e + 1x1o
        widths = [width * (2 ** l) for l in range(levels + 1)]
        # represent widths using only scalars and vectors to keep gating simple
        def make_ir(w):
            # approx split: half scalars, half vectors
            s = max(1, w // 2)
            v = max(1, w - s)
            return o3.Irreps(f"{s}x0e + {v}x1o")
        self.blocks = nn.ModuleList()
        self.proj_in = o3.Linear(self.ir_in, make_ir(width))
        ir_prev = make_ir(width)
        for l in range(levels):
            ir_next = make_ir(widths[l+1])
            self.blocks.append(EquivariantEdgeConv(ir_prev, ir_next, sh_lmax=sh_lmax))
            ir_prev = ir_next
        self.ir_out = ir_prev

    def forward(self, graphs: List[SphereGraph]) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        x = self.proj_in(graphs[0].x)
        for l in range(self.levels):
            g = graphs[l]
            x = self.blocks[l](x, g.edge_index, g.pos)
            # pool to next level using fixed parent map (mean over children)
            parent = graphs[l+1].parent
            if parent is None:
                feats.append(x)
                continue
            num_par = graphs[l+1].pos.size(0)
            pooled = torch.zeros(num_par, x.size(1), device=x.device, dtype=x.dtype)
            pooled = pooled.index_add(0, parent, x)
            # average by child counts
            counts = torch.zeros(num_par, device=x.device).index_add(0, parent, torch.ones(x.size(0), device=x.device))
            counts = counts.clamp(min=1).unsqueeze(1)
            x = pooled / counts
            feats.append(x)
        feats.append(x)
        return feats  # low→high level features (pooled)


class EquivariantDecoder(nn.Module):
    def __init__(self, levels: int, enc_irreps: List[o3.Irreps], width: int = 32, sh_lmax: int = 2):
        super().__init__()
        self.levels = levels
        # Build blocks from top (coarsest) back to level 0 (finest)
        self.blocks = nn.ModuleList()
        self.up_lin = nn.ModuleList()
        for l in reversed(range(levels)):
            # concatenate skip from encoder at this level with upsampled prev
            ir_in = enc_irreps[l] + enc_irreps[l+1]  # simple concat in irreps space
            ir_out = enc_irreps[l]
            self.blocks.append(EquivariantEdgeConv(ir_in, ir_out, sh_lmax=sh_lmax))
            self.up_lin.append(o3.Linear(enc_irreps[l+1], enc_irreps[l]))

    def forward(self, graphs: List[SphereGraph], enc_feats: List[torch.Tensor]) -> torch.Tensor:
        # enc_feats: list per level (0..L) where L is top; graphs[0] is finest
        x_up = enc_feats[-1]
        for i, l in enumerate(reversed(range(self.levels))):
            # unpool: tile parent feature to its children via parent map
            parent = graphs[l+1].parent
            child_n = graphs[l].pos.size(0)
            x_child = x_up[parent]  # broadcast down
            x_child = self.up_lin[i](x_child)
            # concat skip feature from encoder at level l (recompute local conv)
            x_cat = torch.cat([enc_feats[l], x_child], dim=-1)
            x_up = self.blocks[i](x_cat, graphs[l].edge_index, graphs[l].pos)
        return x_up  # finest level latent


# ---------------------------
# SIREN with FiLM conditioning
# ---------------------------
class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, w0=30.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * self.lin(x))


class FiLM(nn.Module):
    def __init__(self, latent_dim, feat_dim):
        super().__init__()
        self.to_gamma = nn.Linear(latent_dim, feat_dim)
        self.to_beta = nn.Linear(latent_dim, feat_dim)

    def forward(self, h, z):
        gamma = self.to_gamma(z)
        beta = self.to_beta(z)
        return gamma * h + beta


class SirenDecoder(nn.Module):
    def __init__(self, latent_irreps: o3.Irreps, hidden: int = 128, layers: int = 4):
        super().__init__()
        # extract only scalars (l == 0)
        self.latent_ir = o3.Irreps([ir for ir in latent_irreps if ir.ir.l == 0])
        self.latent_dim = self.latent_ir.dim
        self.lat_proj = o3.Linear(latent_irreps, self.latent_ir)
        self.inp = nn.Linear(3, hidden)
        self.siren = nn.ModuleList([SirenLayer(hidden, hidden) for _ in range(layers)])
        self.film = nn.ModuleList([FiLM(self.latent_dim, hidden) for _ in range(layers)])
        self.out = nn.Linear(hidden, 3)  # 3D displacement

    def forward(self, coords: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        # coords: (M,3), latent: (N, C) at sphere nodes — we aggregate to a single
        # global latent by average over nodes (you may switch to attention or grid sample)
        z = self.lat_proj(latent)
        z = z.mean(dim=0, keepdim=True)
        h = self.inp(coords)
        for layer, mod in zip(self.siren, self.film):
            h = layer(h)
            h = mod(h, z)
        u = self.out(h)
        return u


# ---------------------------------
# Local NCC (LNCC) and regularizers
# ---------------------------------
class LNCC(nn.Module):
    def __init__(self, radius: float = 0.02, eps: float = 1e-6):
        super().__init__()
        self.radius = radius
        self.eps = eps

    def forward(self, pf_pos, pf_int, pm_warp_pos, pm_int):
        """Local normalized cross-correlation over spherical neighborhoods.
        pf/pm_int: (N,1) intensities sampled at pf_pos / pm_warp_pos (match cardinality)
        """
        # kNN neighborhoods by radius on pf_pos
        d = torch.cdist(pf_pos, pf_pos)
        mask = (d <= self.radius).float()
        # local means
        mu_f = (mask @ pf_int) / (mask.sum(dim=1, keepdim=True).clamp(min=1.))
        mu_m = (mask @ pm_int) / (mask.sum(dim=1, keepdim=True).clamp(min=1.))
        # numerators / denominators
        num = (mask @ (pf_int * pm_int)) - (mask @ pf_int) * (mask @ pm_int) / mask.sum(dim=1, keepdim=True).clamp(min=1.)
        var_f = (mask @ (pf_int ** 2)) - (mask @ pf_int) ** 2 / mask.sum(dim=1, keepdim=True).clamp(min=1.)
        var_m = (mask @ (pm_int ** 2)) - (mask @ pm_int) ** 2 / mask.sum(dim=1, keepdim=True).clamp(min=1.)
        lncc = num / (torch.sqrt(var_f * var_m) + self.eps)
        # maximize NCC -> minimize negative
        return -lncc.mean()


def smoothness_loss(coords, disp):
    """E[||∇u||^2] via autograd Jacobian-vector products.
    coords: (M,3), disp: (M,3) as u(x)
    """
    grad = []
    for i in range(3):
        grad_i = torch.autograd.grad(disp[:, i].sum(), coords, create_graph=True)[0]
        grad.append(grad_i)
    J = torch.stack(grad, dim=-1)  # (M,3,3) where J[a,b]=∂u_b/∂x_a
    return (J.pow(2).sum(dim=[1,2]).mean())


def jacobian_folding_loss(coords, disp):
    grad = []
    for i in range(3):
        grad_i = torch.autograd.grad(disp[:, i].sum(), coords, create_graph=True)[0]
        grad.append(grad_i)
    J = torch.stack(grad, dim=-1)  # (M,3,3)
    Jphi = torch.eye(3, device=coords.device).expand(coords.size(0), 3, 3) + J
    det = torch.det(Jphi)
    return F.relu(-det).mean()


# -------------------------
# Spherical pre-processing
# -------------------------
class SphericalPreprocessor(nn.Module):
    def __init__(self, icosphere: IcoSphere, knn_k: int = 3):
        super().__init__()
        self.ico = icosphere
        self.knn_k = knn_k

    def build_graphs(self, points_xyz: torch.Tensor, intensity: torch.Tensor, neighbor_k: int = 6) -> List[SphereGraph]:
        # center to COM
        com = points_xyz.mean(dim=0, keepdim=True)
        xyz0 = points_xyz - com
        # sample intensity to sphere vertices at each level by nearest neighbor on the *ray*
        graphs: List[SphereGraph] = []
        for l in range(self.ico.levels + 1):
            pos = self.ico.vertices[l].to(points_xyz.device)
            # project sphere vertex to ray and find nearest point along ray direction
            # use cosine similarity to pick points close to direction
            dir = F.normalize(pos, dim=-1)  # (N,3)
            dots = torch.matmul(F.normalize(xyz0, dim=-1), dir.t())  # (P,N)
            idx = dots.topk(k=self.knn_k, dim=0).indices  # (k,N)
            # average intensities of top-k along each ray
            inten = intensity[idx].mean(dim=0)  # (N,1)
            feats = pack_initial_features(inten, pos)
            edge_index = self.ico.neighbors_knn(l, k=neighbor_k).to(points_xyz.device)
            parent = self.ico.parent[l]
            graphs.append(SphereGraph(pos=pos, x=feats, edge_index=edge_index, level=l, parent=parent))
        return graphs


# ---------------
# End-to-end model
# ---------------
class ScoreNet(nn.Module):
    def __init__(self, levels: int = 3, width: int = 32, sh_lmax: int = 2):
        super().__init__()
        self.ico = IcoSphere(levels=levels)
        self.prep = SphericalPreprocessor(self.ico)
        self.encoder = EquivariantEncoder(levels=levels, width=width, sh_lmax=sh_lmax)
        # keep track of irreps at each level for decoder
        enc_irreps = []
        # simulate forward to capture irreps progression
        w0 = width
        widths = [w0 * (2 ** l) for l in range(levels + 1)]
        def make_ir(w):
            s = max(1, w // 2)
            v = max(1, w - s)
            return o3.Irreps(f"{s}x0e + {v}x1o")
        enc_irreps = [make_ir(w) for w in widths]
        self.decoder = EquivariantDecoder(levels=levels, enc_irreps=enc_irreps, width=width, sh_lmax=sh_lmax)
        self.siren = SirenDecoder(latent_irreps=enc_irreps[0], hidden=128, layers=4)
        self.lncc = LNCC(radius=0.05)

    def forward(self,
                pf_points: torch.Tensor, pf_int: torch.Tensor,
                pm_points: torch.Tensor, pm_int: torch.Tensor,
                neighbor_k: int = 6):
        # Build spherical graphs (same ico for both)
        gF = self.prep.build_graphs(pf_points, pf_int, neighbor_k)
        gM = self.prep.build_graphs(pm_points, pm_int, neighbor_k)
        # Siamese encoder
        fF = self.encoder(gF)
        fM = self.encoder(gM)
        # Correlate by concatenation at each level (simple but effective)
        fCat = [torch.cat([a, b], dim=-1) for a, b in zip(fF, fM)]
        # Decode to finest latent
        z = self.decoder(gF, fCat)
        # SIREN: query displacements for moving points
        pm_points = pm_points.requires_grad_(True)
        disp = self.siren(pm_points, z)
        pm_warp = pm_points + disp
        # sample moving intensities remain associative; caller may resample as needed
        # Loss terms
        l_sim = self.lncc(gF[0].pos, gF[0].x[:, :1], gF[0].pos, gM[0].x[:, :1])  # proxy LNCC at sphere nodes
        l_smooth = smoothness_loss(pm_points, disp)
        l_jac = jacobian_folding_loss(pm_points, disp)
        return {
            "pm_warp": pm_warp,
            "disp": disp,
            "latent": z,
            "losses": {"sim": l_sim, "smooth": l_smooth, "jac": l_jac}
        }


# -----------------
# Minimal train loop
# -----------------
class ScoreNetTrainer:
    def __init__(self, model: ScoreNet, lr: float = 1e-3, lam_smooth: float = 1.0, lam_jac: float = 1.0):
        self.model = model
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)
        self.lam_smooth = lam_smooth
        self.lam_jac = lam_jac

    def step(self, pf_pts, pf_int, pm_pts, pm_int):
        self.opt.zero_grad()
        out = self.model(pf_pts, pf_int, pm_pts, pm_int)
        l = out["losses"]["sim"] + self.lam_smooth * out["losses"]["smooth"] + self.lam_jac * out["losses"]["jac"]
        l.backward()
        self.opt.step()
        return {k: v.detach().item() for k, v in out["losses"].items()}


# -----------------
# Quick smoke test
# -----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fake brain point clouds (replace with actual cortical surface samples)
    N = 2048
    pf_pts = F.normalize(torch.randn(N, 3, device=device), dim=-1) * 100.0
    pm_pts = F.normalize(torch.randn(N, 3, device=device), dim=-1) * 100.0
    pf_int = torch.randn(N, 1, device=device)
    pm_int = torch.randn(N, 1, device=device)

    model = ScoreNet(levels=3, width=16, sh_lmax=2).to(device)
    trainer = ScoreNetTrainer(model, lr=1e-3, lam_smooth=0.1, lam_jac=0.1)

    stats = trainer.step(pf_pts, pf_int, pm_pts, pm_int)
    print({k: round(v, 4) for k, v in stats.items()})
