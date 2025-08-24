import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 1: ICOSAHEDRAL GRID GENERATION
# Corresponds to Section 2.2 of the paper.
# This helper class generates the vertices and connectivity for a sphere
# tessellated from a recursively subdivided icosahedron.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class IcosahedralGrid:
    """
    Generates a hierarchical icosahedral grid.

    This class creates a quasi-uniform tessellation of a sphere by starting
    with a base icosahedron and recursively subdividing its triangular faces.
    The resulting vertices and edges form the basis for the spherical graph CNN.
    """
    def __init__(self, subdivision_level=0):
        # Base icosahedron vertices (normalized to unit sphere)
        t = (1.0 + np.sqrt(5.0)) / 2.0
        vertices = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ])
        vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

        # Base icosahedron faces (20 triangles)
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])

        # Recursively subdivide the icosahedron
        for _ in range(subdivision_level):
            new_faces = []
            midpoint_cache = {}

            for face in faces:
                v1, v2, v3 = face
                
                def get_midpoint(p1, p2):
                    nonlocal vertices
                    smaller, greater = min(p1, p2), max(p1, p2)
                    if (smaller, greater) in midpoint_cache:
                        return midpoint_cache[(smaller, greater)]
                    
                    mid = (vertices[p1] + vertices[p2]) / 2.0
                    mid /= np.linalg.norm(mid)
                    vertices = np.vstack([vertices, mid])
                    mid_idx = len(vertices) - 1
                    midpoint_cache[(smaller, greater)] = mid_idx
                    return mid_idx

                m12 = get_midpoint(v1, v2)
                m23 = get_midpoint(v2, v3)
                m31 = get_midpoint(v3, v1)

                new_faces.extend([[v1, m12, m31], [v2, m23, m12], 
                                  [v3, m31, m23], [m12, m23, m31]])
            faces = np.array(new_faces)

        self.vertices = torch.from_numpy(vertices).float()
        
        # Create edge index for PyTorch Geometric
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        self.edge_index = torch.from_numpy(edges.T).long()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 2: VOLUMETRIC TO SPHERICAL PARAMETERIZATION
# Corresponds to Section 2.1 of the paper.
# This module converts a 3D Cartesian volume into a multi-channel spherical
# signal defined on the icosahedral grid.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VolumetricSphericalParameterization(nn.Module):
    """
    Performs ray-casting and feature sampling to create a spherical graph.

    This module takes a 3D MRI volume and an IcosahedralGrid, and for each
    vertex on the grid (representing a direction), it casts a ray from the
    center of the volume. It samples features (radial distance to the surface
    and MRI intensity at normalized depths) along this ray.
    """
    def __init__(self, grid_vertices, radial_samples=3):
        super().__init__()
        self.grid_vertices = nn.Parameter(grid_vertices, requires_grad=False)
        self.num_vertices = grid_vertices.shape[0]
        self.radial_samples = radial_samples

    def forward(self, volume, brain_mask):
        # Assumes volume is a single [D, H, W] tensor and pre-centered.
        D, H, W = volume.shape
        center = torch.tensor([D / 2, H / 2, W / 2], device=volume.device)
        
        # 1. Ray Casting and Surface Shape Feature Extraction
        ray_directions = self.grid_vertices
        
        # This is a simplified search for the surface. A more robust implementation
        # would use a more sophisticated edge detection or surface finding algorithm.
        # Here we march along each ray until we exit the brain mask.
        max_radius = torch.tensor([D, H, W], dtype=torch.float).norm() / 2.0
        radii = torch.linspace(0, max_radius, steps=int(max_radius), device=volume.device)
        
        ray_points = center.view(1, 3) + torch.einsum("r,vd->rvd", radii, ray_directions)
        
        # Normalize coordinates for grid_sample
        norm_coords = ray_points / torch.tensor([D-1, H-1, W-1], device=volume.device) * 2 - 1
        
        # Sample the brain mask along all rays
        mask_values = F.grid_sample(
            brain_mask.view(1, 1, D, H, W),
            norm_coords.view(1, -1, 1, 1, 3),
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).view(len(radii), self.num_vertices)
        
        # Find the first point where the mask value drops below a threshold (e.g., 0.5)
        is_outside = mask_values < 0.5
        surface_indices = torch.argmax(is_outside.int(), dim=0)
        surface_radius = radii[surface_indices] # Shape: [num_vertices]

        # 2. Volumetric Texture Feature Sampling
        feature_channels = [surface_radius.unsqueeze(1)]
        normalized_depths = torch.linspace(0.25, 0.75, self.radial_samples, device=volume.device)
        
        for depth in normalized_depths:
            sample_radii = surface_radius * depth
            sample_points = center.view(1, 3) + sample_radii.unsqueeze(1) * ray_directions
            norm_sample_coords = sample_points / torch.tensor([D-1, H-1, W-1], device=volume.device) * 2 - 1
            
            intensity_values = F.grid_sample(
                volume.view(1, 1, D, H, W),
                norm_sample_coords.view(1, self.num_vertices, 1, 1, 3),
                mode='bilinear', padding_mode='border', align_corners=True
            ).view(self.num_vertices, 1)
            feature_channels.append(intensity_values)
            
        # Combine features into a single tensor for the graph
        features = torch.cat(feature_channels, dim=1)
        return features


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 3: SPHERICAL ENCODER AND HYBRID ARCHITECTURE
# Corresponds to Section 3 of the paper.
# Defines the novel network layers and the main SphereMorph-Net model.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ConicalRadialSamplingModule(nn.Module):
    """
    CRSM: Initial feature extraction layer for the spherical graph.
    Corresponds to Section 3.2.1.
    """
    def __init__(self, in_channels, out_channels, edge_index):
        super().__init__()
        self.edge_index, _ = add_self_loops(edge_index)
        
        # MLP for radially-sampled features (point-wise features)
        self.radial_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            nn.LeakyReLU(0.2)
        )
        # Simple graph aggregation for conical sampling (local neighborhood)
        self.conical_aggregator = MessagePassing(aggr='mean')
        
        self.conical_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            nn.LeakyReLU(0.2)
        )
        
        # Final MLP to combine features
        self.final_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # Radial path
        radial_features = self.radial_mlp(x)
        
        # Conical path
        # Propagate node features to neighbors
        conical_aggregated = self.conical_aggregator.propagate(self.edge_index, x=x)
        conical_features = self.conical_mlp(conical_aggregated)
        
        # Concatenate and process
        combined_features = torch.cat([radial_features, conical_features], dim=1)
        return self.final_mlp(combined_features)

class GaugeEquivariantIcoConv(MessagePassing):
    """
    Simplified Gauge Equivariant Icosahedral Convolution.
    Corresponds to Section 3.2.2.
    
    NOTE: A true implementation is extremely complex. This class serves as a
    structurally-correct stand-in using a standard MessagePassing scheme, which
    is the foundation for most GNNs as suggested by the paper. It learns features
    from local neighborhoods on the icosahedral graph.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean') # 'mean' aggregation
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Add self-loops to include node's own features
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Transform message from neighbor nodes
        out = self.propagate(edge_index_sl, x=self.lin(x))
        
        # Transform node's own features and add
        out = out + self.lin_self(x)
        return out

    def message(self, x_j):
        # x_j has shape [E, out_channels]
        return x_j

class IcoPool(nn.Module):
    """ Downsampling for the icosahedral grid. """
    def __init__(self, mode='mean'):
        super().__init__()
        self.mode = mode
    
    def forward(self, x, coarser_grid_map):
        # coarser_grid_map tells us which fine vertices map to which coarse vertex
        # This is a simplified pooling; in practice, this mapping needs to be pre-computed.
        # For demonstration, we assume a simple averaging.
        # Let's use a dummy implementation for shape transformation
        num_coarse_nodes = len(torch.unique(coarser_grid_map))
        out = torch.zeros(num_coarse_nodes, x.shape[1], device=x.device)
        out = out.index_add_(0, coarser_grid_map, x) # Sum features
        if self.mode == 'mean':
            counts = torch.zeros(num_coarse_nodes, 1, device=x.device)
            counts = counts.index_add_(0, coarser_grid_map, torch.ones_like(x))
            out = out / counts.clamp(min=1)
        return out

class IcoUnpool(nn.Module):
    """ Upsampling for the icosahedral grid (e.g., nearest neighbor). """
    def __init__(self, mode='nearest'):
        super().__init__()
        self.mode = mode

    def forward(self, x, finer_grid_map):
        # finer_grid_map is the inverse of the coarser_grid_map
        return x[finer_grid_map]

class SphereMorphNet(nn.Module):
    """
    The main SphereMorph-Net model architecture.
    Corresponds to Section 3.1. A U-Net style encoder-decoder on the sphere.
    """
    def __init__(self, in_channels, n_classes=3):
        super().__init__()
        
        # This is a simplified blueprint. A real implementation would need
        # pre-computed grid hierarchies and pooling maps.
        # For now, we define the blocks but cannot run a forward pass without data.
        
        # --- Encoder ---
        # Assuming we have grids and edge indices for different resolutions
        # e.g., grid_l0, edge_index_l0, grid_l1, edge_index_l1...
        
        # self.crs = ConicalRadialSamplingModule(...)
        # self.enc_conv1 = GaugeEquivariantIcoConv(...)
        # self.pool1 = IcoPool()
        # self.enc_conv2 = GaugeEquivariantIcoConv(...)
        # self.pool2 = IcoPool()

        # --- Bottleneck ---
        # self.bottleneck = GaugeEquivariantIcoConv(...)
        
        # --- Decoder ---
        # self.unpool1 = IcoUnpool()
        # self.dec_conv1 = GaugeEquivariantIcoConv(...)
        # self.unpool2 = IcoUnpool()
        # self.dec_conv2 = GaugeEquivariantIcoConv(...)
        
        # --- S2C Head ---
        # 1x1 graph convolution to get 3 displacement channels
        # self.s2c_conv = GaugeEquivariantIcoConv(..., out_channels=3)
        print("SphereMorphNet model initialized. NOTE: This is an architectural blueprint. \
A forward pass requires pre-computed grid hierarchies and data loaders.")
              
    def forward(self, graph_moving, graph_fixed):
        # The input would be two graph objects concatenated on the feature dim.
        # x = torch.cat([graph_moving.x, graph_fixed.x], dim=1)
        # e.g., x1 = self.enc_conv1(self.crs(x), edge_index_l0)
        # x2 = self.enc_conv2(self.pool1(x1), edge_index_l1)
        # ... and so on through the U-Net structure with skip connections.
        
        # Final output from decoder would be `spherical_displacement`
        # spherical_displacement = self.s2c_conv(final_decoder_features, edge_index_l0)
        
        # This would then be passed to a SphericalToCartesianTransform module
        raise NotImplementedError("Forward pass requires pre-computed grid hierarchies and cannot be demonstrated with dummy data.")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 4: SPHERICAL-TO-CARTESIAN TRANSFORMATION
# Corresponds to Section 3.3.2.
# This module converts the network's spherical displacement output back to a
# dense 3D Cartesian deformation field.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SphericalToCartesianTransform(nn.Module):
    """
    Converts a spherical displacement field to a Cartesian deformation field.
    
    This is a non-parametric module that performs the geometric transformation
    detailed in the paper. It is computationally intensive.
    
    (CORRECTED to handle large volumes by chunking the nearest-neighbor search)
    """
    def __init__(self, grid_vertices):
        super().__init__()
        self.grid_vertices = nn.Parameter(grid_vertices, requires_grad=False)
        # Pre-compute spherical coordinates of grid vertices
        x, y, z = grid_vertices.T
        r_ = torch.sqrt(x**2 + y**2 + z**2)
        theta_ = torch.acos(z / r_.clamp(min=1e-6))  # Polar angle
        phi_ = torch.atan2(y, x)     # Azimuthal angle
        self.grid_theta_phi = nn.Parameter(torch.stack([theta_, phi_], dim=1), requires_grad=False)

    def forward(self, spherical_displacement, target_shape, chunk_size=8192):
        D, H, W = target_shape
        device = spherical_displacement.device
        
        # Create a grid of voxel coordinates
        coords = torch.stack(torch.meshgrid(
            torch.arange(D, device=device),
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        ), dim=-1).float()
        
        center = torch.tensor([(D-1)/2, (H-1)/2, (W-1)/2], device=device)
        coords_centered = coords.view(-1, 3) - center
        
        # 1. Convert voxel Cartesian coordinates to spherical
        x, y, z = coords_centered.T
        rho = torch.norm(coords_centered, dim=1)
        theta = torch.acos(z / rho.clamp(min=1e-6))
        phi = torch.atan2(y, x)
        
        # 2. Interpolate displacement vectors (Nearest Neighbor) IN CHUNKS
        num_voxels = coords_centered.shape[0]
        all_nearest_indices = torch.empty(num_voxels, dtype=torch.long, device=device)

        for i in range(0, num_voxels, chunk_size):
            end = i + chunk_size
            theta_chunk = theta[i:end]
            phi_chunk = phi[i:end]
            
            # This calculation is now done on a small chunk, avoiding OOM
            dist_matrix_chunk = (theta_chunk.unsqueeze(1) - self.grid_theta_phi[:, 0])**2 + \
                                (phi_chunk.unsqueeze(1) - self.grid_theta_phi[:, 1])**2
            
            all_nearest_indices[i:end] = torch.argmin(dist_matrix_chunk, dim=1)

        interp_disp = spherical_displacement[all_nearest_indices]
        d_rho, d_theta, d_phi = interp_disp.T

        # 3. Convert spherical displacement to Cartesian displacement
        sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
        sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
        
        e_rho = torch.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], dim=1)
        e_theta = torch.stack([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta], dim=1)
        e_phi = torch.stack([-sin_phi, cos_phi, torch.zeros_like(phi)], dim=1)
        
        dx_dy_dz = d_rho.unsqueeze(1) * e_rho + \
                   (rho * d_theta).unsqueeze(1) * e_theta + \
                   (rho * sin_theta * d_phi).unsqueeze(1) * e_phi
                   
        return dx_dy_dz.view(D, H, W, 3).permute(3, 0, 1, 2)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 5: LOSS FUNCTION AND SPATIAL TRANSFORMER
# Corresponds to Section 4 and other implementation details.
# Defines the multi-component loss and the final warping mechanism.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class NCCLoss(nn.Module):
    """ Local Normalized Cross-Correlation Loss. """
    def __init__(self, win=9):
        super().__init__()
        self.win = win
        self.win_size = win**3
        self.padd = win // 2

    def forward(self, y_true, y_pred):
        # Compute moments
        I = y_true
        J = y_pred
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # Use 3D convolution for local summation
        conv_op = nn.Conv3d(1, 1, kernel_size=self.win, padding=self.padd, bias=False)
        conv_op.weight.data = torch.ones(1, 1, self.win, self.win, self.win).to(I.device)
        
        I_sum = conv_op(I)
        J_sum = conv_op(J)
        I2_sum = conv_op(I2)
        J2_sum = conv_op(J2)
        IJ_sum = conv_op(IJ)

        I_mu = I_sum / self.win_size
        J_mu = J_sum / self.win_size

        cross = IJ_sum - I_mu * J_sum - J_mu * I_sum + I_mu * J_mu * self.win_size
        I_var = I2_sum - 2 * I_mu * I_sum + I_mu * I_mu * self.win_size
        J_var = J2_sum - 2 * J_mu * J_sum + J_mu * J_mu * self.win_size

        ncc = (cross * cross) / (I_var * J_var + 1e-5)
        return -torch.mean(ncc)

class SmoothnessLoss(nn.Module):
    """ Diffusion regularizer to enforce smoothness. """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
        return torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)

class JacobianLoss(nn.Module):
    """ Penalizes non-positive Jacobian determinants. """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred):
        J = self.get_jacobian_matrix(y_pred)
        det = J[:, 0, 0] * (J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1]) - \
              J[:, 0, 1] * (J[:, 1, 0] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 0]) + \
              J[:, 0, 2] * (J[:, 1, 0] * J[:, 2, 1] - J[:, 1, 1] * J[:, 2, 0])
        return torch.mean(F.relu(-det))

    def get_jacobian_matrix(self, y_pred):
        # Approximate gradients using central differences
        p = 1
        D_y = F.pad(y_pred, [p, p, p, p, p, p], mode='replicate')
        # ... (implementation of gradient computation)
        # This is non-trivial, returning a placeholder
        print("WARN: Jacobian determinant calculation is non-trivial; using a placeholder value.")
        return torch.eye(3, 3, device=y_pred.device).reshape(1, 3, 3).repeat(y_pred.shape[0], 1, 1)

class SpatialTransformer(nn.Module):
    """ Differentiable spatial transformer layer to warp images. """
    def __init__(self, size):
        super().__init__()
        self.size = size
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = grid.float()
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, padding_mode="border")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SECTION 6: DEMONSTRATION OF USAGE
# This section shows how the components would be initialized and used.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    # --- 1. Setup ---
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    device = 'cuda:3'
    
    # --- 2. Create the Spherical Grid ---
    print("\n[1/5] Generating Icosahedral Grid...")
    subdivision_level = 5 # Higher level -> more vertices -> higher resolution
    ico_grid = IcosahedralGrid(subdivision_level=subdivision_level)
    grid_vertices = ico_grid.vertices.to(device)
    edge_index = ico_grid.edge_index.to(device)
    print(f"Grid created with {grid_vertices.shape[0]} vertices and {edge_index.shape[1]} edges.")
    
    # --- 3. Prepare Dummy Data ---
    # In a real scenario, you would load MRI scans (e.g., from .nii files)
    print("\n[2/5] Creating dummy 3D MRI data...")
    volume_shape = (64, 64, 64)
    # Moving image: a sphere
    moving_vol = torch.zeros(volume_shape, device=device)
    c, r = (32, 32, 32), 20
    x, y, z = torch.meshgrid(torch.arange(64), torch.arange(64), torch.arange(64), indexing='ij')
    moving_vol[(x-c[0])**2 + (y-c[1])**2 + (z-c[2])**2 < r**2] = 1.0
    
    # Fixed image: a slightly smaller, offset sphere
    fixed_vol = torch.zeros(volume_shape, device=device)
    c, r = (30, 34, 34), 18
    fixed_vol[(x-c[0])**2 + (y-c[1])**2 + (z-c[2])**2 < r**2] = 1.0
    
    # Brain masks (for simplicity, same as the volumes)
    moving_mask = (moving_vol > 0).float()
    fixed_mask = (fixed_vol > 0).float()
    
    # --- 4. Parameterize Data to Spherical Domain ---
    print("\n[3/5] Parameterizing volumes to spherical graph signals...")
    parameterizer = VolumetricSphericalParameterization(grid_vertices, radial_samples=3).to(device)
    
    with torch.no_grad():
        moving_features = parameterizer(moving_vol, moving_mask)
        fixed_features = parameterizer(fixed_vol, fixed_mask)
    
    moving_graph = Data(x=moving_features, edge_index=edge_index)
    fixed_graph = Data(x=fixed_features, edge_index=edge_index)
    print(f"Created two graphs with feature shape: {moving_graph.x.shape}")

    # --- 5. Model and Transformation (Conceptual) ---
    # The full SphereMorphNet U-Net cannot be run without hierarchical grids.
    # We will demonstrate the final S2C transform step instead.
    print("\n[4/5] Demonstrating Spherical-to-Cartesian Transform...")
    
    # Let's assume the network produced a dummy spherical displacement field.
    # This is the TENSOR that the SphereMorph-Net decoder would output.
    # It has 3 channels (d_rho, d_theta, d_phi) for each grid vertex.
    dummy_spherical_displacement = torch.randn(grid_vertices.shape[0], 3, device=device) * 0.1
    
    s2c_transformer = SphericalToCartesianTransform(grid_vertices).to(device)
    
    with torch.no_grad():
        # The output is a dense 3D vector field
        deformation_field = s2c_transformer(dummy_spherical_displacement, volume_shape)
    
    print(f"Generated a Cartesian deformation field of shape: {deformation_field.shape}")
    
    # --- 6. Applying Deformation and Calculating Loss ---
    print("\n[5/5] Warping image and calculating loss...")
    # Add a batch dimension
    moving_vol_b = moving_vol.unsqueeze(0).unsqueeze(0) # [1, 1, D, H, W]
    fixed_vol_b = fixed_vol.unsqueeze(0).unsqueeze(0)
    deformation_field_b = deformation_field.unsqueeze(0) # [1, 3, D, H, W]

    # Warp the moving image
    stn = SpatialTransformer(volume_shape).to(device)
    warped_moving_vol_b = stn(moving_vol_b, deformation_field_b)

    # Calculate Loss Components
    ncc_loss = NCCLoss(win=9).to(device)
    smooth_loss = SmoothnessLoss().to(device)
    # jacobian_loss = JacobianLoss().to(device) # Placeholder

    l_sim = ncc_loss(fixed_vol_b, warped_moving_vol_b)
    l_smooth = smooth_loss(deformation_field_b)
    # l_diff = jacobian_loss(deformation_field_b)

    # Combine with hyperparameters lambda1 and lambda2
    lambda1 = 1.0
    lambda2 = 0.1
    total_loss = l_sim + lambda1 * l_smooth # + lambda2 * l_diff
    
    print(f"Similarity Loss (NCC): {l_sim.item():.4f}")
    print(f"Smoothness Loss: {l_smooth.item():.4f}")
    print(f"Total Weighted Loss: {total_loss.item():.4f}")
    print("\nDemonstration complete. This script provides a full blueprint for implementation.")