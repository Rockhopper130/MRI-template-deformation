import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os

class SegDataset(Dataset):
    """
    Dataset for 3D point cloud registration.
    Converts voxel masks to point clouds by sampling a fixed number of points.
    """

    @staticmethod
    def _create_coordinate_grids(size: tuple, device: torch.device = 'cpu') -> torch.Tensor:
        """
        Creates a 6-channel tensor of coordinates (x, y, z, r, theta, phi).
        """
        D, H, W = size
        
        d_coords, h_coords, w_coords = torch.meshgrid(
            torch.arange(D, device=device),
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        # --- Cartesian coordinates (normalized to [-1, 1]) ---
        z = (d_coords.float() / (D - 1)) * 2.0 - 1.0
        y = (h_coords.float() / (H - 1)) * 2.0 - 1.0
        x = (w_coords.float() / (W - 1)) * 2.0 - 1.0

        # --- Spherical coordinates (calculated from centered grid) ---
        zc = d_coords.float() - (D - 1) / 2.0
        yc = h_coords.float() - (H - 1) / 2.0
        xc = w_coords.float() - (W - 1) / 2.0
        
        r = torch.sqrt(xc**2 + yc**2 + zc**2)
        r_prime = r.clone()
        r_prime[r_prime == 0] = 1e-9 # Avoid division by zero
        theta = torch.acos(zc / r_prime)
        phi = torch.atan2(yc, xc)

        return torch.stack([x, y, z, r, theta, phi], dim=0)

    def __init__(self, data_list_file: str, template_path: str, target_size=(128, 128, 128), num_points=8192):
        print(f"Loading subject paths from: {data_list_file}")
        with open(data_list_file, 'r') as file:
            self.subject_paths = file.read().splitlines()
        print(f"Loaded {len(self.subject_paths)} subject paths.")
        
        self.target_size = target_size
        self.num_points = num_points
        print(f"Target size: {self.target_size}, Points per cloud: {self.num_points}")

        # Pre-calculate the dense coordinate grid for lookups
        print("Pre-calculating coordinate grids...")
        self.coord_grids = self._create_coordinate_grids(self.target_size)
        print(f"Coordinate grid created with shape: {self.coord_grids.shape}")

        # --- Process Moving Template into a Point Cloud ---
        print(f"Loading and processing template from: {template_path}")
        moving_template_onehot = np.load(template_path, mmap_mode='r')
        label_map = np.argmax(moving_template_onehot, axis=-1)
        binary_mask_np = (label_map > 0).astype(np.float32)
        
        # We don't need to interpolate, we'll work with original and then resize if needed
        # For simplicity here, assuming input is already near target size
        binary_mask_torch = torch.from_numpy(binary_mask_np).unsqueeze(0).unsqueeze(0)
        binary_mask_resized = F.interpolate(
            binary_mask_torch, size=self.target_size, mode='nearest'
        ).squeeze()
        
        self.moving_pc = self._mask_to_pointcloud(binary_mask_resized)
        print(f"Created moving template point cloud with shape: {self.moving_pc.shape}")

    def _mask_to_pointcloud(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Converts a binary mask tensor to a fixed-size point cloud.
        """
        # Get coordinates of all foreground points
        foreground_coords = torch.nonzero(mask, as_tuple=False) # (N_foreground, 3) [z, y, x]
        num_foreground = foreground_coords.shape[0]

        if num_foreground == 0:
            # Handle empty masks by returning a zero point cloud
            return torch.zeros(self.num_points, 6)

        # Sample with replacement if not enough points, or randomly select if too many
        sample_indices = torch.randint(0, num_foreground, (self.num_points,))
        sampled_coords = foreground_coords[sample_indices] # (num_points, 3)

        # Look up the features from our pre-calculated grid
        d, h, w = sampled_coords[:, 0], sampled_coords[:, 1], sampled_coords[:, 2]
        point_features = self.coord_grids[:, d, h, w] # (6, num_points)
        
        return point_features.T # (num_points, 6)

    def __len__(self):
        return len(self.subject_paths)

    def __getitem__(self, idx):
        # --- Process Fixed Map ---
        fixed_map_onehot = np.load(self.subject_paths[idx], mmap_mode='r')
        label_map = np.argmax(fixed_map_onehot, axis=-1)
        binary_mask_np = (label_map > 0).astype(np.float32)
        
        binary_mask_torch = torch.from_numpy(binary_mask_np).unsqueeze(0).unsqueeze(0)
        binary_mask_resized = F.interpolate(
            binary_mask_torch, size=self.target_size, mode='nearest'
        ).squeeze()
        
        fixed_pc = self._mask_to_pointcloud(binary_mask_resized)
        
        return self.moving_pc, fixed_pc

# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# import numpy as np
# import os

# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# import numpy as np
# import os

# class SegDataset(Dataset):
#     """
#     Dataset for 3D image registration.
#     This version is OPTIMIZED to pre-calculate the coordinate grid once.
#     """

#     @staticmethod
#     def _create_spherical_coordinates(size: tuple, device: torch.device = 'cpu') -> torch.Tensor:
#         """
#         Creates a 3-channel tensor of spherical coordinates (r, theta, phi) for a given size.
#         This is a helper function to be called only once.

#         Args:
#             size (tuple): The size of the volume (D, H, W).
#             device (torch.device): The device to create the tensor on.

#         Returns:
#             A tensor of shape (3, D, H, W) containing the coordinate grid.
#         """
#         D, H, W = size
        
#         # Create a grid of coordinates for the entire volume
#         d_coords, h_coords, w_coords = torch.meshgrid(
#             torch.arange(D, device=device),
#             torch.arange(H, device=device),
#             torch.arange(W, device=device),
#             indexing='ij'
#         )

#         # Translate coordinates to be relative to the volume's center
#         z = d_coords.float() - (D - 1) / 2.0
#         y = h_coords.float() - (H - 1) / 2.0
#         x = w_coords.float() - (W - 1) / 2.0

#         # Calculate spherical coordinates
#         r = torch.sqrt(x**2 + y**2 + z**2)
#         # Add a small epsilon to avoid division by zero
#         r_prime = r.clone()
#         r_prime[r_prime == 0] = 1e-9
        
#         theta = torch.acos(z / r_prime)
#         phi = torch.atan2(y, x)

#         # Stack the coordinate channels
#         return torch.stack([r, theta, phi], dim=0)

#     def __init__(self, data_list_file: str, template_path: str, target_size=(128, 128, 128)):
#         print(f"Loading subject paths from: {data_list_file}")
#         with open(data_list_file, 'r') as file:
#             self.subject_paths = file.read().splitlines()
#         print(f"Loaded {len(self.subject_paths)} subject paths.")
        
#         self.target_size = target_size
#         print(f"Target size set to: {self.target_size}")

#         # --- OPTIMIZATION: Pre-calculate the spherical coordinate grid ONCE ---
#         print("Pre-calculating spherical coordinate grid...")
#         self.spherical_coords = self._create_spherical_coordinates(self.target_size)
#         print(f"Coordinate grid created with shape: {self.spherical_coords.shape}")

#         # --- Process Moving Template (as before) ---
#         print(f"Loading template from: {template_path}")
#         moving_template_onehot = np.load(template_path, mmap_mode='r')
#         label_map = np.argmax(moving_template_onehot, axis=-1)
#         binary_mask_np = (label_map > 0).astype(np.float32)
#         moving_template = torch.from_numpy(binary_mask_np).unsqueeze(0)

#         moving_template_resized = F.interpolate(
#             moving_template.unsqueeze(0), 
#             size=self.target_size,
#             mode='nearest'
#         ).squeeze(0)

#         # Concatenate the binary mask with the PRE-CALCULATED coordinate grid
#         self.moving_template = torch.cat([moving_template_resized, self.spherical_coords], dim=0)
#         print(f"Final moving template (4-channel) shape: {self.moving_template.shape}")

#     def __len__(self):
#         return len(self.subject_paths)

#     def __getitem__(self, idx):
#         # --- Process Fixed Map ---
#         fixed_map_onehot = np.load(self.subject_paths[idx], mmap_mode='r')
#         label_map = np.argmax(fixed_map_onehot, axis=-1)
#         binary_mask_np = (label_map > 0).astype(np.float32)
#         fixed_map = torch.from_numpy(binary_mask_np).unsqueeze(0)

#         fixed_map_resized = F.interpolate(
#             fixed_map.unsqueeze(0),
#             size=self.target_size,
#             mode='nearest'
#         ).squeeze(0)
    
#         # --- OPTIMIZATION: Concatenate with the pre-calculated grid ---
#         # This is now just a simple, fast concatenation operation.
#         fixed_map_multichannel = torch.cat([fixed_map_resized, self.spherical_coords], dim=0)
        
#         return self.moving_template, fixed_map_multichannel

# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# import numpy as np
# import os

# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# import numpy as np
# import os

# class SegDataset(Dataset):
#     """
#     Dataset for 3D image registration using a binary mask.
    
#     It prepares multi-channel tensors where the object is represented by a single
#     binary channel (0 for background, 1 for the object). The final tensor has 4 channels:
#     - Channel 0: Binary segmentation mask.
#     - Channel 1: r (radius) coordinate.
#     - Channel 2: theta (polar angle) coordinate.
#     - Channel 3: phi (azimuthal angle) coordinate.
#     """
    
#     @staticmethod
#     def create_multichannel_input(binary_mask: torch.Tensor) -> torch.Tensor:
#         """
#         Converts a binary mask into a multi-channel tensor including
#         spherical coordinates for every point.

#         Args:
#             binary_mask: A tensor of shape (1, D, H, W) representing the object.

#         Returns:
#             A multi-channel tensor of shape (4, D, H, W).
#         """
#         _, D, H, W = binary_mask.shape
#         device = binary_mask.device

#         # Create a grid of coordinates for the entire volume
#         d_coords, h_coords, w_coords = torch.meshgrid(
#             torch.arange(D, device=device),
#             torch.arange(H, device=device),
#             torch.arange(W, device=device),
#             indexing='ij'
#         )

#         # Translate coordinates to be relative to the volume's center
#         z = d_coords.float() - (D - 1) / 2.0
#         y = h_coords.float() - (H - 1) / 2.0
#         x = w_coords.float() - (W - 1) / 2.0

#         # Calculate spherical coordinates
#         r = torch.sqrt(x**2 + y**2 + z**2)
#         # Add a small epsilon to r where it is zero to avoid division by zero
#         r_prime = r.clone()
#         r_prime[r_prime == 0] = 1e-9
        
#         theta = torch.acos(z / r_prime)
#         phi = torch.atan2(y, x)

#         # Stack the new coordinate channels
#         spherical_channels = torch.stack([r, theta, phi], dim=0)

#         # Concatenate the binary mask with the new spherical channels
#         multichannel_tensor = torch.cat([binary_mask, spherical_channels], dim=0)
        
#         return multichannel_tensor

#     def __init__(self, data_list_file: str, template_path: str, target_size=(128, 128, 128)):
#         print(f"Loading subject paths from: {data_list_file}")
#         with open(data_list_file, 'r') as file:
#             self.subject_paths = file.read().splitlines()
#         print(f"Loaded {len(self.subject_paths)} subject paths.")
        
#         self.target_size = target_size
#         print(f"Target size set to: {self.target_size}")

#         # --- Process Moving Template ---
#         print(f"Loading template from: {template_path}")
#         moving_template_onehot = np.load(template_path, mmap_mode='r')
#         print(f"Loaded template with shape (one-hot): {moving_template_onehot.shape}")

#         label_map = np.argmax(moving_template_onehot, axis=-1)
#         print(f"Converted to label map with shape: {label_map.shape}, unique labels: {np.unique(label_map)}")

#         binary_mask_np = (label_map > 0).astype(np.float32)
#         print(f"Binary mask created with shape: {binary_mask_np.shape}, values: {np.unique(binary_mask_np)}")

#         moving_template = torch.from_numpy(binary_mask_np).unsqueeze(0)  # Shape: (1, D, H, W)
#         print(f"Converted to torch tensor with shape: {moving_template.shape}")

#         moving_template_resized = F.interpolate(
#             moving_template.unsqueeze(0), 
#             size=self.target_size,
#             mode='nearest'
#         ).squeeze(0)
#         print(f"Resized moving template to: {moving_template_resized.shape}")

#         self.moving_template = self.create_multichannel_input(moving_template_resized)
#         print(f"Final moving template (4-channel) shape: {self.moving_template.shape}")


#     def __len__(self):
#         return len(self.subject_paths)

#     def __getitem__(self, idx):
#         # --- Process Fixed Map ---
#         # Load the 5-channel one-hot numpy array
#         fixed_map_onehot = np.load(self.subject_paths[idx], mmap_mode='r')
#         # Convert to a single binary mask
#         label_map = np.argmax(fixed_map_onehot, axis=-1)
#         binary_mask_np = (label_map > 0).astype(np.float32)
        
#         fixed_map = torch.from_numpy(binary_mask_np).unsqueeze(0) # Shape: (1, D, H, W)

#         # Resize fixed map
#         fixed_map_resized = F.interpolate(
#             fixed_map.unsqueeze(0),
#             size=self.target_size,
#             mode='nearest'
#         ).squeeze(0)
    
#         # Create the final 4-channel input tensor
#         fixed_map_multichannel = self.create_multichannel_input(fixed_map_resized)
        
#         return self.moving_template, fixed_map_multichannel

# class SegDataset(Dataset):
    
#     def mask_to_spherical(label_map: np.ndarray) -> np.ndarray:
#         """
#         Converts a 3D mask with labels 0 to 4 into spherical coordinates
#         for each class relative to its own centroid.

#         Args:
#             label_map: A 3D NumPy array with integer values from 0 to 4.

#         Returns:
#             A NumPy array of shape (N, 4) where each row is (r_scaled, theta, phi, class_id).
#         """
#         all_coords = []

#         for class_id in range(1, 5):  # Assuming classes 1 to 4 are valid
#             binary_mask = (label_map == class_id)
#             coords = np.argwhere(binary_mask).astype(np.float64)
#             if coords.shape[0] == 0:
#                 continue

#             centroid = coords.mean(axis=0)
#             translated_coords = coords - centroid
#             z, y, x = translated_coords[:, 0], translated_coords[:, 1], translated_coords[:, 2]

#             r_prime = np.sqrt(x**2 + y**2 + z**2)
#             r_prime[r_prime == 0] = 1e-9

#             theta = np.arccos(z / r_prime)
#             phi = np.arctan2(y, x)
#             phi[phi < 0] += 2 * np.pi

#             r_max = np.max(r_prime)
#             r_scaled = r_prime * (100 / r_max) if r_max > 0 else 0

#             spherical_coords = np.vstack((r_scaled, theta, phi)).T
#             class_column = np.full((spherical_coords.shape[0], 1), class_id)
#             spherical_with_class = np.hstack((spherical_coords, class_column))

#             all_coords.append(spherical_with_class)

#         if all_coords:
#             return np.vstack(all_coords)
#         else:
#             return np.empty((0, 4))
    
#     def __init__(self, data_list_file: str, template_path: str, target_size=(128, 128, 128)):
#         # Read subject paths from txt file
#         with open(data_list_file, 'r') as file:
#             self.subject_paths = file.read().splitlines()

#         self.moving_template = np.load(template_path, mmap_mode='r', allow_pickle=True)
#         self.moving_template = torch.from_numpy(self.moving_template).float()
#         # self.moving_template = self.moving_template.permute(3, 0, 1, 2)  # (5, D, H, W)

#         self.moving_template = F.interpolate(
#             self.moving_template.unsqueeze(0), 
#             size=target_size,
#             mode='nearest'
#         ).squeeze(0)  # Back to (5, 128, 128, 128)
        
#         label_map = np.argmax(self.moving_template, axis=0)  
#         self.spherical_moving_template = self.mask_to_spherical(label_map) # shape: (N, 4)

#         self.target_size = target_size

#     def __len__(self):
#         return len(self.subject_paths)

#     def __getitem__(self, idx):

#         fixed_map = np.load(self.subject_paths[idx], mmap_mode='r')  # (D, H, W, 5)
#         fixed_map = torch.from_numpy(fixed_map).float()
#         # fixed_map = fixed_map.permute(3, 0, 1, 2)  # (5, D, H, W)

#         # Resize fixed map
#         fixed_map = F.interpolate(
#             fixed_map.unsqueeze(0),  # (1, 5, D, H, W)
#             size=self.target_size,
#             mode='nearest'
#         ).squeeze(0)  # (5, 128, 128, 128)
    
#         assert torch.all(torch.sum(fixed_map, dim=0) == 1),  "Invalid onehot"
#         assert torch.all(torch.sum(self.moving_template, dim=0) == 1), "Template corrupted"
        
#         label_map = np.argmax(fixed_map, axis=0)  
#         spherical_fixed_map = self.mask_to_spherical(label_map) # shape: (N, 4)
        
#         return self.moving_template, spherical_fixed_map