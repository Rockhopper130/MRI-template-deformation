import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class OASISDataset(Dataset):
    """
    Custom PyTorch Dataset for loading OASIS brain scans.
    It loads a moving scan and the fixed template scan.
    """
    def __init__(self, data_dir, scan_ids, template_path):
        self.data_dir = data_dir
        self.scan_ids = scan_ids
        self.template_path = template_path
        self.target_size = (128, 128, 128)
        self._load_template()

    def _load_template(self):
        """Pre-loads the fixed template volume to save time."""
        try:
            template_npy = np.load(self.template_path)
            template_torch = torch.from_numpy(template_npy).float()  # [C, D, H, W] one-hot
            # One-hot template resized with nearest to preserve labels
            self.fixed_onehot = F.interpolate(
                template_torch.unsqueeze(0),  # [1, C, D, H, W]
                size=self.target_size,
                mode='nearest'
            ).squeeze(0)  # [C, D, H, W]
            # Single-channel intensity/mask for VSP
            labels_fixed = torch.argmax(self.fixed_onehot, dim=0)
            self.fixed_intensity = (labels_fixed > 0).float()  # [D,H,W]

        except FileNotFoundError:
            print(f"ERROR: Template file not found at {self.template_path}")
            raise

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, idx):
        """Loads a moving volume and returns it with the fixed volume."""
        scan_id = self.scan_ids[idx]
        moving_path = os.path.join(self.data_dir, scan_id, 'seg4_onehot.npy')
        
        moving_npy = np.load(moving_path)
        moving_torch = torch.from_numpy(moving_npy).float()  # [C, D, H, W]
        moving_onehot = F.interpolate(
            moving_torch.unsqueeze(0),  # [1, C, D, H, W]
            size=self.target_size,
            mode='nearest'
        ).squeeze(0)  # [C, D, H, W]
        labels_moving = torch.argmax(moving_onehot, dim=0)
        moving_intensity = (labels_moving > 0).float()  # [D,H,W]

        return moving_intensity, self.fixed_intensity, moving_onehot, self.fixed_onehot