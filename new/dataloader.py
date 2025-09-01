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
            template_torch = torch.from_numpy(template_npy)
            labels_fixed = torch.argmax(template_torch, dim=0)
            self.fixed_volume = (labels_fixed > 0).float()
            
            self.fixed_volume = F.interpolate(
                self.fixed_volume.unsqueeze(0).unsqueeze(0),
                size=self.target_size,
                mode='nearest'
            ).squeeze(0).squeeze(0)
            
            # Ensure the volume is properly normalized
            self.fixed_volume = torch.clamp(self.fixed_volume, 0.0, 1.0)

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
        moving_torch = torch.from_numpy(moving_npy)
        labels_moving = torch.argmax(moving_torch, dim=0)
        moving_volume = (labels_moving > 0).float()
        
        moving_volume = F.interpolate(
            moving_volume.unsqueeze(0).unsqueeze(0), 
            size=self.target_size,
            mode='nearest'
        ).squeeze(0).squeeze(0)
        
        # Ensure the volume is properly normalized
        moving_volume = torch.clamp(moving_volume, 0.0, 1.0)
        
        return moving_volume, self.fixed_volume