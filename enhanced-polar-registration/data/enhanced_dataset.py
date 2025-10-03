"""
Enhanced Dataset for Polar Coordinate Registration
Includes advanced data loading, preprocessing, and augmentation
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any
import logging

from .data_augmentation import PolarDataAugmentation
from .data_utils import create_polar_coordinates, normalize_segmentation


class EnhancedSegDataset(Dataset):
    """
    Enhanced segmentation dataset with polar coordinate support and advanced preprocessing
    """
    
    def __init__(self, 
                 data_list_file: str, 
                 template_path: str,
                 target_size: Tuple[int, int, int] = (128, 128, 128),
                 use_polar_coords: bool = True,
                 use_augmentation: bool = True,
                 augmentation_prob: float = 0.5,
                 cache_data: bool = False):
        """
        Initialize enhanced segmentation dataset
        
        Args:
            data_list_file: Path to file listing subject paths
            template_path: Path to template segmentation map
            target_size: Target size for resizing
            use_polar_coords: Whether to include polar coordinates
            use_augmentation: Whether to apply data augmentation
            augmentation_prob: Probability of applying augmentation
            cache_data: Whether to cache loaded data in memory
        """
        self.target_size = target_size
        self.use_polar_coords = use_polar_coords
        self.use_augmentation = use_augmentation
        self.cache_data = cache_data
        
        # Read subject paths
        with open(data_list_file, 'r') as file:
            self.subject_paths = file.read().splitlines()
        
        # Load and preprocess template
        self.moving_template = self._load_and_preprocess_template(template_path)
        
        # Initialize data augmentation
        if use_augmentation:
            self.augmentation = PolarDataAugmentation(
                target_size=target_size,
                augmentation_prob=augmentation_prob
            )
        else:
            self.augmentation = None
        
        # Data cache
        if cache_data:
            self.data_cache = {}
        else:
            self.data_cache = None
        
        # Create polar coordinate grid
        if use_polar_coords:
            self.polar_grid = create_polar_coordinates(target_size)
            # Transpose from [D, H, W, 6] to [6, D, H, W] for consistency
            self.polar_grid = self.polar_grid.permute(3, 0, 1, 2)
        
        logging.info(f"Enhanced dataset initialized with {len(self.subject_paths)} samples")
        logging.info(f"Template shape: {self.moving_template.shape}")
        logging.info(f"Target size: {target_size}")
        logging.info(f"Polar coordinates: {use_polar_coords}")
        logging.info(f"Data augmentation: {use_augmentation}")
    
    def _load_and_preprocess_template(self, template_path: str) -> torch.Tensor:
        """Load and preprocess template segmentation map"""
        try:
            # Load template
            template = np.load(template_path, mmap_mode='r', allow_pickle=True)
            template = torch.from_numpy(template).float()
            
            # Resize to target size
            template = F.interpolate(
                template.unsqueeze(0), 
                size=self.target_size,
                mode='nearest'
            ).squeeze(0)
            
            # Normalize
            template = normalize_segmentation(template)
            
            # Validate template
            assert torch.all(torch.sum(template, dim=0) == 1), "Template corrupted - not one-hot"
            
            return template
            
        except Exception as e:
            logging.error(f"Error loading template from {template_path}: {e}")
            raise
    
    def _load_subject_data(self, subject_path: str) -> torch.Tensor:
        """Load and preprocess subject segmentation data"""
        try:
            # Check cache first
            if self.cache_data and subject_path in self.data_cache:
                return self.data_cache[subject_path]
            
            # Load data
            fixed_map = np.load(subject_path, mmap_mode='r')
            fixed_map = torch.from_numpy(fixed_map).float()
            
            # Resize to target size
            fixed_map = F.interpolate(
                fixed_map.unsqueeze(0),
                size=self.target_size,
                mode='nearest'
            ).squeeze(0)
            
            # Normalize
            fixed_map = normalize_segmentation(fixed_map)
            
            # Validate
            assert torch.all(torch.sum(fixed_map, dim=0) == 1), f"Invalid one-hot for {subject_path}"
            
            # Cache if enabled
            if self.cache_data:
                self.data_cache[subject_path] = fixed_map
            
            return fixed_map
            
        except Exception as e:
            logging.error(f"Error loading subject data from {subject_path}: {e}")
            raise
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.subject_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item with enhanced features"""
        # Load fixed segmentation map
        fixed_map = self._load_subject_data(self.subject_paths[idx])
        
        # Create sample
        sample = {
            'moving': self.moving_template.clone(),
            'fixed': fixed_map,
            'subject_path': self.subject_paths[idx]
        }
        
        # Add polar coordinates if requested
        if self.use_polar_coords:
            sample['polar_coords'] = self.polar_grid.clone()
        
        # Apply augmentation if enabled
        if self.augmentation is not None:
            sample = self.augmentation(sample)
        
        return sample
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a specific sample"""
        return {
            'index': idx,
            'subject_path': self.subject_paths[idx],
            'template_shape': self.moving_template.shape,
            'target_size': self.target_size,
            'has_polar_coords': self.use_polar_coords,
            'has_augmentation': self.use_augmentation
        }
    
    def clear_cache(self):
        """Clear data cache to free memory"""
        if self.cache_data and self.data_cache is not None:
            self.data_cache.clear()
            logging.info("Data cache cleared")


class MultiScaleDataset(Dataset):
    """
    Multi-scale dataset that provides data at different resolutions
    """
    
    def __init__(self, 
                 base_dataset: EnhancedSegDataset,
                 scales: Tuple[int, ...] = (1, 2, 4)):
        self.base_dataset = base_dataset
        self.scales = scales
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get base sample
        sample = self.base_dataset[idx]
        
        # Create multi-scale versions
        multi_scale_sample = {}
        for scale in self.scales:
            if scale > 1:
                # Downsample
                scale_size = tuple(s // scale for s in self.base_dataset.target_size)
                
                moving_scale = F.interpolate(
                    sample['moving'].unsqueeze(0),
                    size=scale_size,
                    mode='nearest'
                ).squeeze(0)
                
                fixed_scale = F.interpolate(
                    sample['fixed'].unsqueeze(0),
                    size=scale_size,
                    mode='nearest'
                ).squeeze(0)
                
                multi_scale_sample[f'moving_scale_{scale}'] = moving_scale
                multi_scale_sample[f'fixed_scale_{scale}'] = fixed_scale
                
                if 'polar_coords' in sample:
                    polar_scale = F.interpolate(
                        sample['polar_coords'].unsqueeze(0),
                        size=scale_size,
                        mode='trilinear',
                        align_corners=False
                    ).squeeze(0)
                    multi_scale_sample[f'polar_coords_scale_{scale}'] = polar_scale
            else:
                multi_scale_sample[f'moving_scale_{scale}'] = sample['moving']
                multi_scale_sample[f'fixed_scale_{scale}'] = sample['fixed']
                if 'polar_coords' in sample:
                    multi_scale_sample[f'polar_coords_scale_{scale}'] = sample['polar_coords']
        
        # Add metadata
        multi_scale_sample['subject_path'] = sample['subject_path']
        multi_scale_sample['scales'] = self.scales
        
        return multi_scale_sample


class BalancedDataset(Dataset):
    """
    Balanced dataset that ensures equal representation of different classes
    """
    
    def __init__(self, 
                 data_list_file: str,
                 template_path: str,
                 target_size: Tuple[int, int, int] = (128, 128, 128),
                 balance_classes: bool = True):
        self.target_size = target_size
        self.balance_classes = balance_classes
        
        # Load base dataset
        self.base_dataset = EnhancedSegDataset(
            data_list_file, template_path, target_size,
            use_polar_coords=True, use_augmentation=False, cache_data=True
        )
        
        if balance_classes:
            self._create_balanced_indices()
        else:
            self.balanced_indices = list(range(len(self.base_dataset)))
    
    def _create_balanced_indices(self):
        """Create balanced indices based on class distribution"""
        # Analyze class distribution in the dataset
        class_counts = {}
        
        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset[idx]
            fixed_map = sample['fixed']
            
            # Count classes
            class_labels = torch.argmax(fixed_map, dim=0)
            unique_classes, counts = torch.unique(class_labels, return_counts=True)
            
            for cls, count in zip(unique_classes, counts):
                if cls.item() not in class_counts:
                    class_counts[cls.item()] = []
                class_counts[cls.item()].append((idx, count.item()))
        
        # Create balanced sampling
        self.balanced_indices = []
        max_samples_per_class = max(len(indices) for indices in class_counts.values())
        
        for class_id, indices in class_counts.items():
            # Repeat indices to balance classes
            repeat_factor = max_samples_per_class // len(indices)
            remainder = max_samples_per_class % len(indices)
            
            balanced_indices = indices * repeat_factor + indices[:remainder]
            self.balanced_indices.extend([idx for idx, _ in balanced_indices])
        
        logging.info(f"Created balanced dataset with {len(self.balanced_indices)} samples")
        logging.info(f"Class distribution: {[(cls, len(indices)) for cls, indices in class_counts.items()]}")
    
    def __len__(self) -> int:
        return len(self.balanced_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        actual_idx = self.balanced_indices[idx]
        return self.base_dataset[actual_idx]
