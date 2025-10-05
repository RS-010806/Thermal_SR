"""
Data loader for Landsat 8 SSL4EO thermal super-resolution dataset.

Handles:
- Loading multi-band GeoTIFF files
- Simulating low-resolution thermal bands
- Data augmentation
- Physics-based preprocessing
"""

import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from pathlib import Path
import random
from typing import Dict, Tuple, Optional, List
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    A = None


class ThermalSRDataset(Dataset):
    """
    Dataset for thermal super-resolution training.
    
    The dataset contains Landsat 8 scenes with all bands at 30m resolution.
    We simulate the low-resolution thermal bands by downsampling with PSF.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        patch_size: int = 192,
        scale_factor: int = 2,
        simulate_lr: bool = True,
        augment: bool = True,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            data_dir: Path to ssl4eo_l_oli_tirs_toa_benchmark directory
            split: 'train', 'val', or 'test'
            patch_size: Size of patches to extract
            scale_factor: Super-resolution scale factor (2 or 4)
            simulate_lr: Whether to simulate low-res thermal from high-res
            augment: Whether to apply data augmentation
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.simulate_lr = simulate_lr
        self.augment = augment and split == 'train'
        
        # Find all scenes
        self.scene_paths = list(self.data_dir.glob("*/*/all_bands.tif"))
        
        # Split dataset
        random.seed(seed)
        random.shuffle(self.scene_paths)
        
        n_total = len(self.scene_paths)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        
        if split == 'test':
            self.scene_paths = self.scene_paths[:n_test]
        elif split == 'val':
            self.scene_paths = self.scene_paths[n_test:n_test + n_val]
        else:  # train
            self.scene_paths = self.scene_paths[n_test + n_val:]
            
        print(f"Loaded {len(self.scene_paths)} scenes for {split} split")
        
        # Define augmentation pipeline
        if HAS_ALBUMENTATIONS:
            self.transform = A.Compose([
                A.RandomCrop(patch_size, patch_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]) if self.augment else A.CenterCrop(patch_size, patch_size)
        else:
            self.transform = None
        
        # Band indices (0-indexed)
        self.optical_bands = list(range(9))  # Bands 1-9
        self.thermal_bands = [9, 10]  # Bands 10-11
        
        # Normalization statistics (computed from a sample of the dataset)
        # These should be computed properly from the full dataset
        self.optical_mean = np.array([74.3, 66.0, 60.6, 65.5, 116.5, 140.3, 105.6, 61.6, 0.08])
        self.optical_std = np.array([5.4, 6.7, 8.4, 13.2, 10.5, 23.3, 23.7, 10.3, 0.26])
        self.thermal_mean = np.array([230.3, 222.8])
        self.thermal_std = np.array([11.1, 10.3])
        
    def __len__(self):
        # Multiple patches per scene
        return len(self.scene_paths) * (4 if self.split == 'train' else 1)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Get scene index and patch index
        patches_per_scene = 4 if self.split == 'train' else 1
        scene_idx = idx // patches_per_scene
        
        scene_path = self.scene_paths[scene_idx]
        
        # Load the scene
        with rasterio.open(scene_path) as src:
            # Read all bands
            all_bands = src.read()  # Shape: (11, H, W)
            
            # Convert to float32 and scale if needed
            all_bands = all_bands.astype(np.float32)
            
        # Transpose to HWC for albumentations
        all_bands = np.transpose(all_bands, (1, 2, 0))  # Shape: (H, W, 11)
        
        # Apply spatial augmentation/cropping
        if self.transform is not None:
            transformed = self.transform(image=all_bands)
            all_bands = transformed['image']  # Shape: (patch_size, patch_size, 11)
        else:
            # Fallback without albumentations
            H, W, C = all_bands.shape
            if self.augment and self.split == 'train':
                # Random crop
                y = random.randint(0, H - self.patch_size)
                x = random.randint(0, W - self.patch_size)
                all_bands = all_bands[y:y+self.patch_size, x:x+self.patch_size]
                
                # Random flips
                if random.random() > 0.5:
                    all_bands = np.flip(all_bands, axis=0).copy()
                if random.random() > 0.5:
                    all_bands = np.flip(all_bands, axis=1).copy()
                if random.random() > 0.5:
                    all_bands = np.rot90(all_bands, k=random.randint(0, 3)).copy()
            else:
                # Center crop
                y = (H - self.patch_size) // 2
                x = (W - self.patch_size) // 2
                all_bands = all_bands[y:y+self.patch_size, x:x+self.patch_size]
        
        # Split optical and thermal bands
        optical = all_bands[:, :, self.optical_bands]  # Shape: (H, W, 9)
        thermal_hr = all_bands[:, :, self.thermal_bands]  # Shape: (H, W, 2)
        
        # Normalize
        optical = (optical - self.optical_mean) / (self.optical_std + 1e-8)
        thermal_hr = (thermal_hr - self.thermal_mean) / (self.thermal_std + 1e-8)
        
        # Convert to tensors and transpose to CHW
        optical = torch.from_numpy(optical).permute(2, 0, 1).float()
        thermal_hr = torch.from_numpy(thermal_hr).permute(2, 0, 1).float()
        
        # Simulate low-resolution thermal if requested
        if self.simulate_lr:
            thermal_lr = self._simulate_low_res(thermal_hr)
        else:
            # Use the provided 30m thermal as "low-res" and need external HR reference
            thermal_lr = thermal_hr
            thermal_hr = None  # Would need external HR reference
            
        sample = {
            'thermal_lr': thermal_lr,
            'thermal_hr': thermal_hr,
            'optical': optical,
            'scene_id': str(scene_path.parent.name),
            'patch_idx': idx % patches_per_scene
        }
        
        return sample
    
    def _simulate_low_res(self, thermal_hr: torch.Tensor) -> torch.Tensor:
        """
        Simulate low-resolution thermal imagery from high-resolution.
        
        This simulates the actual TIRS imaging process:
        1. Apply PSF (Point Spread Function)
        2. Downsample
        3. Add sensor noise
        4. Upsample back to original size for training
        """
        # Simple Gaussian blur to simulate PSF
        from torchvision.transforms import GaussianBlur
        
        # PSF blur with kernel size proportional to scale factor
        kernel_size = 2 * self.scale_factor + 1
        sigma = self.scale_factor / 2.0
        
        blurred = GaussianBlur(kernel_size, sigma)(thermal_hr)
        
        # Downsample
        downsampled = torch.nn.functional.interpolate(
            blurred.unsqueeze(0),
            scale_factor=1/self.scale_factor,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Add realistic sensor noise (NEΔT ~ 0.1K normalized)
        if self.split == 'train':
            noise_level = 0.1 / self.thermal_std.mean()  # Normalized noise
            noise = torch.randn_like(downsampled) * noise_level
            downsampled = downsampled + noise
        
        # Upsample back to original size (this is what we get from resampled data)
        upsampled = torch.nn.functional.interpolate(
            downsampled.unsqueeze(0),
            size=thermal_hr.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return upsampled
    
    def denormalize_thermal(self, thermal_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized thermal values back to brightness temperature."""
        thermal_mean = torch.tensor(self.thermal_mean).view(-1, 1, 1).to(thermal_norm.device)
        thermal_std = torch.tensor(self.thermal_std).view(-1, 1, 1).to(thermal_norm.device)
        
        return thermal_norm * thermal_std + thermal_mean
    
    def denormalize_optical(self, optical_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized optical values back to original scale."""
        optical_mean = torch.tensor(self.optical_mean).view(-1, 1, 1).to(optical_norm.device)
        optical_std = torch.tensor(self.optical_std).view(-1, 1, 1).to(optical_norm.device)
        
        return optical_norm * optical_std + optical_mean


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    patch_size: int = 192,
    scale_factor: int = 2,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for training
        patch_size: Size of patches to extract
        scale_factor: Super-resolution scale factor
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        **kwargs: Additional arguments for ThermalSRDataset
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = ThermalSRDataset(
        data_dir=data_dir,
        split='train',
        patch_size=patch_size,
        scale_factor=scale_factor,
        augment=True,
        **kwargs
    )
    
    val_dataset = ThermalSRDataset(
        data_dir=data_dir,
        split='val',
        patch_size=patch_size,
        scale_factor=scale_factor,
        augment=False,
        **kwargs
    )
    
    test_dataset = ThermalSRDataset(
        data_dir=data_dir,
        split='test',
        patch_size=patch_size,
        scale_factor=scale_factor,
        augment=False,
        **kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Test one at a time for detailed evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


class InferenceDataset(Dataset):
    """
    Dataset for inference on full scenes without patching.
    """
    
    def __init__(self, scene_paths: List[str], normalize: bool = True):
        """
        Args:
            scene_paths: List of paths to scene files
            normalize: Whether to normalize the data
        """
        self.scene_paths = scene_paths
        self.normalize = normalize
        
        # Same normalization statistics as training
        self.optical_mean = np.array([74.3, 66.0, 60.6, 65.5, 116.5, 140.3, 105.6, 61.6, 0.08])
        self.optical_std = np.array([5.4, 6.7, 8.4, 13.2, 10.5, 23.3, 23.7, 10.3, 0.26])
        self.thermal_mean = np.array([230.3, 222.8])
        self.thermal_std = np.array([11.1, 10.3])
        
    def __len__(self):
        return len(self.scene_paths)
    
    def __getitem__(self, idx):
        scene_path = self.scene_paths[idx]
        
        with rasterio.open(scene_path) as src:
            # Read all bands
            all_bands = src.read().astype(np.float32)
            
            # Get metadata
            transform = src.transform
            crs = src.crs
            
        # Split bands
        optical = all_bands[:9]  # Bands 1-9
        thermal = all_bands[9:11]  # Bands 10-11
        
        # Normalize if requested
        if self.normalize:
            for i in range(9):
                optical[i] = (optical[i] - self.optical_mean[i]) / (self.optical_std[i] + 1e-8)
            for i in range(2):
                thermal[i] = (thermal[i] - self.thermal_mean[i]) / (self.thermal_std[i] + 1e-8)
                
        # Convert to tensors
        optical = torch.from_numpy(optical).float()
        thermal = torch.from_numpy(thermal).float()
        
        return {
            'thermal': thermal,
            'optical': optical,
            'scene_path': str(scene_path),
            'transform': transform,
            'crs': crs
        }
