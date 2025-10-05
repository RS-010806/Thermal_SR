"""
Utility functions for thermal super-resolution project.
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Union, List
import cv2
import rasterio
from rasterio.transform import from_bounds


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   checkpoint_path: str,
                   **kwargs):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Add any additional items
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, 
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: torch.device = torch.device('cpu')):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: {checkpoint_path} (epoch {epoch}, loss {loss:.4f})")
    
    return model, optimizer, epoch, checkpoint


def visualize_results(results_dict: Dict[str, torch.Tensor], 
                     save_path: Optional[str] = None,
                     dpi: int = 150) -> plt.Figure:
    """
    Visualize thermal super-resolution results.
    
    Args:
        results_dict: Dictionary containing:
            - thermal_lr: Low-resolution thermal input
            - thermal_hr: High-resolution thermal ground truth
            - thermal_sr: Super-resolved thermal output
            - optical_rgb: RGB composite from optical bands (optional)
            - emissivity: Estimated emissivity map (optional)
        save_path: Path to save the figure
        dpi: DPI for saved figure
        
    Returns:
        Figure object
    """
    num_plots = len(results_dict)
    fig, axes = plt.subplots(1, num_plots, figsize=(4*num_plots, 4), dpi=dpi)
    
    if num_plots == 1:
        axes = [axes]
        
    for idx, (key, data) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        # Handle different data types
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
            
        # Handle multi-channel data
        if len(data.shape) == 3:
            if data.shape[0] in [1, 2, 3]:  # CHW format
                if data.shape[0] == 1:
                    data = data[0]  # Remove channel dimension
                elif data.shape[0] == 2:
                    # Average thermal bands
                    data = data.mean(axis=0)
                else:  # 3 channels - RGB
                    data = np.transpose(data, (1, 2, 0))
                    
        # Plot based on data type
        if key.startswith('thermal'):
            # Thermal data - use hot colormap
            if len(data.shape) == 2:
                im = ax.imshow(data, cmap='hot', interpolation='bilinear')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f'{key.replace("_", " ").title()} (K)')
            else:
                ax.imshow(data, interpolation='bilinear')
                ax.set_title(key.replace("_", " ").title())
                
        elif key == 'optical_rgb':
            # RGB composite - normalize for display
            data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
            ax.imshow(data_norm, interpolation='bilinear')
            ax.set_title('Optical RGB')
            
        elif key == 'emissivity':
            # Emissivity map
            im = ax.imshow(data, cmap='viridis', vmin=0.9, vmax=1.0, interpolation='bilinear')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('Emissivity')
            
        else:
            # Default visualization
            if len(data.shape) == 2:
                im = ax.imshow(data, cmap='gray', interpolation='bilinear')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(data, interpolation='bilinear')
            ax.set_title(key.replace("_", " ").title())
            
        ax.axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
    return fig


def save_geotiff(data: np.ndarray, 
                 save_path: str,
                 transform: rasterio.transform.Affine,
                 crs: str = 'EPSG:4326',
                 band_descriptions: Optional[List[str]] = None):
    """
    Save data as a GeoTIFF file with proper georeferencing.
    
    Args:
        data: Data to save (bands, height, width)
        save_path: Path to save the GeoTIFF
        transform: Affine transformation matrix
        crs: Coordinate reference system
        band_descriptions: Optional descriptions for each band
    """
    # Handle single band case
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
        
    num_bands, height, width = data.shape
    
    # Create GeoTIFF
    with rasterio.open(
        save_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=num_bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        # Write bands
        for i in range(num_bands):
            dst.write(data[i], i + 1)
            
            # Set band description if provided
            if band_descriptions and i < len(band_descriptions):
                dst.set_band_description(i + 1, band_descriptions[i])
                
    print(f"Saved GeoTIFF: {save_path}")


def create_false_color_composite(optical_bands: torch.Tensor, 
                                band_indices: List[int] = [4, 3, 2]) -> np.ndarray:
    """
    Create false color composite from optical bands.
    
    Args:
        optical_bands: Optical bands tensor (C, H, W)
        band_indices: Indices of bands to use for R, G, B
        
    Returns:
        False color composite (H, W, 3)
    """
    # Extract bands
    composite = []
    for idx in band_indices:
        if idx < optical_bands.shape[0]:
            band = optical_bands[idx].detach().cpu().numpy()
            # Normalize to [0, 1]
            band_norm = (band - band.min()) / (band.max() - band.min() + 1e-8)
            composite.append(band_norm)
            
    # Stack into RGB
    composite = np.stack(composite, axis=-1)
    
    # Enhance contrast
    composite = np.clip(composite * 1.5, 0, 1)
    
    return composite


def apply_thermal_colormap(thermal_data: np.ndarray, 
                          colormap: str = 'hot',
                          temp_min: Optional[float] = None,
                          temp_max: Optional[float] = None) -> np.ndarray:
    """
    Apply colormap to thermal data for visualization.
    
    Args:
        thermal_data: Thermal data in Kelvin
        colormap: Matplotlib colormap name
        temp_min: Minimum temperature for scaling
        temp_max: Maximum temperature for scaling
        
    Returns:
        Colored thermal image (H, W, 3)
    """
    # Set temperature range if not provided
    if temp_min is None:
        temp_min = thermal_data.min()
    if temp_max is None:
        temp_max = thermal_data.max()
        
    # Normalize to [0, 1]
    thermal_norm = (thermal_data - temp_min) / (temp_max - temp_min + 1e-8)
    thermal_norm = np.clip(thermal_norm, 0, 1)
    
    # Apply colormap
    cmap = plt.cm.get_cmap(colormap)
    thermal_colored = cmap(thermal_norm)[:, :, :3]  # Remove alpha channel
    
    return thermal_colored


def compute_gradient_map(image: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient magnitude map using Sobel operators.
    
    Args:
        image: Input image (C, H, W)
        
    Returns:
        Gradient magnitude map (C, H, W)
    """
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=image.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=image.device)
    
    # Reshape kernels for conv2d
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(image.shape[0], 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(image.shape[0], 1, 1, 1)
    
    # Compute gradients
    grad_x = torch.nn.functional.conv2d(
        image.unsqueeze(0), sobel_x, padding=1, groups=image.shape[0]
    ).squeeze(0)
    
    grad_y = torch.nn.functional.conv2d(
        image.unsqueeze(0), sobel_y, padding=1, groups=image.shape[0]
    ).squeeze(0)
    
    # Compute magnitude
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
    return grad_mag


def create_comparison_plot(lr_image: np.ndarray,
                          sr_image: np.ndarray,
                          hr_image: np.ndarray,
                          metric_values: Dict[str, float],
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create detailed comparison plot with zoomed regions and metrics.
    
    Args:
        lr_image: Low-resolution input
        sr_image: Super-resolved output
        hr_image: High-resolution ground truth
        metric_values: Dictionary of computed metrics
        save_path: Path to save the figure
        
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Main images
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(lr_image, cmap='hot', interpolation='bilinear')
    ax1.set_title('Low Resolution Input')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(sr_image, cmap='hot', interpolation='bilinear')
    ax2.set_title('Super-Resolved Output')
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(hr_image, cmap='hot', interpolation='bilinear')
    ax3.set_title('High Resolution Ground Truth')
    ax3.axis('off')
    
    # Zoomed regions (center crop)
    h, w = lr_image.shape
    crop_size = min(h, w) // 4
    center_h, center_w = h // 2, w // 2
    
    crop_slice = (
        slice(center_h - crop_size, center_h + crop_size),
        slice(center_w - crop_size, center_w + crop_size)
    )
    
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(lr_image[crop_slice], cmap='hot', interpolation='bilinear')
    ax4.set_title('LR Zoom')
    ax4.axis('off')
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(sr_image[crop_slice], cmap='hot', interpolation='bilinear')
    ax5.set_title('SR Zoom')
    ax5.axis('off')
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(hr_image[crop_slice], cmap='hot', interpolation='bilinear')
    ax6.set_title('HR Zoom')
    ax6.axis('off')
    
    # Add metrics text
    metrics_text = "Metrics:\n"
    for key, value in metric_values.items():
        if isinstance(value, float):
            metrics_text += f"{key}: {value:.3f}\n"
            
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def batch_inference(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   device: torch.device,
                   save_dir: Optional[str] = None) -> Dict[str, List]:
    """
    Run inference on a batch of data.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        save_dir: Directory to save results
        
    Returns:
        Dictionary of results
    """
    model.eval()
    results = {
        'predictions': [],
        'targets': [],
        'inputs': [],
        'scene_ids': []
    }
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            thermal_lr = batch['thermal_lr'].to(device)
            optical = batch['optical'].to(device)
            
            # Run inference
            thermal_sr, emissivity, _ = model(thermal_lr, optical)
            
            # Store results
            results['predictions'].append(thermal_sr.cpu())
            results['inputs'].append(thermal_lr.cpu())
            
            if 'thermal_hr' in batch:
                results['targets'].append(batch['thermal_hr'])
                
            if 'scene_id' in batch:
                results['scene_ids'].extend(batch['scene_id'])
                
            # Save individual results if requested
            if save_dir:
                for i in range(thermal_sr.shape[0]):
                    scene_id = batch['scene_id'][i] if 'scene_id' in batch else f'batch{batch_idx}_sample{i}'
                    
                    # Save as numpy arrays
                    np.save(save_path / f'{scene_id}_sr.npy', thermal_sr[i].cpu().numpy())
                    np.save(save_path / f'{scene_id}_lr.npy', thermal_lr[i].cpu().numpy())
                    np.save(save_path / f'{scene_id}_emissivity.npy', emissivity[i].cpu().numpy())
                    
    # Concatenate results
    results['predictions'] = torch.cat(results['predictions'], dim=0)
    results['inputs'] = torch.cat(results['inputs'], dim=0)
    
    if results['targets']:
        results['targets'] = torch.cat(results['targets'], dim=0)
        
    return results
