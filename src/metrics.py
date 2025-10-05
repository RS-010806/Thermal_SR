"""
Evaluation metrics for thermal super-resolution.

Includes:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- RMSE in Kelvin
- Edge preservation metrics
- Physics consistency metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
try:
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    compare_ssim = None
    compare_psnr = None


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image
        target: Ground truth image
        data_range: Maximum possible pixel value range
        
    Returns:
        PSNR in dB
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, 
                 data_range: float = 1.0, multichannel: bool = True) -> float:
    """
    Compute Structural Similarity Index.
    
    Args:
        pred: Predicted image (B, C, H, W)
        target: Ground truth image (B, C, H, W)
        data_range: Maximum possible pixel value range
        multichannel: Whether to treat as multichannel image
        
    Returns:
        SSIM value
    """
    if not HAS_SKIMAGE:
        # Simple fallback SSIM implementation
        # Based on the formula: SSIM = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2) / 
        #                              ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        mu_x = pred.mean(dim=[2, 3], keepdim=True)
        mu_y = target.mean(dim=[2, 3], keepdim=True)
        
        sigma_x = pred.var(dim=[2, 3], keepdim=True)
        sigma_y = target.var(dim=[2, 3], keepdim=True)
        sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=[2, 3], keepdim=True)
        
        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
        
        return ssim.mean().item()
    
    # Convert to numpy
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Compute SSIM for each sample in batch
    ssim_values = []
    
    for i in range(pred_np.shape[0]):
        # Transpose to HWC format for skimage
        pred_sample = np.transpose(pred_np[i], (1, 2, 0))
        target_sample = np.transpose(target_np[i], (1, 2, 0))
        
        # Compute SSIM
        if pred_sample.shape[2] == 1:
            # Single channel
            ssim = compare_ssim(
                pred_sample[:, :, 0],
                target_sample[:, :, 0],
                data_range=data_range
            )
        else:
            # Multi-channel
            ssim = compare_ssim(
                pred_sample,
                target_sample,
                data_range=data_range,
                channel_axis=2
            )
            
        ssim_values.append(ssim)
        
    return np.mean(ssim_values)


def compute_rmse_kelvin(pred: torch.Tensor, target: torch.Tensor,
                       thermal_mean: np.ndarray, thermal_std: np.ndarray) -> float:
    """
    Compute RMSE in Kelvin after denormalization.
    
    Args:
        pred: Predicted normalized thermal image
        target: Ground truth normalized thermal image
        thermal_mean: Mean values used for normalization
        thermal_std: Std values used for normalization
        
    Returns:
        RMSE in Kelvin
    """
    # Denormalize to get back to brightness temperature
    thermal_mean_t = torch.tensor(thermal_mean).view(1, -1, 1, 1).to(pred.device)
    thermal_std_t = torch.tensor(thermal_std).view(1, -1, 1, 1).to(pred.device)
    
    pred_kelvin = pred * thermal_std_t + thermal_mean_t
    target_kelvin = target * thermal_std_t + thermal_mean_t
    
    # Compute RMSE
    mse = F.mse_loss(pred_kelvin, target_kelvin)
    rmse = torch.sqrt(mse)
    
    return rmse.item()


def compute_edge_preservation_index(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Edge Preservation Index (EPI).
    Measures how well edges are preserved in the super-resolved image.
    
    Args:
        pred: Predicted image
        target: Ground truth image
        
    Returns:
        EPI value (higher is better, 1.0 is perfect)
    """
    # Sobel filters for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    
    # Compute edges for each channel
    epi_values = []
    
    for c in range(pred.shape[1]):
        # Extract channel
        pred_c = pred[:, c:c+1, :, :]
        target_c = target[:, c:c+1, :, :]
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred_c, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_c, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_c, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_c, sobel_y, padding=1)
        
        # Compute edge magnitudes
        pred_edges = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_edges = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        # Normalize edge maps
        pred_edges_norm = pred_edges / (pred_edges.max() + 1e-8)
        target_edges_norm = target_edges / (target_edges.max() + 1e-8)
        
        # Compute correlation
        mean_pred = pred_edges_norm.mean()
        mean_target = target_edges_norm.mean()
        
        cov = ((pred_edges_norm - mean_pred) * (target_edges_norm - mean_target)).mean()
        std_pred = torch.sqrt(((pred_edges_norm - mean_pred)**2).mean() + 1e-8)
        std_target = torch.sqrt(((target_edges_norm - mean_target)**2).mean() + 1e-8)
        
        correlation = cov / (std_pred * std_target)
        epi_values.append(correlation.item())
        
    return np.mean(epi_values)


def compute_spectral_consistency(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute spectral consistency between Band 10 and Band 11.
    
    Args:
        pred: Predicted thermal image with 2 bands
        target: Ground truth thermal image with 2 bands
        
    Returns:
        Spectral consistency error (lower is better)
    """
    if pred.shape[1] != 2:
        return 0.0
        
    # Compute band ratios
    pred_ratio = pred[:, 0] / (pred[:, 1] + 1e-8)
    target_ratio = target[:, 0] / (target[:, 1] + 1e-8)
    
    # Clamp to reasonable range
    pred_ratio = torch.clamp(pred_ratio, 0.8, 1.2)
    target_ratio = torch.clamp(target_ratio, 0.8, 1.2)
    
    # Compute error
    ratio_error = F.l1_loss(pred_ratio, target_ratio)
    
    return ratio_error.item()


def compute_sharpness_index(image: torch.Tensor) -> float:
    """
    Compute image sharpness using gradient magnitude.
    
    Args:
        image: Input image
        
    Returns:
        Sharpness index (higher means sharper)
    """
    # Compute gradients
    grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
    
    # Compute gradient magnitude
    grad_mag_x = torch.abs(grad_x).mean()
    grad_mag_y = torch.abs(grad_y).mean()
    
    sharpness = (grad_mag_x + grad_mag_y) / 2.0
    
    return sharpness.item()


def compute_no_reference_metrics(image: torch.Tensor) -> Dict[str, float]:
    """
    Compute no-reference image quality metrics.
    Useful for evaluating super-resolved images without ground truth.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary of no-reference metrics
    """
    metrics = {}
    
    # Sharpness
    metrics['sharpness'] = compute_sharpness_index(image)
    
    # Contrast (standard deviation)
    metrics['contrast'] = image.std().item()
    
    # Entropy (information content)
    # Normalize to [0, 1] for entropy calculation
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Compute histogram
    hist = torch.histc(image_norm.flatten(), bins=256, min=0, max=1)
    hist = hist / hist.sum()
    
    # Compute entropy
    entropy = -(hist * torch.log2(hist + 1e-8)).sum()
    metrics['entropy'] = entropy.item()
    
    return metrics


def compute_metrics(pred: torch.Tensor, target: torch.Tensor,
                   data_range: float = 1.0,
                   compute_kelvin: bool = True,
                   thermal_mean: Optional[np.ndarray] = None,
                   thermal_std: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        pred: Predicted image
        target: Ground truth image
        data_range: Maximum possible pixel value range
        compute_kelvin: Whether to compute RMSE in Kelvin
        thermal_mean: Mean values for denormalization
        thermal_std: Std values for denormalization
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['psnr'] = compute_psnr(pred, target, data_range)
    metrics['ssim'] = compute_ssim(pred, target, data_range)
    
    # Kelvin RMSE
    if compute_kelvin and thermal_mean is not None and thermal_std is not None:
        metrics['rmse_kelvin'] = compute_rmse_kelvin(pred, target, thermal_mean, thermal_std)
    else:
        metrics['rmse_kelvin'] = 0.0
        
    # Edge preservation
    metrics['epi'] = compute_edge_preservation_index(pred, target)
    
    # Spectral consistency
    metrics['spectral_consistency'] = compute_spectral_consistency(pred, target)
    
    # Sharpness comparison
    metrics['sharpness_pred'] = compute_sharpness_index(pred)
    metrics['sharpness_target'] = compute_sharpness_index(target)
    metrics['sharpness_ratio'] = metrics['sharpness_pred'] / (metrics['sharpness_target'] + 1e-8)
    
    # No-reference metrics for predicted image
    nr_metrics = compute_no_reference_metrics(pred)
    for key, value in nr_metrics.items():
        metrics[f'nr_{key}'] = value
        
    return metrics


class MetricTracker:
    """
    Track metrics over time for monitoring training progress.
    """
    
    def __init__(self, metrics_to_track: list = None):
        if metrics_to_track is None:
            metrics_to_track = ['psnr', 'ssim', 'rmse_kelvin', 'epi']
            
        self.metrics_to_track = metrics_to_track
        self.history = {metric: [] for metric in metrics_to_track}
        self.best_values = {metric: None for metric in metrics_to_track}
        self.best_epochs = {metric: 0 for metric in metrics_to_track}
        
    def update(self, metrics: Dict[str, float], epoch: int):
        """Update metric history and track best values."""
        for metric in self.metrics_to_track:
            if metric in metrics:
                value = metrics[metric]
                self.history[metric].append(value)
                
                # Update best value
                if self.best_values[metric] is None:
                    self.best_values[metric] = value
                    self.best_epochs[metric] = epoch
                else:
                    # Higher is better for PSNR, SSIM, EPI
                    # Lower is better for RMSE
                    if metric in ['rmse_kelvin', 'spectral_consistency']:
                        if value < self.best_values[metric]:
                            self.best_values[metric] = value
                            self.best_epochs[metric] = epoch
                    else:
                        if value > self.best_values[metric]:
                            self.best_values[metric] = value
                            self.best_epochs[metric] = epoch
                            
    def get_best(self) -> Dict[str, Tuple[float, int]]:
        """Get best values and corresponding epochs."""
        return {
            metric: (self.best_values[metric], self.best_epochs[metric])
            for metric in self.metrics_to_track
        }
        
    def get_latest(self) -> Dict[str, float]:
        """Get latest metric values."""
        return {
            metric: self.history[metric][-1] if self.history[metric] else None
            for metric in self.metrics_to_track
        }
