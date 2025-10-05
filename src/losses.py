"""
Loss functions for physics-informed thermal super-resolution.

Combines:
- Reconstruction losses (L1, L2)
- Perceptual losses
- Physics consistency losses
- Adversarial losses
- Cross-spectral consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision.models as models
except ImportError:
    # Fallback if torchvision is not available
    models = None
from typing import Dict, List, Optional, Tuple

from .physics_utils import (
    PhysicsConsistencyLoss, ThermalPSF,
    brightness_temp_to_radiance, radiance_to_brightness_temp
)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.
    Adapted for thermal imagery by using early layers.
    """
    
    def __init__(self, feature_layers: List[str] = None):
        super().__init__()
        
        if feature_layers is None:
            feature_layers = ['relu1_2', 'relu2_2', 'relu3_3']
            
        # Check if torchvision is available
        if models is None:
            # Simple fallback - use conv layers for feature extraction
            self.feature_extractors = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
            ])
            self.layer_weights = [1.0]
        else:
            # Load pre-trained VGG19
            vgg = models.vgg19(pretrained=True).features
            
            # Extract relevant layers
            self.feature_extractors = nn.ModuleList()
            self.layer_weights = []
            
            layer_map = {
                'relu1_2': 4,
                'relu2_2': 9,
                'relu3_3': 18,
                'relu4_3': 27,
                'relu5_3': 36
            }
            
            for layer_name in feature_layers:
                if layer_name in layer_map:
                    layer_idx = layer_map[layer_name]
                    extractor = nn.Sequential(*list(vgg.children())[:layer_idx + 1])
                    self.feature_extractors.append(extractor)
                    
                    # Weight deeper features less
                    if 'relu1' in layer_name:
                        self.layer_weights.append(1.0)
                    elif 'relu2' in layer_name:
                        self.layer_weights.append(0.5)
                    else:
                        self.layer_weights.append(0.25)
                    
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between predicted and target images.
        
        For thermal images, we need to adapt them to VGG's expected input:
        - Convert 2-channel thermal to 3-channel by duplicating
        - Normalize to ImageNet statistics
        """
        # Adapt thermal images to VGG input
        pred_adapted = self._adapt_thermal_to_vgg(pred)
        target_adapted = self._adapt_thermal_to_vgg(target)
        
        loss = 0.0
        
        for extractor, weight in zip(self.feature_extractors, self.layer_weights):
            pred_features = extractor(pred_adapted)
            target_features = extractor(target_adapted)
            
            loss += weight * F.l1_loss(pred_features, target_features)
            
        return loss / len(self.feature_extractors)
    
    def _adapt_thermal_to_vgg(self, thermal: torch.Tensor) -> torch.Tensor:
        """Adapt 2-channel thermal to 3-channel RGB for VGG."""
        B, C, H, W = thermal.shape
        
        if C == 2:
            # Average the two thermal bands and replicate to 3 channels
            thermal_avg = thermal.mean(dim=1, keepdim=True)
            thermal_rgb = thermal_avg.repeat(1, 3, 1, 1)
        else:
            # Single channel - replicate to 3
            thermal_rgb = thermal.repeat(1, 3, 1, 1)
            
        # Normalize to ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(thermal.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(thermal.device)
        
        # Scale thermal to [0, 1] range first
        thermal_min = thermal_rgb.flatten(2).min(dim=2, keepdim=True)[0].unsqueeze(3)
        thermal_max = thermal_rgb.flatten(2).max(dim=2, keepdim=True)[0].unsqueeze(3)
        thermal_rgb = (thermal_rgb - thermal_min) / (thermal_max - thermal_min + 1e-8)
        
        # Apply ImageNet normalization
        thermal_normalized = (thermal_rgb - mean) / std
        
        return thermal_normalized


class SpectralConsistencyLoss(nn.Module):
    """
    Ensures consistency between thermal bands based on spectral properties.
    Band 10 and Band 11 should maintain physically plausible relationships.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral consistency loss.
        
        Args:
            pred: Predicted thermal bands (B, 2, H, W)
            target: Target thermal bands (B, 2, H, W)
        """
        if pred.shape[1] != 2:
            return torch.tensor(0.0, device=pred.device)
            
        # Band difference should be consistent
        pred_diff = pred[:, 0] - pred[:, 1]  # Band 10 - Band 11
        target_diff = target[:, 0] - target[:, 1]
        
        # L1 loss on band difference
        diff_loss = F.l1_loss(pred_diff, target_diff)
        
        # Ratio consistency (avoid division by zero)
        pred_ratio = pred[:, 0] / (pred[:, 1] + 1e-8)
        target_ratio = target[:, 0] / (target[:, 1] + 1e-8)
        
        # Clamp ratios to reasonable range
        pred_ratio = torch.clamp(pred_ratio, 0.8, 1.2)
        target_ratio = torch.clamp(target_ratio, 0.8, 1.2)
        
        ratio_loss = F.l1_loss(pred_ratio, target_ratio)
        
        return diff_loss + 0.1 * ratio_loss


class EdgePreservationLoss(nn.Module):
    """
    Preserves edges in super-resolved thermal images.
    Uses Sobel operators to compute gradients.
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                optical_edges: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute edge preservation loss.
        
        Args:
            pred: Predicted thermal image
            target: Target thermal image
            optical_edges: Optional optical edge map for guidance
        """
        # Compute gradients for each channel
        pred_edges = self._compute_edges(pred)
        target_edges = self._compute_edges(target)
        
        # Basic edge loss
        edge_loss = F.l1_loss(pred_edges, target_edges)
        
        # If optical edges provided, encourage alignment
        if optical_edges is not None:
            # Align optical edge map spatially and by dtype/device
            if optical_edges.shape[-2:] != pred.shape[-2:]:
                optical_edges = F.interpolate(
                    optical_edges, size=pred.shape[-2:], mode='bilinear', align_corners=False
                )
            optical_edges = optical_edges.to(pred_edges.dtype).to(pred_edges.device)

            # Normalize edge magnitudes
            pred_edges_norm = pred_edges / (pred_edges.max() + 1e-8)
            optical_edges_norm = optical_edges / (optical_edges.max() + 1e-8)
            
            # Correlation loss - edges should align
            correlation = (pred_edges_norm * optical_edges_norm).mean()
            edge_loss = edge_loss - 0.1 * correlation  # Negative because we want to maximize correlation
            
        return edge_loss
    
    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel operators."""
        B, C, H, W = x.shape
        
        # Reshape for convolution
        x_reshaped = x.view(B * C, 1, H, W)
        
        # Apply Sobel filters
        grad_x = F.conv2d(x_reshaped, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_reshaped, self.sobel_y, padding=1)
        
        # Compute magnitude
        edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Reshape back
        edges = edges.view(B, C, H, W)
        
        return edges


class ThermalSRLoss(nn.Module):
    """
    Combined loss function for thermal super-resolution.
    Integrates all loss components with appropriate weighting.
    """
    
    def __init__(
        self,
        lambda_pixel: float = 1.0,
        lambda_perceptual: float = 0.1,
        lambda_physics: float = 1.0,
        lambda_spectral: float = 0.5,
        lambda_edge: float = 0.1,
        lambda_adversarial: float = 0.01,
        use_l1: bool = True
    ):
        super().__init__()
        
        # Loss weights
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_physics = lambda_physics
        self.lambda_spectral = lambda_spectral
        self.lambda_edge = lambda_edge
        self.lambda_adversarial = lambda_adversarial
        
        # Loss modules
        self.pixel_loss = nn.L1Loss() if use_l1 else nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.physics_loss = PhysicsConsistencyLoss()
        self.spectral_loss = SpectralConsistencyLoss()
        self.edge_loss = EdgePreservationLoss()
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lr_input: torch.Tensor,
        emissivity: torch.Tensor,
        optical_features: Optional[torch.Tensor] = None,
        discriminator_pred: Optional[List[torch.Tensor]] = None,
        training_phase: str = 'initial'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss and individual components.
        
        Args:
            pred: Predicted super-resolved thermal image
            target: Ground truth high-resolution thermal image
            lr_input: Low-resolution input thermal image
            emissivity: Estimated emissivity map
            optical_features: Optional optical features for guidance
            discriminator_pred: Optional discriminator predictions for adversarial loss
            training_phase: Current training phase ('initial', 'refinement', 'adversarial')
            
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        
        # Pixel-wise reconstruction loss
        loss_dict['pixel'] = self.pixel_loss(pred, target)
        
        # Perceptual loss (disabled in initial training for stability)
        if training_phase != 'initial':
            loss_dict['perceptual'] = self.perceptual_loss(pred, target)
        else:
            loss_dict['perceptual'] = torch.tensor(0.0, device=pred.device)
            
        # Physics consistency loss
        physics_losses = self.physics_loss(pred, lr_input, emissivity, optical_features)
        loss_dict.update({f'physics_{k}': v for k, v in physics_losses.items()})
        loss_dict['physics_total'] = sum(physics_losses.values())
        
        # Spectral consistency loss
        loss_dict['spectral'] = self.spectral_loss(pred, target)
        
        # Edge preservation loss
        if optical_features is not None:
            # Extract edges from optical features
            optical_edges = self.edge_loss._compute_edges(optical_features.mean(dim=1, keepdim=True))
        else:
            optical_edges = None
        loss_dict['edge'] = self.edge_loss(pred, target, optical_edges)
        
        # Adversarial loss (only in adversarial training phase)
        if discriminator_pred is not None and training_phase == 'adversarial':
            # Multi-scale adversarial loss
            adv_loss = 0.0
            for pred_scale in discriminator_pred:
                adv_loss += -pred_scale.mean()  # Negative because we want to fool discriminator
            loss_dict['adversarial'] = adv_loss / len(discriminator_pred)
        else:
            loss_dict['adversarial'] = torch.tensor(0.0, device=pred.device)
            
        # Compute total loss with phase-dependent weighting
        if training_phase == 'initial':
            # Focus on pixel and physics losses
            total_loss = (
                self.lambda_pixel * loss_dict['pixel'] +
                self.lambda_physics * loss_dict['physics_total'] +
                self.lambda_spectral * loss_dict['spectral']
            )
        elif training_phase == 'refinement':
            # Add perceptual and edge losses
            total_loss = (
                self.lambda_pixel * loss_dict['pixel'] +
                self.lambda_perceptual * loss_dict['perceptual'] +
                self.lambda_physics * loss_dict['physics_total'] +
                self.lambda_spectral * loss_dict['spectral'] +
                self.lambda_edge * loss_dict['edge']
            )
        else:  # adversarial
            # Full loss with adversarial component
            total_loss = (
                self.lambda_pixel * loss_dict['pixel'] +
                self.lambda_perceptual * loss_dict['perceptual'] +
                self.lambda_physics * loss_dict['physics_total'] +
                self.lambda_spectral * loss_dict['spectral'] +
                self.lambda_edge * loss_dict['edge'] +
                self.lambda_adversarial * loss_dict['adversarial']
            )
            
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


class DiscriminatorLoss(nn.Module):
    """
    Loss function for the discriminator in GAN training.
    """
    
    def __init__(self, loss_type: str = 'lsgan'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'lsgan':
            # Least squares GAN loss
            self.criterion = nn.MSELoss()
        elif loss_type == 'vanilla':
            # Vanilla GAN loss
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
    def forward(
        self,
        real_pred: List[torch.Tensor],
        fake_pred: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute discriminator loss.
        
        Args:
            real_pred: List of discriminator predictions for real images
            fake_pred: List of discriminator predictions for fake images
            
        Returns:
            total_loss: Total discriminator loss
            loss_dict: Dictionary with loss components
        """
        loss_dict = {}
        
        # Compute loss for each scale
        real_loss = 0.0
        fake_loss = 0.0
        
        for i, (real, fake) in enumerate(zip(real_pred, fake_pred)):
            if self.loss_type == 'lsgan':
                # Real images should have output 1
                real_target = torch.ones_like(real)
                # Fake images should have output 0
                fake_target = torch.zeros_like(fake)
            else:
                # For vanilla GAN
                real_target = torch.ones_like(real)
                fake_target = torch.zeros_like(fake)
                
            real_loss_scale = self.criterion(real, real_target)
            fake_loss_scale = self.criterion(fake, fake_target)
            
            real_loss += real_loss_scale
            fake_loss += fake_loss_scale
            
            loss_dict[f'real_scale_{i}'] = real_loss_scale
            loss_dict[f'fake_scale_{i}'] = fake_loss_scale
            
        # Average over scales
        real_loss = real_loss / len(real_pred)
        fake_loss = fake_loss / len(fake_pred)
        
        total_loss = real_loss + fake_loss
        
        loss_dict['real'] = real_loss
        loss_dict['fake'] = fake_loss
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict
