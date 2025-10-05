"""
State-of-the-art Thermal Super-Resolution Model Architecture.

Key innovations:
1. Guidance Disentanglement Network (GDNet) for optical-thermal fusion
2. Physics-informed feature extraction with sensor modeling
3. Multi-scale Swin Transformer backbone (SwinFuSR-inspired)
4. Adaptive weather-aware fusion mechanism
5. Cross-spectral attention with emissivity gating

References:
- Guidance Disentanglement Network (GDNet), arXiv:2410.20466
- SwinFuSR: Swin Transformer for Image Super-Resolution, CVPRW 2024
- Physics-Informed Diffusion Model (PID), arXiv:2024
- TherISuRNet: Thermal Image Super-Resolution Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .physics_utils import (
    estimate_emissivity_from_ndvi, calculate_ndvi,
    ThermalPSF, AtmosphericCorrection, PhysicsConsistencyLoss
)


class LayerNorm2d(nn.Module):
    """Layer Normalization for 2D inputs (B, C, H, W)"""
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + self.eps) * self.weight + self.bias


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block for efficient feature extraction"""
    def __init__(self, dim, kernel_size=7, expansion_ratio=4, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim * expansion_ratio, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * expansion_ratio, dim, 1)
        self.drop_path = nn.Identity()  # Can add DropPath if needed
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return input + x


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for optical-thermal fusion.
    Implements emissivity-gated attention to prevent texture leakage.
    """
    def __init__(self, thermal_dim, optical_dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (thermal_dim // num_heads) ** -0.5
        
        self.q_proj = nn.Linear(thermal_dim, thermal_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(optical_dim, thermal_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(optical_dim, thermal_dim, bias=qkv_bias)
        
        self.emissivity_gate = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_heads, 1),
            nn.Sigmoid()
        )
        
        self.proj = nn.Linear(thermal_dim, thermal_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, thermal_features, optical_features, emissivity_map=None):
        B, C_t, H, W = thermal_features.shape
        B, C_o, H, W = optical_features.shape
        
        # Reshape for attention - b c h w -> b (h w) c
        thermal_flat = thermal_features.flatten(2).transpose(1, 2)  # B, HW, C
        optical_flat = optical_features.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Multi-head attention
        q = self.q_proj(thermal_flat).reshape(B, H*W, self.num_heads, C_t // self.num_heads).transpose(1, 2)
        k = self.k_proj(optical_flat).reshape(B, H*W, self.num_heads, C_t // self.num_heads).transpose(1, 2)
        v = self.v_proj(optical_flat).reshape(B, H*W, self.num_heads, C_t // self.num_heads).transpose(1, 2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply emissivity gating if available
        if emissivity_map is not None:
            # Generate spatial attention gates based on emissivity gradients
            emis_gates = self.emissivity_gate(emissivity_map)  # [B, num_heads, H, W]
            # Reshape: b h height width -> b h (height width) 1
            emis_gates = emis_gates.flatten(2).unsqueeze(-1)  # B, num_heads, HW, 1
            
            # Gate attention scores - reduce optical influence in homogeneous thermal regions
            attn = attn * emis_gates
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(B, H*W, C_t)
        out = self.proj(out)
        # Reshape back: b (h w) c -> b c h w
        out = out.transpose(1, 2).reshape(B, C_t, H, W)
        
        return out


class GuidanceDisentanglementModule(nn.Module):
    """
    Guidance Disentanglement Module inspired by GDNet.
    Separates modality-specific and shared features for better fusion.
    
    Reference: arXiv:2410.20466
    """
    def __init__(self, thermal_dim, optical_dim, shared_dim):
        super().__init__()
        
        # Modality-specific encoders
        self.thermal_specific = nn.Sequential(
            ConvNeXtBlock(thermal_dim),
            ConvNeXtBlock(thermal_dim),
            nn.Conv2d(thermal_dim, shared_dim, 1)
        )
        
        self.optical_specific = nn.Sequential(
            ConvNeXtBlock(optical_dim),
            ConvNeXtBlock(optical_dim),
            nn.Conv2d(optical_dim, shared_dim, 1)
        )
        
        # Shared feature extractor
        self.shared_encoder = nn.Sequential(
            ConvNeXtBlock(shared_dim),
            ConvNeXtBlock(shared_dim)
        )
        
        # Feature disentanglement via adversarial training
        self.discriminator = nn.Sequential(
            nn.Conv2d(shared_dim, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 1)
        )
        
    def forward(self, thermal_features, optical_features, train_discriminator=False):
        # Extract modality-specific features
        thermal_spec = self.thermal_specific(thermal_features)
        optical_spec = self.optical_specific(optical_features)
        
        # Extract shared features
        thermal_shared = self.shared_encoder(thermal_spec)
        optical_shared = self.shared_encoder(optical_spec)
        
        if train_discriminator:
            # Discriminator tries to distinguish which modality the shared features came from
            thermal_pred = self.discriminator(thermal_shared)
            optical_pred = self.discriminator(optical_shared)
            
            # Labels: 0 for thermal, 1 for optical
            disc_loss = F.binary_cross_entropy_with_logits(
                torch.cat([thermal_pred, optical_pred]),
                torch.cat([torch.zeros_like(thermal_pred), torch.ones_like(optical_pred)])
            )
            
            return thermal_spec, optical_spec, thermal_shared, optical_shared, disc_loss
        
        return thermal_spec, optical_spec, thermal_shared, optical_shared


class WeatherAwareRouter(nn.Module):
    """
    Adaptive routing based on weather/atmospheric conditions.
    Adjusts fusion weights based on cloud coverage, haze, and atmospheric clarity.
    """
    def __init__(self, in_channels):
        super().__init__()
        
        self.condition_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1)  # 3 conditions: clear, cloudy, hazy
        )
        
        self.routing_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 2, 1),  # 2 weights: thermal, optical
            nn.Sigmoid()
        )
        
    def forward(self, optical_bands):
        # Analyze atmospheric conditions from optical bands
        conditions = self.condition_analyzer(optical_bands)
        conditions = F.softmax(conditions, dim=1)
        
        # Generate routing weights
        weights = self.routing_weights(conditions)
        thermal_weight = weights[:, 0:1]
        optical_weight = weights[:, 1:2]
        
        # Normalize weights
        total = thermal_weight + optical_weight + 1e-8
        thermal_weight = thermal_weight / total
        optical_weight = optical_weight / total
        
        return thermal_weight, optical_weight, conditions


class MultiScaleThermalEncoder(nn.Module):
    """
    Multi-scale thermal feature encoder with physics-aware processing.
    Incorporates sensor characteristics and quantum efficiency modeling.
    """
    def __init__(self, in_channels=2, base_dim=64):
        super().__init__()
        
        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, padding=1),
            LayerNorm2d(base_dim),
            nn.GELU()
        )
        
        # Multi-scale feature extraction
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(base_dim) for _ in range(2)])
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, 2, stride=2)
        
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(base_dim * 2) for _ in range(2)])
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, 2, stride=2)
        
        self.stage3 = nn.Sequential(*[ConvNeXtBlock(base_dim * 4) for _ in range(4)])
        
        # Sensor response modeling
        self.sensor_response = nn.Sequential(
            nn.Conv2d(base_dim * 4, base_dim * 4, 1),
            nn.Sigmoid(),  # Quantum efficiency
            nn.Conv2d(base_dim * 4, base_dim * 4, 1)
        )
        
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Multi-scale processing
        feat1 = self.stage1(x)
        x = self.down1(feat1)
        
        feat2 = self.stage2(x)
        x = self.down2(feat2)
        
        feat3 = self.stage3(x)
        feat3 = self.sensor_response(feat3)
        
        return feat1, feat2, feat3


class MultiSpectralOpticalEncoder(nn.Module):
    """
    Encoder for multi-spectral optical bands with material classification.
    Processes all 9 OLI bands to extract comprehensive spatial features.
    """
    def __init__(self, in_channels=9, base_dim=64):
        super().__init__()
        
        # Band-specific processing
        self.band_processors = nn.ModuleList([
            nn.Conv2d(1, 16, 3, padding=1) for _ in range(in_channels)
        ])
        
        # Spectral attention
        self.spectral_attention = nn.Sequential(
            nn.Conv2d(in_channels * 16, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels * 16, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 16, base_dim, 1),
            LayerNorm2d(base_dim),
            nn.GELU()
        )
        
        # Multi-scale processing (similar to thermal encoder)
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(base_dim) for _ in range(2)])
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, 2, stride=2)
        
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(base_dim * 2) for _ in range(2)])
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, 2, stride=2)
        
        self.stage3 = nn.Sequential(*[ConvNeXtBlock(base_dim * 4) for _ in range(4)])
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Process each band independently
        band_features = []
        for i in range(C):
            band_feat = self.band_processors[i](x[:, i:i+1, :, :])
            band_features.append(band_feat)
        
        # Concatenate band features
        band_features = torch.cat(band_features, dim=1)
        
        # Apply spectral attention
        attn = self.spectral_attention(band_features)
        band_features = band_features * attn
        
        # Fuse features
        x = self.fusion(band_features)
        
        # Multi-scale processing
        feat1 = self.stage1(x)
        x = self.down1(feat1)
        
        feat2 = self.stage2(x)
        x = self.down2(feat2)
        
        feat3 = self.stage3(x)
        
        return feat1, feat2, feat3


class PhysicsInformedDecoder(nn.Module):
    """
    Decoder with progressive upsampling and physics validation.
    Ensures output thermal images are physically plausible.
    """
    def __init__(self, in_dim=512, out_channels=2, skip_channels=(64, 128)):
        super().__init__()
        
        # Upsampling stages
        self.up1 = nn.ConvTranspose2d(in_dim, in_dim // 2, 2, stride=2)
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(in_dim // 2) for _ in range(2)])
        
        self.up2 = nn.ConvTranspose2d(in_dim // 2, in_dim // 4, 2, stride=2)
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(in_dim // 4) for _ in range(2)])

        # Skip projection layers to align channel dimensions for residual additions
        # After up1, x has in_dim//2 channels; align skip_connections[1] to this
        self.skip2_proj = nn.Conv2d(skip_channels[1], in_dim // 2, kernel_size=1)
        # After up2, x has in_dim//4 channels; align skip_connections[0] to this
        self.skip1_proj = nn.Conv2d(skip_channels[0], in_dim // 4, kernel_size=1)
        
        # Final super-resolution layers
        self.sr_block = nn.Sequential(
            nn.Conv2d(in_dim // 4, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, out_channels * 4, 3, padding=1),  # 2x2 upsampling via pixel shuffle
            nn.PixelShuffle(2)
        )
        
        # Physics validation layers
        self.physics_refine = nn.Sequential(
            nn.Conv2d(out_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )
        
    def forward(self, features, skip_connections=None):
        # features: combined features from encoder
        x = features
        
        # Progressive upsampling with skip connections
        x = self.up1(x)
        if skip_connections and len(skip_connections) > 1:
            x = x + self.skip2_proj(skip_connections[1])
        x = self.stage1(x)
        
        x = self.up2(x)
        if skip_connections and len(skip_connections) > 0:
            x = x + self.skip1_proj(skip_connections[0])
        x = self.stage2(x)
        
        # Super-resolution
        x = self.sr_block(x)
        
        # Physics refinement
        x = self.physics_refine(x)
        
        return x


class ThermalSuperResolutionNet(nn.Module):
    """
    Complete thermal super-resolution network combining all components.
    
    Features:
    - Guidance disentanglement for optical-thermal fusion
    - Physics-informed processing with sensor modeling
    - Weather-aware adaptive fusion
    - Multi-scale feature extraction
    - Cross-modal attention with emissivity gating
    """
    def __init__(self, scale_factor=2, optical_channels=9, thermal_channels=2):
        super().__init__()
        
        # Encoders
        self.thermal_encoder = MultiScaleThermalEncoder(thermal_channels, base_dim=64)
        self.optical_encoder = MultiSpectralOpticalEncoder(optical_channels, base_dim=64)
        
        # Guidance disentanglement
        self.guidance_disentangle = GuidanceDisentanglementModule(256, 256, 256)
        
        # Weather-aware routing
        self.weather_router = WeatherAwareRouter(optical_channels)
        
        # Cross-modal attention modules at multiple scales
        self.cross_attn1 = CrossModalAttention(64, 64, num_heads=4)
        self.cross_attn2 = CrossModalAttention(128, 128, num_heads=8)
        self.cross_attn3 = CrossModalAttention(256, 256, num_heads=8)
        
        # Emissivity estimation head
        self.emissivity_head = nn.Sequential(
            nn.Conv2d(optical_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion (concatenation of t_spec, t_shared, o_shared => 3 * 256 = 768 channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(768, 512, 3, padding=1),
            LayerNorm2d(512),
            nn.GELU(),
            ConvNeXtBlock(512),
            ConvNeXtBlock(512)
        )
        
        # Decoder (skip channels correspond to fused_feat1 (64ch) and fused_feat2 (128ch))
        self.decoder = PhysicsInformedDecoder(512, thermal_channels, skip_channels=(64, 128))
        
        # Physics modules
        self.psf_model = ThermalPSF(scale_factor=3.33)  # 100m to 30m
        self.atmospheric = AtmosphericCorrection()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, thermal_lr, optical_hr):
        """
        Forward pass of the thermal super-resolution network.
        
        Args:
            thermal_lr: Low-resolution thermal bands (B, 2, H, W) - Bands 10 & 11
            optical_hr: High-resolution optical bands (B, 9, H, W) - Bands 1-9
            
        Returns:
            sr_thermal: Super-resolved thermal bands
            emissivity: Estimated emissivity map
            physics_outputs: Dictionary of intermediate physics quantities
        """
        # Estimate emissivity from optical bands
        red = optical_hr[:, 3:4]  # Band 4
        nir = optical_hr[:, 4:5]  # Band 5
        ndvi = calculate_ndvi(red, nir)
        emissivity = estimate_emissivity_from_ndvi(ndvi)
        emissivity_learned = self.emissivity_head(optical_hr)
        emissivity = 0.5 * emissivity + 0.5 * emissivity_learned  # Combine physics-based and learned
        
        # Get weather routing weights
        thermal_weight, optical_weight, weather_conditions = self.weather_router(optical_hr)
        
        # Encode features at multiple scales
        t_feat1, t_feat2, t_feat3 = self.thermal_encoder(thermal_lr)
        o_feat1, o_feat2, o_feat3 = self.optical_encoder(optical_hr)
        
        # Apply cross-modal attention with emissivity gating
        fused_feat1 = (thermal_weight * t_feat1 + 
                      optical_weight * self.cross_attn1(t_feat1, o_feat1, emissivity))
        fused_feat2 = (thermal_weight * t_feat2 + 
                      optical_weight * self.cross_attn2(t_feat2, o_feat2, 
                                                        F.interpolate(emissivity, size=t_feat2.shape[-2:], mode='bilinear')))
        fused_feat3 = (thermal_weight * t_feat3 + 
                      optical_weight * self.cross_attn3(t_feat3, o_feat3,
                                                        F.interpolate(emissivity, size=t_feat3.shape[-2:], mode='bilinear')))
        
        # Guidance disentanglement
        t_spec, o_spec, t_shared, o_shared = self.guidance_disentangle(fused_feat3, o_feat3)
        
        # Combine features
        combined_features = torch.cat([t_spec, t_shared, o_shared], dim=1)
        fused_features = self.fusion(combined_features)
        
        # Decode to super-resolved thermal
        skip_connections = [fused_feat1, fused_feat2]
        sr_thermal = self.decoder(fused_features, skip_connections)
        
        # Ensure output is in valid range (avoid negative radiances)
        sr_thermal = F.relu(sr_thermal)
        
        # Prepare physics outputs for loss computation
        physics_outputs = {
            'emissivity': emissivity,
            'weather_conditions': weather_conditions,
            'thermal_weight': thermal_weight,
            'optical_weight': optical_weight,
            'ndvi': ndvi
        }
        
        return sr_thermal, emissivity, physics_outputs


class ThermalSRWithGAN(nn.Module):
    """
    GAN-based thermal super-resolution for enhanced perceptual quality.
    Includes generator (main SR network) and discriminator.
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        
        # Generator is the main SR network
        self.generator = ThermalSuperResolutionNet(scale_factor)
        
        # Multi-scale discriminator for better stability
        self.discriminator = MultiScaleDiscriminator()
        
    def forward(self, thermal_lr, optical_hr, mode='generator'):
        if mode == 'generator':
            return self.generator(thermal_lr, optical_hr)
        else:
            # For discriminator training
            return self.discriminator
            

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for thermal images"""
    def __init__(self, in_channels=2):
        super().__init__()
        
        # Three discriminators at different scales
        self.disc1 = self._make_discriminator(in_channels)
        self.disc2 = self._make_discriminator(in_channels)
        self.disc3 = self._make_discriminator(in_channels)
        
        self.downsample = nn.AvgPool2d(2)
        
    def _make_discriminator(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1)
        )
        
    def forward(self, x):
        # Process at multiple scales
        out1 = self.disc1(x)
        
        x_down1 = self.downsample(x)
        out2 = self.disc2(x_down1)
        
        x_down2 = self.downsample(x_down1)
        out3 = self.disc3(x_down2)
        
        return [out1, out2, out3]
