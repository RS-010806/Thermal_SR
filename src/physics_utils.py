"""
Physics utilities for thermal infrared super-resolution.
Based on Landsat 8 TIRS sensor characteristics and radiative transfer physics.

References:
- Qin et al., "A mono-window algorithm for retrieving land surface temperature"
- USGS Landsat 8 Data Users Handbook
- Planck's Law and Stefan-Boltzmann radiation physics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ThermalPhysicsConstants:
    """Landsat 8 TIRS calibration constants and physics parameters"""
    # Band 10 (10.60-11.19 μm) calibration constants
    K1_B10 = 774.8853  # W/(m²·sr·μm)
    K2_B10 = 1321.0789  # K
    
    # Band 11 (11.50-12.51 μm) calibration constants  
    K1_B11 = 480.8883
    K2_B11 = 1201.1442
    
    # Central wavelengths (μm)
    LAMBDA_B10 = 10.895
    LAMBDA_B11 = 12.005
    
    # Physical constants
    C1 = 1.19104e-16  # W·m²
    C2 = 1.43877e-2   # m·K
    SIGMA = 5.67e-8   # Stefan-Boltzmann constant W/(m²·K⁴)
    
    # Typical emissivity values
    EMIS_VEGETATION = 0.99
    EMIS_SOIL = 0.96
    EMIS_WATER = 0.995
    EMIS_URBAN = 0.94


def radiance_to_brightness_temp(radiance, band=10):
    """
    Convert TOA radiance to brightness temperature using Planck's law.
    
    Args:
        radiance: TOA radiance in W/(m²·sr·μm)
        band: TIRS band number (10 or 11)
    
    Returns:
        Brightness temperature in Kelvin
    """
    if band == 10:
        K1, K2 = ThermalPhysicsConstants.K1_B10, ThermalPhysicsConstants.K2_B10
    else:
        K1, K2 = ThermalPhysicsConstants.K1_B11, ThermalPhysicsConstants.K2_B11
    
    # Planck's law inversion
    return K2 / torch.log(K1 / radiance + 1.0)


def brightness_temp_to_radiance(temp, band=10):
    """
    Convert brightness temperature to radiance using Planck's law.
    
    Args:
        temp: Brightness temperature in Kelvin
        band: TIRS band number (10 or 11)
    
    Returns:
        Radiance in W/(m²·sr·μm)
    """
    if band == 10:
        K1, K2 = ThermalPhysicsConstants.K1_B10, ThermalPhysicsConstants.K2_B10
    else:
        K1, K2 = ThermalPhysicsConstants.K1_B11, ThermalPhysicsConstants.K2_B11
    
    # Planck's law
    return K1 / (torch.exp(K2 / temp) - 1.0)


def brightness_to_lst(brightness_temp, emissivity, band=10):
    """
    Convert brightness temperature to Land Surface Temperature (LST).
    
    Args:
        brightness_temp: Brightness temperature in Kelvin
        emissivity: Surface emissivity (0-1)
        band: TIRS band number (10 or 11)
    
    Returns:
        Land Surface Temperature in Kelvin
    """
    if band == 10:
        wavelength = ThermalPhysicsConstants.LAMBDA_B10 * 1e-6  # Convert to meters
    else:
        wavelength = ThermalPhysicsConstants.LAMBDA_B11 * 1e-6
    
    # Temperature correction factor
    rho = ThermalPhysicsConstants.C2
    
    # LST calculation using mono-window algorithm
    lst = brightness_temp / (1 + (wavelength * brightness_temp / rho) * torch.log(emissivity))
    
    return lst


def calculate_ndvi(red, nir):
    """
    Calculate Normalized Difference Vegetation Index.
    
    Args:
        red: Red band (Band 4) reflectance
        nir: NIR band (Band 5) reflectance
    
    Returns:
        NDVI values (-1 to 1)
    """
    eps = 1e-8
    return (nir - red) / (nir + red + eps)


def estimate_emissivity_from_ndvi(ndvi, ndvi_soil=0.2, ndvi_veg=0.5):
    """
    Estimate surface emissivity from NDVI using threshold method.
    
    Args:
        ndvi: NDVI values
        ndvi_soil: NDVI threshold for bare soil
        ndvi_veg: NDVI threshold for full vegetation
    
    Returns:
        Estimated emissivity
    
    Reference:
    - Sobrino et al., "Land surface emissivity retrieval from different VNIR and TIR sensors"
    """
    # Initialize emissivity
    emissivity = torch.ones_like(ndvi) * ThermalPhysicsConstants.EMIS_SOIL
    
    # Water pixels (NDVI < 0)
    water_mask = ndvi < 0
    emissivity[water_mask] = ThermalPhysicsConstants.EMIS_WATER
    
    # Bare soil (0 <= NDVI < ndvi_soil)
    soil_mask = (ndvi >= 0) & (ndvi < ndvi_soil)
    emissivity[soil_mask] = ThermalPhysicsConstants.EMIS_SOIL
    
    # Mixed pixels (ndvi_soil <= NDVI < ndvi_veg)
    mixed_mask = (ndvi >= ndvi_soil) & (ndvi < ndvi_veg)
    if mixed_mask.any():
        # Proportion of vegetation
        Pv = ((ndvi[mixed_mask] - ndvi_soil) / (ndvi_veg - ndvi_soil)) ** 2
        
        # Emissivity for mixed pixels
        emis_mixed = (ThermalPhysicsConstants.EMIS_VEGETATION * Pv + 
                      ThermalPhysicsConstants.EMIS_SOIL * (1 - Pv) + 
                      0.005)  # Cavity effect term
        emissivity[mixed_mask] = emis_mixed
    
    # Full vegetation (NDVI >= ndvi_veg)
    veg_mask = ndvi >= ndvi_veg
    emissivity[veg_mask] = ThermalPhysicsConstants.EMIS_VEGETATION
    
    return emissivity


class ThermalPSF(nn.Module):
    """
    Thermal Point Spread Function modeling for TIRS sensor.
    Models the spatial response of the thermal sensor including:
    - Optical blur
    - Detector integration
    - Atmospheric effects
    
    Reference:
    - Montanaro et al., "Landsat-8 Thermal Infrared Sensor (TIRS) Vicarious Radiometric Calibration"
    """
    
    def __init__(self, scale_factor=3.33):  # 100m to 30m is approximately 3.33x
        super().__init__()
        self.scale_factor = scale_factor
        
        # Create Gaussian PSF kernel
        # TIRS has approximately 100m IFOV with Gaussian-like response
        kernel_size = int(2 * scale_factor + 1)
        sigma = scale_factor / 3.0  # Approximate FWHM to sigma conversion
        
        # Generate 2D Gaussian kernel
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        self.register_buffer('psf_kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        """Apply PSF blur to high-resolution thermal image"""
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Apply PSF convolution
        blurred = F.conv2d(x.view(B*C, 1, H, W), self.psf_kernel, padding='same')
        blurred = blurred.view(B, C, H, W)
        
        # Downsample
        downsampled = F.interpolate(blurred, scale_factor=1/self.scale_factor, 
                                   mode='bilinear', align_corners=False)
        
        return downsampled
    
    def transpose(self, x):
        """Transpose operation for gradient computation"""
        # Upsample then apply PSF transpose (same as PSF for symmetric kernel)
        upsampled = F.interpolate(x, scale_factor=self.scale_factor, 
                                 mode='bilinear', align_corners=False)
        
        B, C, H, W = upsampled.shape
        blurred = F.conv2d(upsampled.view(B*C, 1, H, W), self.psf_kernel, padding='same')
        
        return blurred.view(B, C, H, W)


class AtmosphericCorrection(nn.Module):
    """
    Simplified atmospheric correction model for thermal bands.
    
    Based on radiative transfer equation:
    L_sensor = τ * ε * L_surface + L_upwelling + (1-ε) * L_downwelling * τ
    
    Where:
    - τ: atmospheric transmittance
    - ε: surface emissivity  
    - L_surface: surface-leaving radiance
    - L_upwelling: atmospheric upwelling radiance
    - L_downwelling: atmospheric downwelling radiance
    """
    
    def __init__(self):
        super().__init__()
        
        # Learnable atmospheric parameters (can be refined during training)
        self.tau = nn.Parameter(torch.tensor(0.85))  # Typical transmittance
        self.L_up = nn.Parameter(torch.tensor(1.5))  # Typical upwelling radiance
        self.L_down = nn.Parameter(torch.tensor(2.0))  # Typical downwelling radiance
        
    def forward(self, L_surface, emissivity):
        """
        Apply atmospheric effects to surface radiance.
        
        Args:
            L_surface: Surface-leaving radiance
            emissivity: Surface emissivity
            
        Returns:
            At-sensor radiance
        """
        L_sensor = (self.tau * emissivity * L_surface + 
                   self.L_up + 
                   (1 - emissivity) * self.L_down * self.tau)
        
        return L_sensor
    
    def inverse(self, L_sensor, emissivity):
        """
        Remove atmospheric effects from at-sensor radiance.
        
        Args:
            L_sensor: At-sensor radiance
            emissivity: Surface emissivity
            
        Returns:
            Surface-leaving radiance
        """
        L_surface = (L_sensor - self.L_up - (1 - emissivity) * self.L_down * self.tau) / (self.tau * emissivity + 1e-8)
        
        return L_surface


class PhysicsConsistencyLoss(nn.Module):
    """
    Physics-based consistency loss for thermal super-resolution.
    Ensures physical plausibility of super-resolved thermal imagery.
    """
    
    def __init__(self, psf_model=None):
        super().__init__()
        self.psf = psf_model or ThermalPSF()
        self.atmospheric = AtmosphericCorrection()
        
    def forward(self, sr_thermal, lr_thermal, emissivity, optical_features=None):
        """
        Compute physics consistency loss.
        
        Args:
            sr_thermal: Super-resolved thermal image (high-res)
            lr_thermal: Original low-res thermal image
            emissivity: Estimated emissivity map
            optical_features: Optional optical features for guidance
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # 1. Sensor consistency loss
        # Downsample SR image through PSF to native LR sampling
        sr_downsampled = self.psf(sr_thermal)
        # The provided lr_thermal is typically upsampled to HR grid for training.
        # Bring it to the same spatial resolution as sr_downsampled for a fair comparison.
        if lr_thermal.shape[-2:] != sr_downsampled.shape[-2:]:
            lr_matched = F.interpolate(
                lr_thermal, size=sr_downsampled.shape[-2:], mode='bilinear', align_corners=False
            )
        else:
            lr_matched = lr_thermal
        losses['sensor_consistency'] = F.l1_loss(sr_downsampled, lr_matched)
        
        # 2. Energy conservation loss
        # Total energy should be preserved (integral of radiance)
        sr_energy = sr_thermal.mean(dim=[2, 3])
        lr_energy = lr_thermal.mean(dim=[2, 3])
        scale_factor = (sr_thermal.shape[-1] / lr_thermal.shape[-1]) ** 2
        losses['energy_conservation'] = F.l1_loss(sr_energy, lr_energy * scale_factor)
        
        # 3. Cross-band consistency (if both Band 10 and 11 are available)
        if sr_thermal.shape[1] == 2:
            # Temperature difference between bands should be physically plausible
            temp_diff = sr_thermal[:, 0] - sr_thermal[:, 1]
            losses['cross_band'] = torch.abs(temp_diff).mean() * 0.1
            
        # 4. Smoothness constraint weighted by emissivity gradients
        # Thermal gradients should be smooth except at material boundaries
        if emissivity is not None:
            # Ensure emissivity matches SR spatial size
            if emissivity.shape[-2:] != sr_thermal.shape[-2:]:
                emissivity_hr = F.interpolate(
                    emissivity, size=sr_thermal.shape[-2:], mode='bilinear', align_corners=False
                )
            else:
                emissivity_hr = emissivity

            # Create float kernels on correct device/dtype
            kx_e = torch.tensor([[-1.0, 0.0, 1.0]], device=emissivity_hr.device, dtype=emissivity_hr.dtype).view(1, 1, 1, 3)
            ky_e = torch.tensor([[-1.0], [0.0], [1.0]], device=emissivity_hr.device, dtype=emissivity_hr.dtype).view(1, 1, 3, 1)

            emis_grad = torch.sqrt(
                F.conv2d(emissivity_hr, kx_e, padding='same')**2 +
                F.conv2d(emissivity_hr, ky_e, padding='same')**2
            )
            
            # Thermal gradients with channel-wise grouped convs
            kx_t = torch.tensor([[-1.0, 0.0, 1.0]], device=sr_thermal.device, dtype=sr_thermal.dtype).view(1, 1, 1, 3)
            ky_t = torch.tensor([[-1.0], [0.0], [1.0]], device=sr_thermal.device, dtype=sr_thermal.dtype).view(1, 1, 3, 1)
            kx_t = kx_t.repeat(sr_thermal.shape[1], 1, 1, 1)
            ky_t = ky_t.repeat(sr_thermal.shape[1], 1, 1, 1)

            thermal_grad_x = F.conv2d(sr_thermal, kx_t, groups=sr_thermal.shape[1], padding='same')
            thermal_grad_y = F.conv2d(sr_thermal, ky_t, groups=sr_thermal.shape[1], padding='same')
            thermal_grad = torch.sqrt(thermal_grad_x**2 + thermal_grad_y**2)
            
            # Weight smoothness by inverse of emissivity gradients
            weight = 1.0 / (1.0 + 10 * emis_grad)
            losses['weighted_smoothness'] = (thermal_grad * weight).mean()
        
        return losses


class SensorNoiseModel(nn.Module):
    """
    Model TIRS sensor noise characteristics.
    
    TIRS has NEΔT (Noise Equivalent Delta Temperature) of ~0.1K at 300K
    """
    
    def __init__(self, nedt=0.1):
        super().__init__()
        self.nedt = nedt
        
    def forward(self, x, training=True):
        """Add realistic sensor noise during training"""
        if not training:
            return x
            
        # Convert to temperature space for noise addition
        temp = radiance_to_brightness_temp(x)
        
        # Add Gaussian noise scaled by NEΔT
        noise = torch.randn_like(temp) * self.nedt
        noisy_temp = temp + noise
        
        # Convert back to radiance
        return brightness_temp_to_radiance(noisy_temp)
