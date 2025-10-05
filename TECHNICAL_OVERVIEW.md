## Optical-Guided Super-Resolution for Thermal IR Imagery — End-to-End Overview

### 1) Problem Statement
- Goal: Super-resolve Landsat-8 Thermal Infrared Sensor (TIRS) Bands 10 and 11 (originally 100 m) to 30 m using guidance from Optical Land Imager (OLI) bands, while preserving true temperature fidelity.
- Constraints: Outputs must be physically consistent (thermodynamically plausible), sharp, and free from optical-texture hallucinations; scalable to large regions.

### 2) Dataset Structure
- Source: SSL4EO Landsat benchmark (downloaded locally), each scene is an `all_bands.tif` with 11 channels aligned to 30 m.
  - Bands 1–9: OLI (Coastal, Blue, Green, Red, NIR, SWIR1, SWIR2, PAN-resampled, Cirrus)
  - Bands 10–11: TIRS (thermal IR; upsampled from 100 m to 30 m in the package)
- Path pattern we observed: `ssl4eo_l_oli_tirs_toa_benchmark/<id>/<LC08_...>/all_bands.tif`, size typically 264×264 px.

### 3) High-Level Solution
- Design a physics-aware, multi-branch deep model:
  - Thermal branch encodes TIRS bands (2 channels).
  - Optical branch encodes 9 OLI bands with spectral attention.
  - Emissivity is estimated (NDVI-based + learned head) and used to gate optical guidance to avoid texture leakage.
  - Weather-aware router modulates reliance on optical vs thermal features.
  - Cross-modal attention fuses features with emissivity gating.
  - A physics layer enforces sensor-consistency via PSF/downsampling and atmospheric modeling during training.

### 4) Core Novelties (feasible, research-grade)
- Emissivity-gated fusion: Uses NDVI-derived emissivity + learned emissivity head to gate optical guidance.
- Weather-aware routing: Lightweight CNN estimates scene conditions (clear/cloud/haze) and adjusts fusion weights.
- Physics consistency: Differentiable PSF/downsample and simple radiative transfer ensure SR predictions are sensor-consistent.
- Guidance disentanglement: Separate modality-specific and shared features to mitigate cross-modal leakage.

### 5) Code Layout (what each file does)
- `src/models.py`
  - MultiScaleThermalEncoder: 2-channel thermal encoder (ConvNeXt blocks; multi-scale).
  - MultiSpectralOpticalEncoder: 9-band optical encoder with per-band convs + spectral attention + multi-scale.
  - CrossModalAttention: attention from thermal to optical features, gated by emissivity.
  - GuidanceDisentanglementModule: splits modality-specific vs shared features.
  - PhysicsInformedDecoder: progressive upsampling with skip connections; channel-aligned via 1×1 convs.
  - ThermalSuperResolutionNet: wires all modules; outputs SR thermal (2 ch) + emissivity map.

- `src/physics_utils.py`
  - Radiometric utilities (brightness temperature/radiance; emissivity from NDVI).
  - ThermalPSF: approximated Gaussian PSF + bilinear downsample; used for sensor-consistency loss.
  - AtmosphericCorrection: simple learnable transmittance/up/down radiance (kept lightweight).
  - PhysicsConsistencyLoss: sums sensor-consistency, energy conservation, cross-band plausibility, and edge-weighted smoothness.

- `src/losses.py`
  - ThermalSRLoss: combines pixel, perceptual (disabled in initial phase), physics, spectral consistency, edge preservation, and adversarial losses (if enabled later).
  - EdgePreservationLoss: Sobel-gradient-based; aligns optional optical edge map spatially to SR.

- `src/data_loader.py`
  - ThermalSRDataset: reads `all_bands.tif`, extracts patches, normalization, optional LR simulation.
  - create_dataloaders: train/val/test loaders.

- `src/metrics.py`
  - PSNR, SSIM (with fallback), RMSE in Kelvin (after denormalization), edge preservation, spectral consistency, and no-reference metrics.

- `train.py`
  - Three-phase training: initial (reconstruction + physics), refinement (add perceptual, edge), adversarial (optional GAN).
  - Logging via TensorBoard; checkpointing; validation per epoch.

- `inference.py`
  - Loads checkpoint, runs model on one or many scenes, saves GeoTIFF outputs (SR Band 10/11, emissivity) and visualizations.

### 6) Model Architecture (step-by-step)
1) Inputs: thermal_lr (B×2×H×W), optical_hr (B×9×H×W)
2) Emissivity estimation:
   - Physics-based: NDVI from Red/NIR, then emissivity via thresholds.
   - Learned head: CNN on optical bands produces emissivity; combine 50/50.
3) Weather-aware router: predicts routing weights to balance thermal vs optical reliance.
4) Encoders:
   - Thermal encoder → features at 64, 128, 256 channels.
   - Optical encoder → features at 64, 128, 256 channels with spectral attention.
5) Cross-modal attention (per scale): emissivity-gated attention from thermal queries to optical keys/values.
6) Guidance disentanglement (high scale): produce thermal-specific, and shared features.
7) Fusion: concatenate [thermal-specific (256), shared-thermal (256), shared-optical (256)] → 768 ch, reduce to 512 ch and refine.
8) Decoder:
   - upsample 512→256, add skip (aligned 128→256), refine.
   - upsample 256→128, add skip (aligned 64→128), refine.
   - final SR head (PixelShuffle 2×) → 2 output channels (Bands 10 & 11).
9) Output clamp: ReLU to keep non-negative outputs.

### 7) Training Phases and Losses
- Initial phase (default): focuses on reconstruction + physics + spectral consistency.
  - total = λ_pixel·L1 + λ_physics·L_phys + λ_spectral·L_spec
- Refinement phase: adds perceptual (VGG features) and edge preservation terms.
- Adversarial phase: optional LSGAN with multi-scale discriminator.

PhysicsConsistencyLoss components:
- Sensor-consistency: PSF(sr_thermal) vs matched lr_thermal (sizes aligned internally).
- Energy conservation: mean energy consistency across scales.
- Cross-band plausibility: penalize extreme Band10–Band11 differences.
- Edge-weighted smoothness: thermal gradients weighted by inverse emissivity gradients; emissivity is upsampled to SR size internally.

### 8) Evaluation Metrics (what they mean)
- PSNR/SSIM: fidelity vs reference (for synthetic scale or held-out GT).
- RMSE (Kelvin): computed after denormalizing; measures physical temperature error.
- EPI (edge preservation index): correlation of edge maps between SR and GT.
- Spectral consistency: Band10/Band11 ratio and difference consistency.
- Sensor-consistency RMSE: PSF-downsampled SR vs observed LR.

### 9) How to Run
1) Environment:
   - `pip install -r requirements.txt`
2) Quick sanity tests:
   - `python test_simple.py`
3) Training:
   - `python train.py --config configs/train_config.yaml`
4) Inference:
   - `python inference.py --checkpoint checkpoints/thermal_sr/best_model.pth --input ssl4eo_l_oli_tirs_toa_benchmark/ssl4eo_l_oli_tirs_toa_benchmark --output results/ --save_visualization`

### 10) References
- Achermann et al., Cross-spectral registration of thermal and optical aerial imagery, CVPRW 2021.
- Dong et al., SwinFuSR (Swin Transformer for Fusion SR), CVPRW 2024.
- Raissi et al., Physics-Informed Neural Networks, JCP 2019.
- Saragadam et al., Physics-Inspired Deep Networks, ICCVW 2021.
- USGS Landsat 8 TIRS calibration notes; use Band 10 for robust LST.


