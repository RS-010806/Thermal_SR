# Optical-Guided Super-Resolution for Thermal IR Imagery

A state-of-the-art deep learning solution for super-resolving Landsat-8 thermal infrared (TIRS) bands using optical guidance from OLI bands. This project targets CVPR 2026 submission.

## Key Features

### Novel Architecture Components
1. **Guidance Disentanglement Network (GDNet)** - Separates modality-specific and shared features for optimal fusion
2. **Physics-Informed Processing** - Incorporates sensor modeling, atmospheric correction, and radiative transfer physics
3. **Multi-Scale Swin Transformer Backbone** - Efficient attention mechanism for capturing long-range dependencies
4. **Weather-Aware Adaptive Fusion** - Dynamically adjusts optical guidance based on atmospheric conditions
5. **Emissivity-Gated Cross-Modal Attention** - Prevents texture leakage in thermally homogeneous regions

### Physics Components
- **Thermal PSF Modeling** - Simulates TIRS sensor response including optical blur and detector integration
- **Atmospheric Correction** - Models atmospheric transmittance and radiance effects
- **Land Surface Emissivity Estimation** - NDVI-based emissivity computation with learnable refinement
- **Sensor Noise Modeling** - Realistic NEΔT-based noise injection during training
- **Cross-Band Consistency** - Ensures physical plausibility between Band 10 and Band 11

## Dataset

The project uses the SSL4EO Landsat dataset containing:
- **OLI Bands (30m)**: Bands 1-9 covering visible, NIR, and SWIR
- **TIRS Bands (100m → 30m)**: Bands 10-11 thermal infrared
- **25,000 scenes** with global coverage

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/thermal-sr.git
cd thermal-sr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train with default configuration
python train.py --config configs/train_config.yaml

# Resume training from checkpoint
python train.py --config configs/train_config.yaml --resume --resume_path checkpoints/thermal_sr/checkpoint_epoch_100.pth
```

### Inference

```bash
# Run inference on a single scene
python inference.py --checkpoint checkpoints/thermal_sr/best_model.pth \
                   --input path/to/scene/all_bands.tif \
                   --output results/ \
                   --save_visualization

# Run inference on multiple scenes
python inference.py --checkpoint checkpoints/thermal_sr/best_model.pth \
                   --input ssl4eo_l_oli_tirs_toa_benchmark/ssl4eo_l_oli_tirs_toa_benchmark/ \
                   --output results/ \
                   --batch_size 4
```

## Model Architecture

```
Input (11 bands, 30m)
    ├── Thermal Bands (10, 11) → Multi-Scale Thermal Encoder
    │   └── Quantum-Aware Processing → Sensor Response Modeling
    │
    ├── Optical Bands (1-9) → Multi-Spectral Optical Encoder
    │   └── Band-Specific Processing → Spectral Attention
    │
    ├── Weather Analysis → Adaptive Routing Module
    │   └── Cloud/Haze Detection → Fusion Weights
    │
    └── NDVI Calculation → Emissivity Estimation Head
        └── Physics-Based + Learned Refinement

Cross-Modal Fusion
    ├── Guidance Disentanglement (GDNet)
    │   ├── Modality-Specific Features
    │   └── Shared Features (Adversarial Training)
    │
    └── Emissivity-Gated Cross-Attention
        └── Multi-Scale Feature Fusion

Physics-Informed Decoder
    ├── Progressive Upsampling (2x/4x)
    ├── Physics Validation Layers
    └── Sensor Consistency Check

Output
    ├── Super-Resolved Thermal (Bands 10, 11)
    ├── Emissivity Map
    └── Physics Metrics
```

## Training Strategy

### Three-Phase Curriculum Learning
1. **Initial Phase (100 epochs)**
   - Focus on pixel reconstruction and physics consistency
   - Basic thermal-only features with minimal optical guidance

2. **Refinement Phase (100 epochs)**
   - Add perceptual and edge preservation losses
   - Enable full cross-modal attention

3. **Adversarial Phase (100 epochs)**
   - Full GAN training with multi-scale discriminator
   - Fine-tune all components jointly

### Loss Functions
- **Pixel Loss**: L1 reconstruction error
- **Perceptual Loss**: VGG-based feature matching
- **Physics Consistency Loss**: Sensor downprojection matching
- **Spectral Consistency Loss**: Cross-band correlation
- **Edge Preservation Loss**: Gradient magnitude matching
- **Adversarial Loss**: LSGAN formulation

## Evaluation Metrics

- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index
- **RMSE (Kelvin)**: Root Mean Square Error in temperature units
- **Edge Preservation Index**: Correlation of edge maps
- **Spectral Consistency**: Band ratio preservation
- **Physics Consistency**: Sensor model compliance

## References

Key papers this work builds upon:
- Guidance Disentanglement Network (GDNet), arXiv:2410.20466
- SwinFuSR: Swin Transformer for Multi-Modal Super-Resolution, CVPRW 2024
- Physics-Informed Neural Networks, Raissi et al., JCP 2019
- Cross-spectral registration of thermal and optical aerial imagery, CVPRW 2021
- Thermal Image Processing via Physics-Inspired Deep Networks, ICCVW 2021
## Acknowledgments

- ISRO for the problem statement and evaluation framework
- SSL4EO dataset creators
- PyTorch and torchvision teams
