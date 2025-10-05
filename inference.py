"""
Inference script for thermal super-resolution model.

Usage:
    python inference.py --checkpoint path/to/checkpoint.pth --input path/to/input --output path/to/output
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import rasterio
from tqdm import tqdm

from src.models import ThermalSuperResolutionNet
from src.data_loader import InferenceDataset
from src.utils import visualize_results, save_geotiff, apply_thermal_colormap
from src.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description='Thermal Super-Resolution Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input file or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--save_visualization', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ThermalSuperResolutionNet(scale_factor=2)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {args.checkpoint}")
    
    # Prepare input paths
    input_path = Path(args.input)
    if input_path.is_file():
        scene_paths = [str(input_path)]
    else:
        scene_paths = list(input_path.glob("*/*/all_bands.tif"))
        scene_paths = [str(p) for p in scene_paths]
    
    print(f"Found {len(scene_paths)} scenes to process")
    
    # Create dataset
    dataset = InferenceDataset(scene_paths, normalize=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Process scenes
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing scenes")):
            # Get data
            thermal_lr = batch['thermal'].to(device)
            optical = batch['optical'].to(device)
            scene_path = batch['scene_path'][0]
            transform = batch['transform']
            crs = batch['crs']
            
            # Run inference
            thermal_sr, emissivity, physics_outputs = model(thermal_lr, optical)
            
            # Move to CPU and denormalize
            thermal_sr = thermal_sr.cpu()
            thermal_lr = thermal_lr.cpu()
            emissivity = emissivity.cpu()
            
            # Denormalize thermal data
            thermal_mean = torch.tensor(dataset.thermal_mean).view(1, -1, 1, 1)
            thermal_std = torch.tensor(dataset.thermal_std).view(1, -1, 1, 1)
            
            thermal_sr_kelvin = thermal_sr * thermal_std + thermal_mean
            thermal_lr_kelvin = thermal_lr * thermal_std + thermal_mean
            
            # Process each sample in batch
            for i in range(thermal_sr.shape[0]):
                # Create output filename
                scene_name = Path(scene_path).parent.name
                
                # Save super-resolved thermal bands as GeoTIFF
                sr_path = output_dir / f"{scene_name}_thermal_sr.tif"
                save_geotiff(
                    thermal_sr_kelvin[i].numpy(),
                    str(sr_path),
                    transform,
                    crs,
                    band_descriptions=['Band 10 SR (10.60-11.19 μm)', 
                                     'Band 11 SR (11.50-12.51 μm)']
                )
                
                # Save emissivity map
                emis_path = output_dir / f"{scene_name}_emissivity.tif"
                save_geotiff(
                    emissivity[i].numpy(),
                    str(emis_path),
                    transform,
                    crs,
                    band_descriptions=['Estimated Emissivity']
                )
                
                # Save visualization if requested
                if args.save_visualization:
                    # Create RGB composite from optical bands
                    optical_np = optical[i].cpu().numpy()
                    optical_denorm = optical_np * np.array(dataset.optical_std)[:, None, None] + \
                                   np.array(dataset.optical_mean)[:, None, None]
                    
                    # RGB composite (bands 4, 3, 2)
                    rgb = np.stack([
                        optical_denorm[3],  # Red
                        optical_denorm[2],  # Green
                        optical_denorm[1]   # Blue
                    ])
                    rgb = np.transpose(rgb, (1, 2, 0))
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                    
                    # Average thermal bands for visualization
                    thermal_lr_avg = thermal_lr_kelvin[i].mean(0).numpy()
                    thermal_sr_avg = thermal_sr_kelvin[i].mean(0).numpy()
                    
                    # Create visualization
                    vis_dict = {
                        'LR Thermal': thermal_lr_avg,
                        'SR Thermal': thermal_sr_avg,
                        'Optical RGB': rgb,
                        'Emissivity': emissivity[i, 0].numpy()
                    }
                    
                    vis_path = output_dir / f"{scene_name}_visualization.png"
                    fig = visualize_results(vis_dict, save_path=str(vis_path))
                    plt.close(fig)
                    
                    # Save colored thermal images
                    thermal_lr_colored = apply_thermal_colormap(thermal_lr_avg)
                    thermal_sr_colored = apply_thermal_colormap(thermal_sr_avg)
                    
                    cv2.imwrite(
                        str(output_dir / f"{scene_name}_thermal_lr_colored.png"),
                        (thermal_lr_colored * 255).astype(np.uint8)[:, :, ::-1]
                    )
                    cv2.imwrite(
                        str(output_dir / f"{scene_name}_thermal_sr_colored.png"),
                        (thermal_sr_colored * 255).astype(np.uint8)[:, :, ::-1]
                    )
                    
                # Save metadata
                metadata = {
                    'scene_path': scene_path,
                    'scale_factor': 2,
                    'input_shape': list(thermal_lr.shape),
                    'output_shape': list(thermal_sr.shape),
                    'thermal_range_kelvin': {
                        'lr': {
                            'min': float(thermal_lr_kelvin[i].min()),
                            'max': float(thermal_lr_kelvin[i].max()),
                            'mean': float(thermal_lr_kelvin[i].mean())
                        },
                        'sr': {
                            'min': float(thermal_sr_kelvin[i].min()),
                            'max': float(thermal_sr_kelvin[i].max()),
                            'mean': float(thermal_sr_kelvin[i].mean())
                        }
                    },
                    'emissivity_range': {
                        'min': float(emissivity[i].min()),
                        'max': float(emissivity[i].max()),
                        'mean': float(emissivity[i].mean())
                    }
                }
                
                import json
                with open(output_dir / f"{scene_name}_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
    print(f"\nInference completed! Results saved to: {output_dir}")


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import cv2
    
    main()
