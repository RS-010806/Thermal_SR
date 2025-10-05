import rasterio
import numpy as np
from pathlib import Path

# Correct path with nested structure
sample_path = Path("ssl4eo_l_oli_tirs_toa_benchmark/ssl4eo_l_oli_tirs_toa_benchmark/0000000/LC08_045030_20190814/all_bands.tif")

print(f"Inspecting: {sample_path}")
print(f"File exists: {sample_path.exists()}")

if sample_path.exists():
    with rasterio.open(sample_path) as src:
        print(f"\nDataset Info:")
        print(f"- Number of bands: {src.count}")
        print(f"- Width x Height: {src.width} x {src.height}")
        print(f"- Resolution: {src.res}")
        print(f"- CRS: {src.crs}")
        print(f"- Data type: {src.dtypes[0]}")
        
        # All bands are resampled to 30m according to the problem statement
        print(f"\nAll bands have been resampled to 30m resolution")
        
        print(f"\nBand Information (Landsat 8 mapping):")
        band_info = [
            (1, "Coastal Aerosol", "0.43-0.45 μm", "30m"),
            (2, "Blue", "0.45-0.51 μm", "30m"),
            (3, "Green", "0.53-0.59 μm", "30m"),
            (4, "Red", "0.64-0.67 μm", "30m"),
            (5, "NIR", "0.85-0.88 μm", "30m"),
            (6, "SWIR 1", "1.57-1.65 μm", "30m"),
            (7, "SWIR 2", "2.11-2.29 μm", "30m"),
            (8, "PAN", "0.50-0.68 μm", "15m->30m"),
            (9, "Cirrus", "1.36-1.38 μm", "30m"),
            (10, "Thermal IR 1", "10.60-11.19 μm", "100m->30m"),
            (11, "Thermal IR 2", "11.50-12.51 μm", "100m->30m")
        ]
        
        print(f"\nBand Statistics:")
        for i, (band_num, name, wavelength, res) in enumerate(band_info, 1):
            if i <= src.count:
                data = src.read(i)
                print(f"Band {band_num} ({name}, {wavelength}):")
                print(f"  Shape: {data.shape}, Min: {data.min():.3f}, Max: {data.max():.3f}, Mean: {data.mean():.3f}, Std: {data.std():.3f}")
                
        # Special focus on thermal bands
        if src.count >= 11:
            print(f"\n=== THERMAL BANDS (Target for Super-Resolution) ===")
            thermal_10 = src.read(10)
            thermal_11 = src.read(11)
            
            print(f"\nBand 10 (TIRS1, 10.60-11.19 μm):")
            print(f"  Current: 30m (upsampled from 100m)")
            print(f"  Min: {thermal_10.min():.3f}, Max: {thermal_10.max():.3f}")
            print(f"  Mean: {thermal_10.mean():.3f}, Std: {thermal_10.std():.3f}")
            
            print(f"\nBand 11 (TIRS2, 11.50-12.51 μm):")
            print(f"  Current: 30m (upsampled from 100m)")
            print(f"  Min: {thermal_11.min():.3f}, Max: {thermal_11.max():.3f}")
            print(f"  Mean: {thermal_11.mean():.3f}, Std: {thermal_11.std():.3f}")
            
            # Check if values are TOA radiance or brightness temperature
            if thermal_10.max() < 50:  # Likely radiance
                print(f"\nThermal bands appear to be in TOA radiance units (W/m²/sr/μm)")
            else:  # Likely temperature
                print(f"\nThermal bands appear to be in brightness temperature units (K or scaled)")
                
        # Read all bands for shape verification
        all_bands = src.read()
        print(f"\nAll bands tensor shape: {all_bands.shape} (bands, height, width)")
        
else:
    print(f"File not found at {sample_path}")
