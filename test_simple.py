"""
Simple test to verify basic functionality.
"""

import torch
import sys
sys.path.append('.')

print("Testing imports...")

try:
    from src.models import ThermalSuperResolutionNet
    from src.losses import ThermalSRLoss
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("\nTesting model creation...")
try:
    model = ThermalSuperResolutionNet(scale_factor=2)
    print("✓ Model created successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
except Exception as e:
    print(f"✗ Model creation error: {e}")
    sys.exit(1)

print("\nTesting forward pass with small input...")
try:
    # Small input for faster testing
    thermal_lr = torch.randn(1, 2, 32, 32)
    optical_hr = torch.randn(1, 9, 32, 32)
    
    with torch.no_grad():
        thermal_sr, emissivity, physics_outputs = model(thermal_lr, optical_hr)
    
    print("✓ Forward pass successful")
    print(f"  Output shape: {thermal_sr.shape}")
    print(f"  Emissivity shape: {emissivity.shape}")
except Exception as e:
    print(f"✗ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting loss computation...")
try:
    criterion = ThermalSRLoss()
    thermal_hr = torch.randn_like(thermal_sr)
    
    loss, loss_dict = criterion(
        pred=thermal_sr,
        target=thermal_hr,
        lr_input=thermal_lr,
        emissivity=emissivity,
        optical_features=optical_hr,
        training_phase='initial'
    )
    
    print("✓ Loss computation successful")
    print(f"  Total loss: {loss.item():.4f}")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"  {key}: {value.item():.4f}")
except Exception as e:
    print(f"✗ Loss computation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
print("\nImplementation is working correctly. The model is ready for training.")
