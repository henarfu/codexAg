import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add the project root to sys.path to ensure imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

from codeletsA.unet_leon import UNetLeon # Assuming UNetLeon is correctly imported from here

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    n_channels = 3 # RGB input
    target_params = 7_000_000
    
    # Let's check the default base_channel=64 first
    model_default = UNetLeon(n_channels=n_channels, base_channel=64)
    params_default = count_parameters(model_default)
    print(f"UNetLeon with default base_channel=64: {params_default:,} parameters")

    print("\nSearching for base_channel value close to 7,000,000 parameters:")

    current_base_channel = 32 # Starting lower for a finer search
    step = 4 # Small increment
    found_close_match = False
    
    # Keep track of the closest we get
    closest_params = params_default
    closest_base_channel = 64

    # Iterate through a reasonable range of base_channel values
    while current_base_channel <= 128: # Limiting search to a reasonable upper bound
        model = UNetLeon(n_channels=n_channels, base_channel=current_base_channel)
        params = count_parameters(model)
        
        print(f"base_channel={current_base_channel}: {params:,} parameters")

        if abs(params - target_params) < abs(closest_params - target_params):
            closest_params = params
            closest_base_channel = current_base_channel

        if abs(params - target_params) < target_params * 0.05: # Within 5% of target
            print(f"\nFound a base_channel ({current_base_channel}) with approximately {target_params:,} parameters.")
            found_close_match = True
            break
        
        current_base_channel += step
        
    if not found_close_match:
        print(f"\nCould not find a base_channel that results in exactly {target_params:,} parameters within the search range.")
        print(f"Closest found: base_channel={closest_base_channel} with {closest_params:,} parameters.")