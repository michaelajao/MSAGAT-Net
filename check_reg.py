import sys
sys.path.insert(0, 'src')
import torch
from model import MSTAGAT_Net

# Load the model state dict
state_dict = torch.load('save/best_model.pt', map_location='cpu')

# Extract just the regularization weight from the state dict
log_reg_weight = state_dict['spatial_module.log_attention_reg_weight']
reg_weight = torch.exp(log_reg_weight).item()

print(f"Learned regularization weight: {reg_weight:.2e}")
print(f"Started at: 1.00e-05")

if reg_weight < 1e-7:
    print("\n⚠️  WARNING: Weight learned close to zero!")
    print("Regularization is essentially disabled.")
    print("Recommendation: Add clamping to prevent collapse.")
elif reg_weight < 1e-6:
    print("\n⚠️  Weight is very low but not collapsed.")
    print("Consider clamping if this is undesirable.")
elif reg_weight > 1e-3:
    print("\n⚠️  Weight is quite high.")
    print("May be over-regularizing attention.")
else:
    print("\n✓ Weight is in reasonable range (1e-6 to 1e-3)")
