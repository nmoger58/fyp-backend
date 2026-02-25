"""
Debug script to diagnose model loading issues
"""

import torch
from model import MyModel
import os

print("=" * 60)
print("MODEL LOADING DIAGNOSTIC")
print("=" * 60)

# Check model file
print("\n1. Checking model file...")
if os.path.exists("best_model.pth"):
    size_mb = os.path.getsize("best_model.pth") / (1024 * 1024)
    print(f"   ✅ File found: {size_mb:.2f} MB")
else:
    print("   ❌ File not found")
    exit(1)

# Load checkpoint
print("\n2. Loading checkpoint...")
checkpoint = torch.load("best_model.pth", map_location="cpu", weights_only=False)
print(f"   ✅ Checkpoint loaded")
print(f"   Keys in checkpoint: {list(checkpoint.keys())}")

# Check what's in the checkpoint
if "model_state_dict" in checkpoint:
    print(f"   ✅ 'model_state_dict' found")
    state_dict = checkpoint["model_state_dict"]
    print(f"      Keys: {list(state_dict.keys())[:5]}... (showing first 5)")
elif "state_dict" in checkpoint:
    print(f"   ✅ 'state_dict' found")
    state_dict = checkpoint["state_dict"]
elif isinstance(checkpoint, dict) and any(k.startswith("backbone") for k in checkpoint.keys()):
    print(f"   ✅ Direct state dict format")
    state_dict = checkpoint
else:
    print(f"   ⚠️  Unknown checkpoint format")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

# Create model
print("\n3. Creating model...")
model = MyModel(num_frames=16)
print(f"   ✅ Model created")

# Check model structure
print("\n4. Model parameters:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

# Try loading state dict
print("\n5. Loading state dict...")
try:
    model.load_state_dict(state_dict)
    print(f"   ✅ State dict loaded successfully")
except Exception as e:
    print(f"   ❌ Error loading state dict:")
    print(f"      {e}")
    print(f"\n   Model state dict keys: {list(model.state_dict().keys())[:5]}...")
    print(f"   Checkpoint keys: {list(state_dict.keys())[:5]}...")

# Check model eval mode
print("\n6. Model configuration:")
model.eval()
print(f"   ✅ Model set to eval mode")

# Test with dummy input
print("\n7. Testing with dummy input...")
try:
    with torch.no_grad():
        dummy_input = torch.randn(1, 16, 3, 224, 224)
        output = model(dummy_input)
        print(f"   ✅ Forward pass successful")
        print(f"      Input shape: {dummy_input.shape}")
        print(f"      Output shape: {output.shape}")
        print(f"      Output value: {output.item():.4f}")
        print(f"      Sigmoid: {torch.sigmoid(output).item():.4f}")
except Exception as e:
    print(f"   ❌ Forward pass failed:")
    print(f"      {e}")

print("\n" + "=" * 60)
print("CHECKPOINT CONTENTS")
print("=" * 60)
for key, value in checkpoint.items():
    if isinstance(value, dict):
        print(f"{key}: dict with {len(value)} items")
    elif isinstance(value, torch.Tensor):
        print(f"{key}: tensor {value.shape}")
    else:
        print(f"{key}: {type(value)}")

print("\n" + "=" * 60)
