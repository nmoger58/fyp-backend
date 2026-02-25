"""
Test if removing ImageNet normalization improves deepfake detection.

Hypothesis: The model was trained WITHOUT ImageNet normalization,
only with simple division by 255.0 to convert [0,255] to [0,1]
"""

import requests
import numpy as np
from pathlib import Path

BASE_URL = "http://localhost:8000"
FAKE_DIR = Path("c:\\Users\\rajmo\\deepfake\\FAKE_DIR")
REAL_DIR = Path("c:\\Users\\rajmo\\deepfake\\REAL_DIR")

print("\n" + "="*80)
print("NORMALIZATION HYPOTHESIS TEST")
print("="*80)
print("\nCurrent hypothesis:")
print("  The model expects input in range [0, 1] (divided by 255)")
print("  WITHOUT ImageNet normalization (mean subtraction)")
print("\nWe'll test by modifying index.py to remove ImageNet normalization")
print("="*80 + "\n")

# Get sample videos
real_videos = list(REAL_DIR.glob("*.mp4"))[:5]
fake_videos = list(FAKE_DIR.glob("*.mp4"))[:5]

print("To test this hypothesis, we need to:")
print("1. Modify index.py to remove ImageNet normalization")
print("2. Keep only: faces / 255.0 (no mean subtraction)")
print("3. Restart the API")
print("4. Re-test the model\n")

# Test which approach would work
print("Sample videos for testing:")
print(f"Real (first 5): {[v.name for v in real_videos]}")
print(f"Fake (first 5): {[v.name for v in fake_videos]}")
