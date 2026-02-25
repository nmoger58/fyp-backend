import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"
FAKE_DIR = Path("c:\\Users\\rajmo\\deepfake\\FAKE_DIR")
REAL_DIR = Path("c:\\Users\\rajmo\\deepfake\\REAL_DIR")

def detailed_analysis(num_samples=20):
    """Detailed analysis of model predictions"""
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS - Understanding Model Performance")
    print("="*80 + "\n")
    
    real_scores = []
    fake_scores = []
    
    print("Testing REAL videos...")
    for video_path in list(REAL_DIR.glob("*.mp4"))[:num_samples]:
        try:
            with open(video_path, "rb") as f:
                response = requests.post(f"{BASE_URL}/predict_video", files={"file": f}, timeout=300)
            result = response.json()
            if "prediction" in result:
                score = result["prediction"]["raw_score"]
                real_scores.append(score)
        except:
            pass
    
    print(f"Testing DEEPFAKE videos...")
    for video_path in list(FAKE_DIR.glob("*.mp4"))[:num_samples]:
        try:
            with open(video_path, "rb") as f:
                response = requests.post(f"{BASE_URL}/predict_video", files={"file": f}, timeout=300)
            result = response.json()
            if "prediction" in result:
                score = result["prediction"]["raw_score"]
                fake_scores.append(score)
        except:
            pass
    
    print("\n" + "─"*80)
    print(f"REAL Videos ({len(real_scores)} samples):")
    print(f"  Min: {min(real_scores):.4f}")
    print(f"  Max: {max(real_scores):.4f}")
    print(f"  Mean: {sum(real_scores)/len(real_scores):.4f}")
    print(f"  Below 0.5: {sum(1 for s in real_scores if s < 0.5)}/{len(real_scores)}")
    print(f"  Above 0.5: {sum(1 for s in real_scores if s >= 0.5)}/{len(real_scores)}")
    
    print(f"\nDEEPFAKE Videos ({len(fake_scores)} samples):")
    print(f"  Min: {min(fake_scores):.4f}")
    print(f"  Max: {max(fake_scores):.4f}")
    print(f"  Mean: {sum(fake_scores)/len(fake_scores):.4f}")
    print(f"  Below 0.5: {sum(1 for s in fake_scores if s < 0.5)}/{len(fake_scores)}")
    print(f"  Above 0.5: {sum(1 for s in fake_scores if s >= 0.5)}/{len(fake_scores)}")
    
    print("\n" + "─"*80)
    print("INTERPRETATION:")
    print(f"- Current Threshold: 0.5")
    print(f"- Real videos mean: {sum(real_scores)/len(real_scores):.4f} (✓ Good - below 0.5)")
    print(f"- Fake videos mean: {sum(fake_scores)/len(fake_scores):.4f} (✗ Poor - should be > 0.5)")
    
    # Suggest optimal threshold
    if real_scores and fake_scores:
        real_max = max(real_scores)
        fake_min = min(fake_scores)
        if real_max < fake_min:
            print(f"\n✓ Perfect Separation! Threshold can be anywhere between {real_max:.4f} and {fake_min:.4f}")
        elif sum(1 for s in fake_scores if s > 0.5) > 0:
            suggested_threshold = sum(fake_scores) / len(fake_scores)
            print(f"\n⚠ Poor Separation. Suggested threshold: {suggested_threshold:.4f}")
        else:
            print(f"\n❌ No deepfakes scored above 0.5!")
            print(f"   This suggests the deepfake videos might be:")
            print(f"   1. Very similar to real videos (hard to detect)")
            print(f"   2. Different from training data")
            print(f"   3. Require model retraining")

detailed_analysis(num_samples=20)
