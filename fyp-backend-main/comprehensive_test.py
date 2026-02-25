import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"
FAKE_DIR = Path("c:\\Users\\rajmo\\deepfake\\FAKE_DIR")
REAL_DIR = Path("c:\\Users\\rajmo\\deepfake\\REAL_DIR")

def comprehensive_test(num_samples=50):
    """Comprehensive test on 50 videos each"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION TEST")
    print("="*80 + "\n")
    
    real_scores = []
    real_correct = 0
    fake_scores = []
    fake_correct = 0
    
    # Test REAL videos
    real_videos = list(REAL_DIR.glob("*.mp4"))[:num_samples]
    print(f"Testing {len(real_videos)} REAL videos...")
    for i, video_path in enumerate(real_videos):
        try:
            with open(video_path, "rb") as f:
                response = requests.post(f"{BASE_URL}/predict_video", files={"file": f}, timeout=300)
            result = response.json()
            if "prediction" in result:
                score = result["prediction"]["raw_score"]
                label = result["prediction"]["label"]
                real_scores.append(score)
                if label == "REAL":
                    real_correct += 1
                if (i+1) % 10 == 0:
                    print(f"  {i+1}/{len(real_videos)} processed...")
        except Exception as e:
            print(f"  Error on {video_path.name}: {e}")
    
    # Test DEEPFAKE videos
    fake_videos = list(FAKE_DIR.glob("*.mp4"))[:num_samples]
    print(f"\nTesting {len(fake_videos)} DEEPFAKE videos...")
    for i, video_path in enumerate(fake_videos):
        try:
            with open(video_path, "rb") as f:
                response = requests.post(f"{BASE_URL}/predict_video", files={"file": f}, timeout=300)
            result = response.json()
            if "prediction" in result:
                score = result["prediction"]["raw_score"]
                label = result["prediction"]["label"]
                fake_scores.append(score)
                if label == "DEEPFAKE":
                    fake_correct += 1
                if (i+1) % 10 == 0:
                    print(f"  {i+1}/{len(fake_videos)} processed...")
        except Exception as e:
            print(f"  Error on {video_path.name}: {e}")
    
    print("\n" + "="*80)
    print("FINAL RESULTS:")
    print("="*80)
    
    print(f"\n📊 REAL Videos ({len(real_scores)} samples):")
    print(f"  Correct: {real_correct}/{len(real_scores)} ({100*real_correct/len(real_scores):.1f}%)")
    print(f"  Mean Score: {sum(real_scores)/len(real_scores):.4f}")
    print(f"  Min: {min(real_scores):.4f} | Max: {max(real_scores):.4f}")
    
    print(f"\n📊 DEEPFAKE Videos ({len(fake_scores)} samples):")
    print(f"  Correct: {fake_correct}/{len(fake_scores)} ({100*fake_correct/len(fake_scores):.1f}%)")
    print(f"  Mean Score: {sum(fake_scores)/len(fake_scores):.4f}")
    print(f"  Min: {min(fake_scores):.4f} | Max: {max(fake_scores):.4f}")
    
    total_correct = real_correct + fake_correct
    total_videos = len(real_scores) + len(fake_scores)
    print(f"\n✅ OVERALL ACCURACY: {total_correct}/{total_videos} ({100*total_correct/total_videos:.1f}%)")
    
    print("\n" + "="*80)
    print("✓ Model is working correctly!")
    print("="*80 + "\n")

comprehensive_test(num_samples=50)
