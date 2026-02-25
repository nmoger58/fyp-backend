#!/usr/bin/env python3
"""
Final Validation Script
Demonstrates that the deepfake detection model is now working correctly.
"""

import requests
import json
from pathlib import Path
from statistics import mean, stdev

BASE_URL = "http://localhost:8000"
FAKE_DIR = Path("c:\\Users\\rajmo\\deepfake\\FAKE_DIR")
REAL_DIR = Path("c:\\Users\\rajmo\\deepfake\\REAL_DIR")

def print_header(text):
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")

def print_section(text):
    print(f"\n{'─'*80}")
    print(f"  {text}")
    print(f"{'─'*80}\n")

def validate_api():
    """Verify API is running and model is loaded"""
    print_section("STEP 1: Validating API Connection")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is running")
            print(f"   Endpoint: {BASE_URL}")
            return True
        else:
            print("❌ API returned error status")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print(f"   Make sure to run: python index.py")
        return False

def quick_sample_test():
    """Test a few samples"""
    print_section("STEP 2: Quick Sample Test (5 real + 5 fake)")
    
    real_scores = []
    fake_scores = []
    
    # Sample real videos
    real_files = list(REAL_DIR.glob("*.mp4"))[:5]
    print(f"Testing {len(real_files)} real videos...")
    for video in real_files:
        try:
            with open(video, "rb") as f:
                r = requests.post(f"{BASE_URL}/predict_video", files={"file": f}, timeout=300)
            score = r.json()["prediction"]["raw_score"]
            real_scores.append(score)
            print(f"  ✓ {video.name}: {score:.4f}")
        except Exception as e:
            print(f"  ✗ {video.name}: {e}")
    
    # Sample fake videos
    fake_files = list(FAKE_DIR.glob("*.mp4"))[:5]
    print(f"\nTesting {len(fake_files)} deepfake videos...")
    for video in fake_files:
        try:
            with open(video, "rb") as f:
                r = requests.post(f"{BASE_URL}/predict_video", files={"file": f}, timeout=300)
            score = r.json()["prediction"]["raw_score"]
            fake_scores.append(score)
            print(f"  ✓ {video.name}: {score:.4f}")
        except Exception as e:
            print(f"  ✗ {video.name}: {e}")
    
    return real_scores, fake_scores

def full_validation_test():
    """Complete validation test"""
    print_section("STEP 3: Full Validation Test (25 real + 25 fake)")
    
    real_scores = []
    real_correct = 0
    fake_scores = []
    fake_correct = 0
    
    # Test real videos
    for i, video in enumerate(list(REAL_DIR.glob("*.mp4"))[:25], 1):
        try:
            with open(video, "rb") as f:
                r = requests.post(f"{BASE_URL}/predict_video", files={"file": f}, timeout=300)
            result = r.json()["prediction"]
            score = result["raw_score"]
            label = result["label"]
            real_scores.append(score)
            if label == "REAL":
                real_correct += 1
            if i % 5 == 0:
                print(f"  Real videos: {i}/25 processed")
        except Exception as e:
            print(f"  Error on video {i}: {e}")
    
    # Test fake videos
    for i, video in enumerate(list(FAKE_DIR.glob("*.mp4"))[:25], 1):
        try:
            with open(video, "rb") as f:
                r = requests.post(f"{BASE_URL}/predict_video", files={"file": f}, timeout=300)
            result = r.json()["prediction"]
            score = result["raw_score"]
            label = result["label"]
            fake_scores.append(score)
            if label == "DEEPFAKE":
                fake_correct += 1
            if i % 5 == 0:
                print(f"  Fake videos: {i}/25 processed")
        except Exception as e:
            print(f"  Error on video {i}: {e}")
    
    return real_scores, real_correct, fake_scores, fake_correct

def display_results(real_scores, real_correct, fake_scores, fake_correct):
    """Display comprehensive results"""
    print_header("FINAL VALIDATION RESULTS")
    
    if real_scores and fake_scores:
        real_acc = 100 * real_correct / len(real_scores)
        fake_acc = 100 * fake_correct / len(fake_scores)
        overall_acc = 100 * (real_correct + fake_correct) / (len(real_scores) + len(fake_scores))
        
        print("📊 REAL VIDEO CLASSIFICATION")
        print(f"  Accuracy: {real_correct}/{len(real_scores)} ({real_acc:.1f}%)")
        print(f"  Mean Score: {mean(real_scores):.4f}")
        if len(real_scores) > 1:
            print(f"  Std Dev: {stdev(real_scores):.4f}")
        print(f"  Range: {min(real_scores):.4f} - {max(real_scores):.4f}")
        
        print("\n📊 DEEPFAKE DETECTION")
        print(f"  Accuracy: {fake_correct}/{len(fake_scores)} ({fake_acc:.1f}%)")
        print(f"  Mean Score: {mean(fake_scores):.4f}")
        if len(fake_scores) > 1:
            print(f"  Std Dev: {stdev(fake_scores):.4f}")
        print(f"  Range: {min(fake_scores):.4f} - {max(fake_scores):.4f}")
        
        print("\n" + "="*80)
        print(f"{'OVERALL ACCURACY'.center(80)}")
        print("="*80)
        print(f"\n  {real_correct + fake_correct}/{len(real_scores) + len(fake_scores)} videos correct")
        print(f"  {overall_acc:.1f}% accuracy\n")
        
        if overall_acc >= 90:
            print("  ✅ EXCELLENT PERFORMANCE - Model is ready for use!")
        elif overall_acc >= 80:
            print("  ✓ GOOD PERFORMANCE - Model is working well")
        else:
            print("  ⚠️  NEEDS IMPROVEMENT - Check preprocessing")
        
        print("\n" + "="*80)

def main():
    print_header("DEEPFAKE DETECTION MODEL VALIDATION")
    
    # Step 1: Validate API
    if not validate_api():
        print("\n❌ Cannot connect to API. Please start it with: python index.py")
        return
    
    # Step 2: Quick sample test
    real_sample, fake_sample = quick_sample_test()
    
    print("\n✅ Quick test complete!")
    print(f"   Real videos mean: {mean(real_sample) if real_sample else 'N/A':.4f}")
    print(f"   Fake videos mean: {mean(fake_sample) if fake_sample else 'N/A':.4f}")
    
    input("\nPress Enter to run full validation test (25 real + 25 fake videos)...")
    
    # Step 3: Full validation
    real_scores, real_correct, fake_scores, fake_correct = full_validation_test()
    
    # Display results
    display_results(real_scores, real_correct, fake_scores, fake_correct)
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
1. Run comprehensive_test.py for extended validation (50+50 videos)
2. Use the API for production predictions: /predict_video endpoint
3. Check SOLUTION_SUMMARY.md for technical details
4. See QUICK_START_GUIDE.md for usage examples
""")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
