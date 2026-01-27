import requests
import json
from pathlib import Path
from statistics import mean, stdev

BASE_URL = "http://localhost:8000"
FAKE_DIR = Path("c:\\Users\\rajmo\\deepfake\\FAKE_DIR")
REAL_DIR = Path("c:\\Users\\rajmo\\deepfake\\REAL_DIR")

def test_videos(directory, label, num_samples=10):
    """Test videos from a directory"""
    videos = list(directory.glob("*.mp4"))[:num_samples]
    
    print(f"\n{'='*70}")
    print(f"Testing {label.upper()} Videos ({len(videos)} samples)")
    print(f"{'='*70}\n")
    
    results = []
    deepfake_count = 0
    real_count = 0
    
    for i, video_path in enumerate(videos, 1):
        try:
            with open(video_path, "rb") as f:
                files = {"file": f}
                response = requests.post(
                    f"{BASE_URL}/predict_video",
                    files=files,
                    timeout=300
                )
            
            result = response.json()
            if "prediction" in result:
                pred = result["prediction"]
                score = pred["raw_score"]
                confidence = pred["confidence"]
                is_deepfake = pred["is_deepfake"]
                
                results.append(score)
                if is_deepfake:
                    deepfake_count += 1
                else:
                    real_count += 1
                
                status = "🎭 DEEPFAKE" if is_deepfake else "✅ REAL"
                print(f"{i:2d}. {video_path.name:40s} | {status} | Score: {score:.4f} | Confidence: {confidence:.1%}")
            else:
                print(f"{i:2d}. {video_path.name:40s} | ❌ ERROR")
        except Exception as e:
            print(f"{i:2d}. {video_path.name:40s} | ❌ ERROR: {str(e)[:50]}")
    
    # Statistics
    if results:
        print(f"\n{'─'*70}")
        print(f"Statistics:")
        print(f"  Deepfakes Detected: {deepfake_count}/{len(videos)} ({deepfake_count/len(videos)*100:.1f}%)")
        print(f"  Real Detected: {real_count}/{len(videos)} ({real_count/len(videos)*100:.1f}%)")
        print(f"  Mean Score: {mean(results):.4f}")
        if len(results) > 1:
            print(f"  Std Dev: {stdev(results):.4f}")
        print(f"  Min Score: {min(results):.4f}")
        print(f"  Max Score: {max(results):.4f}")
    
    return results, deepfake_count, real_count

def main():
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*10 + "DEEPFAKE DETECTION MODEL - BATCH TEST" + " "*20 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Test health
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        health = response.json()
        print(f"\n✅ API Status: {health['status']}")
        print(f"   Device: {health['device']}")
    except Exception as e:
        print(f"\n❌ API is not running! Start with: python start_api.py")
        return
    
    # Test real videos
    real_results, real_deepfakes, real_legit = test_videos(REAL_DIR, "real", num_samples=15)
    
    # Test fake videos
    fake_results, fake_deepfakes, fake_legit = test_videos(FAKE_DIR, "fake/deepfake", num_samples=15)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    if real_results and fake_results:
        print(f"Real Videos Performance:")
        print(f"  ✅ Correctly Classified as REAL: {real_legit}/15 ({real_legit/15*100:.1f}%)")
        print(f"  ❌ Incorrectly Classified as DEEPFAKE: {real_deepfakes}/15 ({real_deepfakes/15*100:.1f}%)")
        print(f"  Mean Score: {mean(real_results):.4f} (should be < 0.5)")
        
        print(f"\nDeepfake Videos Performance:")
        print(f"  🎭 Correctly Classified as DEEPFAKE: {fake_deepfakes}/15 ({fake_deepfakes/15*100:.1f}%)")
        print(f"  ❌ Incorrectly Classified as REAL: {fake_legit}/15 ({fake_legit/15*100:.1f}%)")
        print(f"  Mean Score: {mean(fake_results):.4f} (should be > 0.5)")
        
        print(f"\nOverall Accuracy:")
        total_correct = real_legit + fake_deepfakes
        total_tests = 30
        accuracy = (total_correct / total_tests) * 100
        print(f"  {total_correct}/30 ({accuracy:.1f}%)")
        
        print(f"\n{'─'*70}")
        
        # Interpretation
        if accuracy > 80:
            print("🎉 EXCELLENT! Model is working very well!")
        elif accuracy > 70:
            print("👍 GOOD! Model is working well.")
        elif accuracy > 60:
            print("⚠️  FAIR! Model has room for improvement.")
        else:
            print("❌ POOR! Model needs adjustment or retraining.")

if __name__ == "__main__":
    main()
