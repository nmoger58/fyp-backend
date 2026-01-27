"""
Deepfake Detection API - Python Client
Simple script to test the API from command line
"""

import requests
import sys
import json
from pathlib import Path


class DeepfakeDetectionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_info(self):
        """Get API information"""
        try:
            response = requests.get(f"{self.base_url}/info", timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict(self, video_path):
        """
        Predict if video is deepfake
        
        Args:
            video_path: Path to video file
        
        Returns:
            dict with prediction results
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            return {"error": f"File not found: {video_path}"}
        
        try:
            with open(video_path, "rb") as f:
                files = {"file": f}
                response = requests.post(
                    f"{self.base_url}/predict_video",
                    files=files,
                    timeout=300  # 5 minute timeout for video processing
                )
            
            return response.json()
        
        except requests.exceptions.Timeout:
            return {"error": "Request timeout - video processing took too long"}
        except Exception as e:
            return {"error": str(e)}
    
    def print_result(self, result):
        """Pretty print prediction result"""
        if "error" in result:
            print(f"\n❌ Error: {result['error']}\n")
            return
        
        print("\n" + "="*60)
        print("DEEPFAKE DETECTION RESULT")
        print("="*60)
        
        if "filename" in result:
            print(f"📁 Filename: {result['filename']}")
        
        if "prediction" in result:
            pred = result["prediction"]
            label = pred["label"]
            confidence = pred["confidence"]
            raw_score = pred["raw_score"]
            
            # Color coding
            if label == "DEEPFAKE":
                emoji = "⚠️ "
                status = "LIKELY DEEPFAKE"
            else:
                emoji = "✅"
                status = "LIKELY REAL"
            
            print(f"\n{emoji} Prediction: {status}")
            print(f"   Raw Score: {raw_score:.4f}")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   Is Deepfake: {pred['is_deepfake']}")
        
        print("\n" + "="*60 + "\n")


def main():
    """Command line interface"""
    
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <video_path>")
        print("\nExamples:")
        print("  python test_api.py video.mp4")
        print("  python test_api.py --health")
        print("  python test_api.py --info")
        sys.exit(1)
    
    client = DeepfakeDetectionClient()
    
    if sys.argv[1] == "--health":
        print("\n🏥 Checking API health...")
        result = client.health_check()
        print(json.dumps(result, indent=2))
    
    elif sys.argv[1] == "--info":
        print("\nℹ️  Getting API information...")
        result = client.get_info()
        print(json.dumps(result, indent=2))
    
    else:
        video_path = sys.argv[1]
        print(f"\n🎬 Processing video: {video_path}")
        print("⏳ This may take a minute...")
        
        result = client.predict(video_path)
        client.print_result(result)


if __name__ == "__main__":
    main()
