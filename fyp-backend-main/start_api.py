#!/usr/bin/env python
"""
Quick Start Script - Deepfake Detection API
Run this to start the API immediately
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header():
    print("\n" + "="*60)
    print("🎬 Deepfake Video Detection API")
    print("="*60 + "\n")


def check_model_file():
    """Check if model file exists"""
    model_path = Path("best_model.pth")
    if not model_path.exists():
        print("❌ Error: best_model.pth not found!")
        print(f"   Expected location: {model_path.absolute()}")
        return False
    print(f"✅ Model file found: {model_path.absolute()}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "torch": "torch",
        "torchvision": "torchvision",
        "cv2": "opencv-python",
        "numpy": "numpy",
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
        return False
    
    return True


def check_model_loads():
    """Test if model can be loaded"""
    try:
        from model import MyModel
        model = MyModel()
        print("✅ Model loads successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def start_api():
    """Start the FastAPI server"""
    print("\n" + "="*60)
    print("🚀 Starting API Server...")
    print("="*60)
    print("\n📍 API will be available at:")
    print("   - API: http://localhost:8000")
    print("   - Docs: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("\n💡 Test with: python test_api.py <video_path>")
    print("   or visit http://localhost:8000/health\n")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "index:app", "--reload", "--port", "8000"],
            check=False
        )
    except KeyboardInterrupt:
        print("\n\n👋 API stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting API: {e}")


def main():
    print_header()
    
    print("📋 Checking Prerequisites...\n")
    
    # Check model file
    if not check_model_file():
        print("\n❌ Setup failed: Model file missing")
        sys.exit(1)
    
    print("\n📦 Checking Dependencies...\n")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup failed: Missing dependencies")
        sys.exit(1)
    
    print("\n🤖 Testing Model Loading...\n")
    
    # Check model loads
    if not check_model_loads():
        print("\n❌ Setup failed: Model load error")
        sys.exit(1)
    
    print("\n✅ All checks passed!")
    
    # Start API
    start_api()


if __name__ == "__main__":
    main()
