#!/usr/bin/env python
"""
Verification Script - Deepfake Detection API
Tests all components to ensure everything is working correctly
"""

import sys
import json
from pathlib import Path


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_imports():
    """Test all required imports"""
    print_section("Testing Imports")
    
    imports = {
        "fastapi": "FastAPI",
        "uvicorn": "uvicorn",
        "torch": "torch",
        "torchvision": "torchvision",
        "cv2": "opencv-python",
        "numpy": "numpy",
        "requests": "requests",
    }
    
    failed = []
    for module, package in imports.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed.append(package)
    
    return len(failed) == 0


def test_model_file():
    """Check model file exists"""
    print_section("Checking Model File")
    
    model_path = Path("best_model.pth")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file found")
        print(f"   Path: {model_path.absolute()}")
        print(f"   Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"❌ Model file NOT found")
        print(f"   Expected: {model_path.absolute()}")
        return False


def test_model_import():
    """Test model import and instantiation"""
    print_section("Testing Model Definition")
    
    try:
        from model import MyModel
        print("✅ MyModel imported successfully")
        
        model = MyModel(num_frames=16)
        print("✅ MyModel instantiated successfully")
        print(f"   Architecture:")
        print(f"   - Backbone: EfficientNet-B0")
        print(f"   - Temporal Convolutions: 3 layers")
        print(f"   - Attention: Temporal + Channel + Spatial")
        print(f"   - Classifier: 4-layer MLP")
        
        return True
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False


def test_face_extractor():
    """Test FaceExtractor import"""
    print_section("Testing Face Extractor")
    
    try:
        from model import FaceExtractor
        print("✅ FaceExtractor imported successfully")
        
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        extractor = FaceExtractor(face_size=224, device=device)
        print("✅ FaceExtractor instantiated successfully")
        print(f"   Device: {device}")
        print(f"   Face Size: 224×224")
        
        return True
    except Exception as e:
        print(f"❌ FaceExtractor error: {e}")
        return False


def test_fastapi():
    """Test FastAPI app structure"""
    print_section("Testing FastAPI Application")
    
    try:
        # Check if index.py exists and can be imported
        import index
        print("✅ index.py imported successfully")
        
        # Check if app exists
        if hasattr(index, 'app'):
            print("✅ FastAPI app created")
            
            # Check endpoints
            endpoints = [route.path for route in index.app.routes]
            print(f"✅ API endpoints registered:")
            for ep in sorted(set(endpoints)):
                print(f"   - {ep}")
            
            return True
        else:
            print("❌ FastAPI app not found in index.py")
            return False
    
    except Exception as e:
        print(f"❌ FastAPI app error: {e}")
        return False


def test_preprocessing():
    """Test preprocessing pipeline"""
    print_section("Testing Preprocessing Pipeline")
    
    try:
        from index import VideoPreprocessor
        import numpy as np
        
        preprocessor = VideoPreprocessor(num_frames=16)
        print("✅ VideoPreprocessor instantiated successfully")
        
        # Test with dummy data
        dummy_faces = np.random.randint(0, 256, (16, 224, 224, 3), dtype=np.uint8)
        result = preprocessor.preprocess_faces(dummy_faces)
        
        if result is not None and result.shape == (1, 16, 3, 224, 224):
            print("✅ Preprocessing pipeline working correctly")
            print(f"   Input shape: (16, 224, 224, 3)")
            print(f"   Output shape: {result.shape}")
            return True
        else:
            print("❌ Preprocessing output shape incorrect")
            return False
    
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return False


def test_device():
    """Check device (GPU/CPU) availability"""
    print_section("Checking Device Availability")
    
    try:
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ PyTorch device: {device}")
        
        if torch.cuda.is_available():
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   GPU Memory: {gpu_memory:.2f} GB")
        else:
            print("   Note: Using CPU (slower inference)")
        
        return True
    
    except Exception as e:
        print(f"❌ Device check error: {e}")
        return False


def test_files():
    """Check required files"""
    print_section("Checking Required Files")
    
    required_files = {
        "index.py": "Main FastAPI application",
        "model/__init__.py": "Model definition",
        "model/face_extractor.py": "Face extraction",
        "best_model.pth": "Trained model weights",
        "requirements.txt": "Dependencies",
        "README.md": "Documentation",
    }
    
    failed = []
    for filename, description in required_files.items():
        path = Path(filename)
        if path.exists():
            print(f"✅ {filename}")
            print(f"   ({description})")
        else:
            print(f"❌ {filename}")
            print(f"   ({description})")
            failed.append(filename)
    
    return len(failed) == 0


def main():
    print("\n" + "="*60)
    print("🧪 DEEPFAKE DETECTION API - VERIFICATION TEST")
    print("="*60)
    
    # Run all tests
    results = {
        "Files": test_files(),
        "Imports": test_imports(),
        "Model File": test_model_file(),
        "Model Definition": test_model_import(),
        "Face Extractor": test_face_extractor(),
        "Preprocessing": test_preprocessing(),
        "Device": test_device(),
        "FastAPI App": test_fastapi(),
    }
    
    # Summary
    print_section("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*60)
        print("🎉 All tests passed! API is ready to use.")
        print("="*60)
        print("\n🚀 To start the API:")
        print("   python start_api.py")
        print("\n🧪 To test the API:")
        print("   python test_api.py video.mp4")
        print("\n📚 Documentation:")
        print("   - README.md")
        print("   - API_TESTING_GUIDE.md")
        print("   - INTEGRATION_SUMMARY.md")
        print()
        return 0
    else:
        print("\n" + "="*60)
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
