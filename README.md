# 🎬 Deepfake Video Detection API

A production-ready FastAPI application for detecting deepfake videos using deep learning.

## 🎯 Features

✅ **Advanced Face Extraction**: Uses OpenCV's DNN-based face detector  
✅ **Temporal Analysis**: Processes 16 frames with 3D convolutions  
✅ **GPU Acceleration**: CUDA support for faster inference  
✅ **Attention Mechanisms**: Channel, spatial, and temporal attention  
✅ **REST API**: Easy-to-use HTTP endpoints  
✅ **Interactive Documentation**: Swagger UI and ReDoc built-in  
✅ **Robust Error Handling**: Comprehensive error messages  

---

## 🏗️ Architecture

### Model: Enhanced EfficientNet-B0
```
Input Video (16 frames × 224×224×3)
    ↓
Face Extraction (OpenCV DNN)
    ↓
Preprocessing (Normalization)
    ↓
EfficientNet-B0 Backbone (Feature Extraction)
    ↓
3D Temporal Convolutions (Temporal Modeling)
    ↓
Attention Mechanisms (Focus Learning)
    ↓
Classification Head (Binary Classification)
    ↓
Output: Real/Deepfake Prediction + Confidence
```

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB RAM minimum (16GB recommended)

### Setup Steps

1. **Activate Virtual Environment**
   ```bash
   .\myenv\Scripts\Activate.ps1
   ```

2. **Install Dependencies** (if not already installed)
   ```bash
   pip install fastapi uvicorn torch torchvision opencv-python numpy
   ```

3. **Verify Model File**
   - Ensure `best_model.pth` exists in the root directory
   - Should be in: `C:\Users\rajmo\OneDrive\fyp\best_model.pth`

---

## 🚀 Running the API

### Start the Server
```bash
cd C:\Users\rajmo\OneDrive\fyp
python -m uvicorn index:app --reload --host 0.0.0.0 --port 8000
```

### Expected Output
```
INFO:     Uvicorn running on http://127.0.0.1:8000
✅ Model loaded successfully
✅ Face extractor initialized
INFO:     Application startup complete
```

---

## 🌐 API Endpoints

### 1. **Health Check**
Check if API is running and model is loaded

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "model": "MyModel",
  "num_frames": 16,
  "face_size": 224
}
```

---

### 2. **API Info**
Get detailed API information

```bash
curl http://localhost:8000/info
```

---

### 3. **Predict Deepfake** ⭐ Main Endpoint

Upload a video file for deepfake detection

```bash
curl -X POST "http://localhost:8000/predict_video" \
  -F "file=@test_video.mp4"
```

**Response:**
```json
{
  "filename": "test_video.mp4",
  "prediction": {
    "label": "REAL",
    "raw_score": 0.2345,
    "confidence": 0.7655,
    "is_deepfake": false
  },
  "status": "success"
}
```

---

## 🧪 Testing the API

### Method 1: Using Python Script
```bash
python test_api.py video.mp4
```

### Method 2: Using Python Client
```python
import requests

url = "http://localhost:8000/predict_video"
with open("video.mp4", "rb") as f:
    response = requests.post(url, files={"file": f})

result = response.json()
print(f"Prediction: {result['prediction']['label']}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
```

### Method 3: Using Swagger UI
1. Open browser: `http://localhost:8000/docs`
2. Click "Try it out" on `/predict_video`
3. Upload a video file
4. Click "Execute"

---

## 📊 Understanding the Predictions

### Output Interpretation

| Score | Label | Meaning |
|-------|-------|---------|
| < 0.5 | REAL | Video is likely authentic |
| ≥ 0.5 | DEEPFAKE | Video is likely deepfake |

### Confidence Score
- **0.5 - 0.6**: Low confidence (close to decision boundary)
- **0.6 - 0.8**: Medium confidence
- **0.8 - 1.0**: High confidence

### Example Results
```
{
  "label": "DEEPFAKE",
  "confidence": 0.92,    # 92% confident
  "is_deepfake": true
}

{
  "label": "REAL",
  "confidence": 0.87,    # 87% confident it's real
  "is_deepfake": false
}
```

---

## 🔄 Video Processing Pipeline

The API performs these steps automatically:

1. **Video Upload** (MP4, MOV, AVI, FLV)
   - Temporary storage in system temp directory
   - Automatic cleanup after processing

2. **Face Extraction**
   - Detects faces using OpenCV DNN detector
   - Extracts 16 faces from evenly spaced frames
   - Resizes faces to 224×224 pixels

3. **Preprocessing**
   ```
   Raw Faces (16 × 224×224×3)
       ↓
   Convert to [0, 1] range
       ↓
   Apply ImageNet Normalization
   (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ↓
   Tensor Shape: (1, 16, 3, 224, 224)
   ```

4. **Inference**
   - Forward pass through MyModel
   - Output: Single value from sigmoid activation
   - Threshold at 0.5 for binary classification

5. **Result Formatting**
   - Compute confidence score
   - Return prediction with metadata

---

## ⚙️ Configuration

### Model Parameters
- **Num Frames**: 16 (fixed)
- **Face Size**: 224×224 pixels
- **Batch Size**: 1 per request
- **Device**: Auto-detect (CUDA if available)

### Device Selection
The API automatically uses:
- **GPU (CUDA)** if available and working
- **CPU** as fallback

Check device: `http://localhost:8000/health`

---

## 🐛 Troubleshooting

### Problem: "No faces detected in video"
**Cause**: Video lacks clear, visible faces  
**Solution**:
- Use videos with clear face visibility
- Ensure good lighting in video
- Try a different video

### Problem: "CUDA out of memory"
**Cause**: Video too large or insufficient GPU memory  
**Solution**:
- Use CPU mode (will be slower)
- Try shorter videos
- Close other GPU applications

### Problem: "Connection refused"
**Cause**: API server not running  
**Solution**:
```bash
python -m uvicorn index:app --reload --port 8000
```

### Problem: Module import errors
**Cause**: Missing dependencies  
**Solution**:
```bash
pip install -r requirements.txt
```

### Problem: Model file not found
**Cause**: `best_model.pth` missing or wrong location  
**Solution**:
- Place `best_model.pth` in: `C:\Users\rajmo\OneDrive\fyp\`
- Verify file exists and is not corrupted

---

## 📈 Performance

### Inference Time (per video)
- **With GPU**: 2-5 seconds
- **With CPU**: 10-30 seconds

### Memory Requirements
- **GPU**: 2-4 GB VRAM
- **CPU**: 4-8 GB RAM

### Throughput
- **Single GPU**: ~12 videos/minute
- **Single CPU**: ~2 videos/minute

---

## 📚 API Documentation

### Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

---

## 🔐 Security Considerations

⚠️ **Current Version**: For development/testing only

For production deployment:
1. Add authentication (API keys, OAuth2)
2. Implement rate limiting
3. Add request validation
4. Use HTTPS
5. Add logging and monitoring
6. Implement request queuing
7. Add disk space monitoring

---

## 📝 File Structure

```
fyp/
├── index.py                    # Main FastAPI application
├── best_model.pth              # Trained model weights
├── test_api.py                 # Python test client
├── API_TESTING_GUIDE.md        # Detailed testing guide
├── requirements.txt            # Python dependencies
│
└── model/
    ├── __init__.py             # Model definition (MyModel)
    ├── face_extractor.py       # Face extraction class
    ├── video_augmentation.py   # Video preprocessing
    ├── preprocessor.py         # Preprocessing utilities
    └── enhanced_resnet_face.py # Alternative model architectures
```

---

## 🤝 Contributing

To improve the model:
1. Prepare new training data
2. Fine-tune the model
3. Save weights as `best_model.pth`
4. Test with `test_api.py`

---

## 📄 License

This project is part of FYP (Final Year Project)

---

## ❓ Support

For issues or questions:
1. Check `API_TESTING_GUIDE.md` for detailed troubleshooting
2. Review error messages in console output
3. Verify model file and dependencies are correct
4. Check GPU/CUDA availability: `python test_api.py --health`

---

## 🎓 Model Details

**Architecture**: Enhanced EfficientNet-B0  
**Training Data**: Deepfake and Real videos  
**Framework**: PyTorch 2.x  
**Backbone**: EfficientNet-B0 (ImageNet pretrained)  
**Input**: Video (16 frames × 224×224)  
**Output**: Deepfake Probability (0-1)  

---

## 🚀 Next Steps

1. **Start API**: `python -m uvicorn index:app --reload`
2. **Test Health**: `curl http://localhost:8000/health`
3. **Test with Video**: `python test_api.py video.mp4`
4. **View Docs**: Open `http://localhost:8000/docs`

Happy Deepfake Detection! 🎬✨
#   f y p - b a c k e n d  
 