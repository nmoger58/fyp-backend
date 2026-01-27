# Deepfake Video Detection API - Testing Guide

## Installation & Setup

### 1. Activate Virtual Environment
```bash
.\myenv\Scripts\Activate.ps1
```

### 2. Run the API
```bash
cd C:\Users\rajmo\OneDrive\fyp
python -m uvicorn index:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
✅ Model loaded successfully
✅ Face extractor initialized
Uvicorn running on http://127.0.0.1:8000
```

---

## API Endpoints

### 1. Health Check
**Endpoint:** `GET /health`

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda" or "cpu",
  "model": "MyModel",
  "num_frames": 16,
  "face_size": 224
}
```

---

### 2. API Info
**Endpoint:** `GET /info`

```bash
curl http://localhost:8000/info
```

---

### 3. Predict Deepfake (Main Endpoint)
**Endpoint:** `POST /predict_video`

#### Using cURL
```bash
curl -X POST "http://localhost:8000/predict_video" \
  -F "file=@path/to/video.mp4"
```

#### Using Python
```python
import requests

url = "http://localhost:8000/predict_video"
files = {"file": open("path/to/video.mp4", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

#### Response Example
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

## Understanding the Response

### Prediction Labels
- **"DEEPFAKE"**: Video is detected as deepfake (score > 0.5)
- **"REAL"**: Video is detected as real (score ≤ 0.5)

### Confidence Score
- Represents how confident the model is in its prediction
- Range: 0.0 to 1.0 (0-100%)
- Higher = more confident

### Raw Score
- The actual output from the model's sigmoid activation
- Range: 0.0 to 1.0
- > 0.5 = Deepfake
- ≤ 0.5 = Real

---

## Processing Pipeline

The API performs the following steps:

1. **Video Upload**: Accepts MP4, MOV, AVI, FLV files
2. **Face Extraction**: Extracts 16 faces evenly spaced across the video using OpenCV's DNN detector
3. **Face Preprocessing**:
   - Resize faces to 224×224 pixels
   - Normalize pixel values to [0, 1]
   - Apply ImageNet normalization (mean, std)
4. **Inference**: Pass frames through MyModel
5. **Prediction**: Apply sigmoid activation and threshold at 0.5

---

## Example Full Test

### Python Test Script
```python
import requests
import json

def test_deepfake_api(video_path):
    """Test the deepfake detection API"""
    
    url = "http://localhost:8000/predict_video"
    
    with open(video_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    
    result = response.json()
    
    print("\n" + "="*50)
    print("DEEPFAKE DETECTION RESULT")
    print("="*50)
    print(f"Filename: {result['filename']}")
    print(f"Status: {result['status']}")
    print(f"\nPrediction: {result['prediction']['label']}")
    print(f"Confidence: {result['prediction']['confidence']:.2%}")
    print(f"Raw Score: {result['prediction']['raw_score']:.4f}")
    print("="*50 + "\n")

# Usage
test_deepfake_api("path/to/test_video.mp4")
```

---

## Troubleshooting

### Issue: "No faces detected in video"
- **Cause**: Video doesn't contain clear faces
- **Solution**: Use videos with visible faces, good lighting, and clear face detection

### Issue: "CUDA out of memory"
- **Cause**: Video is too large or model uses too much memory
- **Solution**: Reduce batch size, use smaller videos, or use CPU

### Issue: Model not loading
- **Cause**: `best_model.pth` not found or corrupted
- **Solution**: Ensure `best_model.pth` is in the root directory (`C:\Users\rajmo\OneDrive\fyp\`)

### Issue: Import errors
- **Cause**: Missing dependencies
- **Solution**: Run `pip install -r requirements.txt`

---

## Performance Tips

1. **Use GPU**: Model will automatically use CUDA if available
2. **Batch Processing**: API processes one video at a time; use load balancing for multiple requests
3. **Video Format**: MP4 format is fastest
4. **Face Detection**: Videos with clear, frontal faces process faster

---

## Model Architecture

**MyModel** (Enhanced EfficientNet-B0):
- **Backbone**: EfficientNet-B0 (ImageNet pretrained)
- **Temporal Processing**: 3D Convolutions (3 layers, 512 channels)
- **Attention Mechanisms**:
  - Temporal Attention
  - Channel Attention
  - Spatial Attention
- **Classification Head**: 4-layer MLP with LayerNorm and Dropout

---

## API Documentation

Visit the interactive Swagger UI:
```
http://localhost:8000/docs
```

Or ReDoc:
```
http://localhost:8000/redoc
```

