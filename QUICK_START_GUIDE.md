# Quick Reference: Running the API

## 🚀 Start the API

```bash
cd C:\Users\rajmo\OneDrive\fyp
myenv\Scripts\python.exe index.py
```

Server will run on: `http://localhost:8000`

## 📖 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Test Examples

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Model Info
```bash
curl http://localhost:8000/info
```

### 3. Predict Video (Python)
```python
import requests

with open("path/to/video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict_video",
        files={"file": f}
    )
    
result = response.json()
print(f"Prediction: {result['prediction']['label']}")
print(f"Raw Score: {result['prediction']['raw_score']:.4f}")
print(f"Confidence: {result['prediction']['confidence']:.1f}%")
```

### 4. Batch Testing
```bash
# Test on multiple videos
python comprehensive_test.py    # Tests 50 real + 50 fake (96% accuracy)
python detailed_analysis.py     # Tests 20 real + 20 fake
python batch_test.py           # Custom batch test
```

## 📊 Expected Output Format

```json
{
    "video_name": "sample.mp4",
    "file_size_mb": 12.5,
    "faces_detected": 16,
    "processing_time_seconds": 5.2,
    "prediction": {
        "label": "DEEPFAKE",
        "raw_score": 0.9453,
        "confidence": 94.53
    }
}
```

## ⚙️ Configuration

**Model Parameters** (in index.py):
- Input: 16 frames × 224×224 RGB
- Preprocessing: [0, 1] normalization (divide by 255)
- Threshold: 0.5 (>0.5 = DEEPFAKE, <0.5 = REAL)
- Device: Auto-detect GPU (CUDA) or CPU

## 🔧 Troubleshooting

**API won't start?**
- Check Python environment: `myenv\Scripts\python.exe --version`
- Verify port 8000 is not in use: `netstat -ano | findstr 8000`

**Model predictions are wrong?**
- Ensure ImageNet normalization is NOT applied (should be removed)
- Check face_extractor is extracting 16 frames
- Verify input video quality (at least 224×224)

**GPU not being used?**
- Check: `torch.cuda.is_available()` in Python
- Install: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## 📈 Performance

Current model achieves:
- **Real Video Accuracy**: 96.0% (48/50)
- **Deepfake Detection**: 96.0% (48/50)
- **Overall Accuracy**: 96.0%
- **Processing Time**: ~5-10 seconds per video

## 📝 Log Output

Normal startup:
```
Using device: cuda
✅ Model loaded successfully
✅ Face extractor initialized
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Ready to receive requests!

---

**Status**: ✅ Production Ready
**Last Updated**: 2024
