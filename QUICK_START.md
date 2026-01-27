# 🚀 Quick Start - Deepfake Detection API

## ⚡ 30-Second Setup

### 1. Start the API
```bash
cd C:\Users\rajmo\OneDrive\fyp
.\myenv\Scripts\Activate.ps1
python start_api.py
```

### 2. Test in Another Terminal
```bash
python test_api.py your_video.mp4
```

Done! 🎉

---

## 🌐 API is Now Running

- **API URL**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📍 Three Ways to Use

### 1️⃣ Python Script (Easiest)
```bash
python test_api.py video.mp4
```

### 2️⃣ cURL Command
```bash
curl -X POST "http://localhost:8000/predict_video" \
  -F "file=@video.mp4"
```

### 3️⃣ Web Browser
Open: http://localhost:8000/docs  
Click "Try it out" on `/predict_video` endpoint

---

## 📊 Understanding Results

```json
{
  "prediction": {
    "label": "REAL",           // DEEPFAKE or REAL
    "confidence": 0.87,        // 0-1 (0-100%)
    "is_deepfake": false       // true/false
  }
}
```

- **Confidence > 0.8**: Highly confident
- **Confidence 0.5-0.8**: Moderately confident  
- **Confidence < 0.5**: Low confidence

---

## 📁 Project Files

```
fyp/
├── index.py                      # Main API (RUN THIS)
├── best_model.pth               # Trained weights (89 MB)
├── test_api.py                  # Test client
├── start_api.py                 # Startup script
├── verify_setup.py              # Verification tool
│
├── README.md                    # Full documentation
├── API_TESTING_GUIDE.md        # Detailed guide
├── INTEGRATION_SUMMARY.md      # What changed
├── QUICK_START.md              # This file
│
└── model/
    ├── __init__.py             # Model definition
    └── face_extractor.py       # Face detection
```

---

## 🧪 Verification

Check if everything is working:
```bash
python verify_setup.py
```

Should show: **8/8 tests passed ✅**

---

## 🎯 Complete Example

```python
import requests

# Test with a video
url = "http://localhost:8000/predict_video"
with open("test_video.mp4", "rb") as f:
    response = requests.post(url, files={"file": f})

result = response.json()
print(f"Prediction: {result['prediction']['label']}")
print(f"Confidence: {result['prediction']['confidence']:.1%}")
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No faces detected" | Use video with clear faces |
| "Connection refused" | Run `python start_api.py` first |
| "Module not found" | Activate venv: `.\myenv\Scripts\Activate.ps1` |
| "CUDA out of memory" | API will use CPU automatically |

---

## 📈 What's Happening Behind the Scenes

```
Your Video
    ↓
Extract 16 Faces (evenly spaced frames)
    ↓
Resize to 224×224 pixels each
    ↓
Apply ImageNet normalization
    ↓
Send through EfficientNet-B0 + 3D Convolutions
    ↓
Apply Attention Mechanisms (temporal, channel, spatial)
    ↓
Classification Head → Probability Score
    ↓
Result: REAL or DEEPFAKE (+ confidence)
```

---

## ✅ Checklist

- [ ] Virtual environment activated
- [ ] API running: `python start_api.py`
- [ ] Health check passes: http://localhost:8000/health
- [ ] Can upload and predict video
- [ ] Results make sense (0-1 probability)

---

## 📚 Next Steps

1. **Try different videos** → See how confident the model is
2. **Check logs** → Console output shows processing steps
3. **Read full docs** → README.md for detailed info
4. **Integrate into app** → Use the test_api.py as a template

---

## 💡 Performance Tips

- **First run**: Slower (model loading)
- **With GPU**: 2-5 seconds per video
- **With CPU**: 10-30 seconds per video
- **Multiple requests**: Start multiple API instances

---

## 🎓 Model Architecture

- **Backbone**: EfficientNet-B0 (ImageNet pretrained)
- **Temporal**: 3D Convolutions
- **Attention**: Temporal + Channel + Spatial
- **Input**: 16 frames × 224×224
- **Output**: Deepfake probability (0-1)

---

## 🎬 Sample Test

```bash
# Start API
python start_api.py

# In another terminal, test with a video
python test_api.py path/to/video.mp4

# Expected output:
# ============================================================
# DEEPFAKE DETECTION RESULT
# ============================================================
# Prediction: REAL
# Confidence: 87.00%
# ============================================================
```

---

## 🔗 Useful Links

- **Interactive Docs**: http://localhost:8000/docs
- **API Schema**: http://localhost:8000/openapi.json
- **Health Check**: http://localhost:8000/health
- **Info Endpoint**: http://localhost:8000/info

---

## ❓ Questions?

Check these in order:
1. **Quick issues**: See Troubleshooting above
2. **Testing help**: API_TESTING_GUIDE.md
3. **How it works**: README.md
4. **Integration details**: INTEGRATION_SUMMARY.md

---

**You're all set! Happy deepfake detecting! 🎉**
