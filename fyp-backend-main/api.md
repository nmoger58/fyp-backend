curl -X 'POST' \
  'http://localhost:8000/predict_video' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@index.html - Calculator - Visual Studio Code 2025-10-03 19-39-16.mp4;type=video/mp4'

expected outputs:
 {
  "error": "No faces detected in video",
  "filename": "index.html - Calculator - Visual Studio Code 2025-10-03 19-39-16.mp4"
}

{
  "filename": "Title 2025-10-16 23-59-36.mp4",
  "prediction": {
    "label": "REAL",
    "raw_score": 0.06101468950510025,
    "confidence": 0.9389853104948997,
    "is_deepfake": false
  },
  "status": "success"
}

{
  "filename": "Watch Bigg Boss Episode 9 on JioHotstar - Brave 2025-10-11 00-26-36.mp4",
  "prediction": {
    "label": "DEEPFAKE",
    "raw_score": 0.946861743927002,
    "confidence": 0.946861743927002,
    "is_deepfake": true
  },
  "status": "success"
}