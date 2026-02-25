# VeriSynth - Deepfake Detection Frontend

A modern React application for detecting deepfakes using a backend AI API. Built with Vite, React, and Lucide icons.

## Project Structure

```
src/
├── components/
│   ├── LandingPage.jsx      # Landing page with features and CTA
│   ├── AuthPage.jsx          # Login and signup forms
│   ├── Dashboard.jsx         # Main upload and analysis interface
│   ├── ReadyPage.jsx         # File preview before analysis
│   ├── AnalyzingPage.jsx     # Loading state during analysis
│   └── ResultPage.jsx        # Analysis results and report
├── utils/
│   └── api.js                # Backend API service
├── App.jsx                   # Main app orchestrator
├── App.css                   # App styling
├── index.css                 # Global styles
└── main.jsx                  # Entry point
```

## Features

- **Landing Page**: Showcases platform features and benefits
- **Authentication**: Login/Signup with hardcoded demo credentials
  - Username: `nmoger58`
  - Password: `Nagu@123`
- **Video Upload**: Drag-and-drop or click to upload video files
- **Real-time Analysis**: Integration with backend deepfake detection API
- **Results Dashboard**: Detailed analysis results with confidence scores
- **Report Download**: Generate and download analysis reports

## Backend API Integration

### Endpoints Used

#### Health Check
```
GET /health
Response:
{
  "status": "healthy",
  "device": "cpu",
  "model": "MyModel",
  "num_frames": 16,
  "face_size": 224
}
```

#### Video Prediction
```
POST /predict_video
Content-Type: multipart/form-data
Body: { file: VideoFile }

Success Response (Authentic):
{
  "filename": "video.mp4",
  "prediction": {
    "label": "REAL",
    "raw_score": 0.061,
    "confidence": 0.939,
    "is_deepfake": false
  },
  "status": "success"
}

Success Response (Deepfake):
{
  "filename": "video.mp4",
  "prediction": {
    "label": "DEEPFAKE",
    "raw_score": 0.947,
    "confidence": 0.947,
    "is_deepfake": true
  },
  "status": "success"
}

Error Response:
{
  "error": "No faces detected in video",
  "filename": "video.mp4"
}
```

## Component Architecture

### App.jsx (Orchestrator)
- Manages global state (currentPage, uploadedFile, analysisResult)
- Handles API calls and state transitions
- Provides callback functions to child components
- Manages CSS variables and styles

### LandingPage.jsx
- Marketing landing page
- Navigation to auth flow
- Feature highlights
- Platform statistics

### AuthPage.jsx
- Toggle between Login and Signup modes
- Email/password form handling
- Basic validation
- Navigation between forms

### Dashboard.jsx
- File upload interface
- Recent scans history
- Analysis statistics
- Navigation bar with user profile

### ReadyPage.jsx
- File preview before analysis
- File name and size display
- Start analysis button
- Cancel option

### AnalyzingPage.jsx
- Loading spinner animation
- Status messages
- Minimal UI during processing

### ResultPage.jsx
- Success/Error state handling
- Confidence score visualization
- Key indicators list
- Download report functionality
- New analysis button

### api.js (Service Layer)
- Centralized API communication
- Health check function
- Video prediction function
- Error handling

## Getting Started

### Prerequisites
- Node.js 16+
- Backend API running on `http://localhost:8000`

### Installation

```bash
cd my-react-app
npm install
```

### Development

```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### Build

```bash
npm run build
```

## Configuration

API base URL can be modified in `src/utils/api.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000';
```

## Styling

The application uses:
- **Tailwind CSS classes** for responsive design
- **CSS Custom Properties** (variables) for theming
- **Lucide React icons** for UI elements

Color scheme:
- Primary: `#4CADE4`
- Success: `#12C99B`
- Danger: `#FF4C4C`
- Warning: `#F4AA2A`
- Neutral 900: `#1A1D1F` (background)
- Neutral 600: `#303235`

## Data Flow

1. User lands on **Landing Page**
2. Clicks "Get Started" → **Auth Page**
3. Logs in with demo credentials → **Dashboard**
4. Uploads video file → **Ready Page**
5. Clicks "Start Analysis" → **Analyzing Page** (API call)
6. API returns result → **Result Page**
7. Can download report or analyze new video

## Error Handling

- Network errors are caught and displayed in the result page
- Backend errors (e.g., "No faces detected") are handled gracefully
- User-friendly error messages guide next actions

## Browser Support

Modern browsers with ES6 support:
- Chrome/Edge latest
- Firefox latest
- Safari latest

## License

This project is part of a Final Year Project (FYP) for deepfake detection.
