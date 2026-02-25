from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import torch
import cv2
import numpy as np
from torchvision import transforms
import tempfile
import os
import json
import base64
import hashlib
import hmac
from typing import Optional
from pydantic import BaseModel

# Import model and utilities
from model import MyModel, FaceExtractor

# ========================================
# Initialize FastAPI App
# ========================================
app = FastAPI(
    title="Deepfake Video Detection API",
    description="Advanced deepfake detection using facial extraction and temporal analysis"
)

# Enable CORS (optional, adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# JWT Authentication
# ========================================
security = HTTPBearer()

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
AUTH_USERNAME = os.getenv("AUTH_USERNAME")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")
AUTH_ROLE = os.getenv("AUTH_ROLE", "user")

if not JWT_SECRET:
    JWT_SECRET = "change-this-secret"
    print("??  JWT_SECRET not set - using insecure default secret. "
          "Set JWT_SECRET in production.")

if not AUTH_USERNAME or not AUTH_PASSWORD:
    AUTH_USERNAME = "admin"
    AUTH_PASSWORD = "admin123"
    print("??  AUTH_USERNAME/AUTH_PASSWORD not set - using development credentials.")


class LoginRequest(BaseModel):
    username: str
    password: str


class SignupRequest(BaseModel):
    username: str
    password: str


USERS_FILE = os.getenv("AUTH_USERS_FILE", "users.json")
PASSWORD_ITERATIONS = 120_000


def _load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return (
        f"pbkdf2_sha256${PASSWORD_ITERATIONS}$"
        f"{base64.b64encode(salt).decode('utf-8')}$"
        f"{base64.b64encode(digest).decode('utf-8')}"
    )


def _verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algorithm, iter_count, salt_b64, digest_b64 = encoded_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64.encode("utf-8"))
        expected = base64.b64decode(digest_b64.encode("utf-8"))
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            int(iter_count),
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def _init_default_user():
    users = _load_users()
    if AUTH_USERNAME in users:
        return
    users[AUTH_USERNAME] = {
        "password_hash": _hash_password(AUTH_PASSWORD),
        "role": AUTH_ROLE,
    }
    _save_users(users)
    print(f"Initialized default auth user in {USERS_FILE}: {AUTH_USERNAME}")


def create_access_token(subject: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


@app.post("/auth/login")
async def login(payload: LoginRequest):
    users = _load_users()
    user = users.get(payload.username)
    if not user or not _verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    access_token = create_access_token(payload.username, user.get("role", AUTH_ROLE))
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in_seconds": JWT_EXPIRE_MINUTES * 60,
        "role": user.get("role", AUTH_ROLE),
        "username": payload.username,
    }


@app.post("/auth/signup")
async def signup(payload: SignupRequest):
    username = payload.username.strip()
    if not username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is required",
        )
    if len(payload.password or "") < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters",
        )

    users = _load_users()
    if username in users:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        )

    users[username] = {
        "password_hash": _hash_password(payload.password),
        "role": "user",
    }
    _save_users(users)

    access_token = create_access_token(username, "user")
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in_seconds": JWT_EXPIRE_MINUTES * 60,
        "role": "user",
        "username": username,
    }


def get_current_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Validate JWT from the Authorization header.
    """
    token = credentials.credentials if credentials else None
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        if not username or not role:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/auth/me")
async def auth_me(token_payload=Depends(get_current_token)):
    return {
        "username": token_payload.get("sub"),
        "role": token_payload.get("role"),
    }
# ========================================
# Set Device
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================================
# Load Model
# ========================================
model = MyModel(num_frames=16).to(device)

# Load checkpoint
try:
    checkpoint = torch.load(
        "best_model.pth",
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Ensure all dropout layers are disabled for inference
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("⚠️  best_model.pth not found - API will start but cannot make predictions")
    print("   Place best_model.pth in the project root directory")
    model.eval()
except Exception as e:
    print(f"⚠️  Error loading model: {e}")
    print("   The API will start but predictions may fail")
    model.eval()

# ========================================
# Initialize FaceExtractor
# ========================================
face_extractor = FaceExtractor(face_size=224, device=device)
print("✅ Face extractor initialized")

# ========================================
# Preprocessing Pipeline
# ========================================
class VideoPreprocessor:
    def __init__(self, num_frames=16):
        self.num_frames = num_frames
        # NOTE: Model trained WITHOUT ImageNet normalization
        # Only standard [0,1] normalization applied
    
    def preprocess_faces(self, faces):
        """
        Convert numpy array of faces to normalized tensor
        Args:
            faces: numpy array of shape (num_frames, H, W, 3) with values in [0, 255]
        Returns:
            torch tensor of shape (1, num_frames, 3, 224, 224)
        """
        if faces is None or len(faces) == 0:
            return None
        
        # Convert to float and normalize to [0, 1]
        faces = faces.astype(np.float32) / 255.0
        
        # Convert from (N, H, W, 3) to (N, 3, H, W)
        faces = np.transpose(faces, (0, 3, 1, 2))
        
        # NO ImageNet normalization applied - model expects [0, 1] range
        processed_frames = []
        for face in faces:
            face_tensor = torch.from_numpy(face)
            processed_frames.append(face_tensor)
        
        # Stack into batch: (num_frames, 3, H, W)
        frames_tensor = torch.stack(processed_frames, dim=0)
        
        # Add batch dimension: (1, num_frames, 3, H, W)
        frames_tensor = frames_tensor.unsqueeze(0)
        
        return frames_tensor

# ========================================
# Prediction Function
# ========================================
def predict_deepfake(frames_tensor):
    """
    Predict if video is deepfake
    Args:
        frames_tensor: torch tensor of shape (1, num_frames, 3, 224, 224)
    Returns:
        dict with prediction and confidence
    """
    with torch.no_grad():
        frames_tensor = frames_tensor.to(device)
        output = model(frames_tensor)
        
        # Apply sigmoid to get probability
        score = torch.sigmoid(output).item()
        
        # Determine label (> 0.5 = Fake)
        label = "DEEPFAKE" if score > 0.5 else "REAL"
        confidence = score if score > 0.5 else (1 - score)
        
        return {
            "label": label,
            "raw_score": float(score),
            "confidence": float(confidence),
            "is_deepfake": score > 0.5
        }

# ========================================
# API Endpoint
# ========================================
@app.post("/predict_video", dependencies=[Depends(get_current_token)])
async def predict_video(file: UploadFile = File(...)):
    """
    Predict if uploaded video is deepfake
    
    Process:
    1. Extract 16 faces from evenly spaced frames
    2. Preprocess faces with normalization
    3. Pass through trained model
    4. Return prediction with confidence score
    """
    temp_path = None
    
    try:
        # -------- Save uploaded file temporarily --------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_path = tmp.name
            content = await file.read()
            tmp.write(content)
        
        print(f"📹 Processing video: {file.filename}")
        
        # -------- Extract faces from video --------
        print("🔍 Extracting faces from video...")
        faces = face_extractor.extract_faces_from_video(
            temp_path,
            num_frames=16,
            device=device
        )
        
        if faces is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "No faces detected in video",
                    "filename": file.filename
                }
            )
        
        print(f"✅ Extracted {len(faces)} faces")
        
        # -------- Preprocess faces --------
        print("🎨 Preprocessing frames...")
        preprocessor = VideoPreprocessor(num_frames=16)
        frames_tensor = preprocessor.preprocess_faces(faces)
        
        if frames_tensor is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Failed to preprocess frames",
                    "filename": file.filename
                }
            )
        
        print(f"📊 Frame tensor shape: {frames_tensor.shape}")
        
        # -------- Get prediction --------
        print("🤖 Running inference...")
        prediction = predict_deepfake(frames_tensor)
        
        # Clear CUDA cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        print(f"✅ Prediction: {prediction['label']} (confidence: {prediction['confidence']:.2%})")
        
        return {
            "filename": file.filename,
            "prediction": prediction,
            "status": "success"
        }
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "filename": file.filename,
                "status": "error"
            }
        )
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Clear CUDA cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

# ========================================
# Health Check Endpoint
# ========================================
@app.get("/health")
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "device": str(device),
        "model": "MyModel",
        "num_frames": 16,
        "face_size": 224
    }

# ========================================
# Info Endpoint
# ========================================
@app.get("/info")
async def get_info():
    """Get API information"""
    return {
        "name": "Deepfake Video Detection API",
        "version": "1.0",
        "description": "Detects deepfake videos using facial extraction and temporal analysis",
        "endpoints": {
            "POST /auth/login": "Get JWT access token",
            "POST /auth/signup": "Create user and get JWT access token",
            "GET /auth/me": "Validate current JWT and get principal details",
            "POST /predict_video": "Upload video file for deepfake detection",
            "GET /health": "Check API health",
            "GET /info": "Get API information"
        },
        "device": str(device),
        "cuda_available": torch.cuda.is_available()
    }

# ========================================
# Root Endpoint
# ========================================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Deepfake Video Detection API",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


_init_default_user()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

