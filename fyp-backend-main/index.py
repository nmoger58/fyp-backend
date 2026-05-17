from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
import torch
import cv2
import numpy as np
from torchvision import transforms
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()
import base64
import hashlib
import hmac
from typing import Optional, List
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Motor (async MongoDB driver)
import motor.motor_asyncio

# Import model and utilities
from model import MyModel, FaceExtractor


# ========================================
# Configuration
# ========================================
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "admin123")
AUTH_ROLE = os.getenv("AUTH_ROLE", "user")
MONGODB_URI = os.getenv("MONGODB_URI", "")

if not JWT_SECRET or JWT_SECRET == "change-this-secret":
    print("⚠️  JWT_SECRET not set - using insecure default. Set JWT_SECRET in .env for production.")

if not MONGODB_URI:
    print("❌  MONGODB_URI not set! Please add it to your .env file.")
else:
    print("✅  MONGODB_URI loaded.")


# ========================================
# MongoDB Setup (motor async client)
# ========================================
mongo_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
db = None
users_col = None
history_col = None


async def connect_to_mongo():
    """Initialize the MongoDB connection and collections."""
    global mongo_client, db, users_col, history_col
    try:
        mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
        # Ping to confirm connection
        await mongo_client.admin.command("ping")
        db = mongo_client["deepfake_db"]
        users_col = db["users"]
        history_col = db["scan_history"]

        # Create indexes for performance
        await users_col.create_index("username", unique=True)
        await history_col.create_index("username")
        await history_col.create_index("analyzed_at")

        print("✅  Connected to MongoDB Atlas — database: deepfake_db")
        await _init_default_user_mongo()
    except Exception as e:
        print(f"❌  MongoDB connection failed: {e}")
        print("    Check your MONGODB_URI in .env and ensure your IP is whitelisted in Atlas.")


async def close_mongo():
    """Close the MongoDB connection on shutdown."""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        print("MongoDB connection closed.")


# ========================================
# App Lifespan (startup / shutdown hooks)
# ========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    yield
    # Shutdown
    await close_mongo()


# ========================================
# Initialize FastAPI App
# ========================================
app = FastAPI(
    title="Deepfake Video Detection API",
    description="Advanced deepfake detection using facial extraction and temporal analysis",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================
# JWT / OAuth2
# ========================================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ========================================
# Pydantic Models
# ========================================
class SignupRequest(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    username: str
    role: str
    full_name: Optional[str] = None


class ScanHistoryItem(BaseModel):
    filename: str
    label: str
    confidence: float
    raw_score: float
    is_deepfake: bool
    analyzed_at: str


# ========================================
# Password Utilities
# ========================================
PASSWORD_ITERATIONS = 120_000


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


# ========================================
# MongoDB User Helpers
# ========================================
async def _init_default_user_mongo():
    """Seed the default admin user into MongoDB if it doesn't already exist."""
    if users_col is None:
        return
    existing = await users_col.find_one({"username": AUTH_USERNAME})
    if existing:
        return
    await users_col.insert_one({
        "username": AUTH_USERNAME,
        "full_name": "Admin",
        "password_hash": _hash_password(AUTH_PASSWORD),
        "role": AUTH_ROLE,
        "created_at": datetime.now(timezone.utc),
    })
    print(f"✅  Default admin user '{AUTH_USERNAME}' created in MongoDB.")


async def _get_user(username: str) -> Optional[dict]:
    if users_col is None:
        return None
    return await users_col.find_one({"username": username})


async def _create_user(username: str, password: str, role: str = "user", full_name: Optional[str] = None) -> dict:
    user_doc = {
        "username": username,
        "full_name": full_name or "",
        "password_hash": _hash_password(password),
        "role": role,
        "created_at": datetime.now(timezone.utc),
    }
    await users_col.insert_one(user_doc)
    return user_doc


# ========================================
# JWT Utilities
# ========================================
def create_access_token(subject: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_token(token: str = Depends(oauth2_scheme)):
    """Validate JWT from the Authorization header."""
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


# ========================================
# Auth Endpoints
# ========================================
@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if users_col is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    user = await _get_user(form_data.username)
    if not user or not _verify_password(form_data.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    access_token = create_access_token(form_data.username, user.get("role", "user"))
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in_seconds": JWT_EXPIRE_MINUTES * 60,
        "role": user.get("role", "user"),
        "username": form_data.username,
        "full_name": user.get("full_name", ""),
    }


@app.post("/auth/signup")
async def signup(payload: SignupRequest):
    if users_col is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    if len(payload.password or "") < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    # Check if user already exists
    existing = await _get_user(username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists")

    # Create the user
    full_name = (payload.full_name or "").strip()
    user_doc = await _create_user(username, payload.password, role="user", full_name=full_name)

    access_token = create_access_token(username, "user")
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in_seconds": JWT_EXPIRE_MINUTES * 60,
        "role": "user",
        "username": username,
        "full_name": full_name,
    }


@app.get("/auth/me")
async def auth_me(token_payload=Depends(get_current_token)):
    username = token_payload.get("sub")
    user = await _get_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "username": user["username"],
        "role": user.get("role", "user"),
        "full_name": user.get("full_name", ""),
    }


# ========================================
# History Endpoints
# ========================================
@app.get("/history")
async def get_history(token_payload=Depends(get_current_token)):
    """Get the scan history for the currently logged-in user."""
    if history_col is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    username = token_payload.get("sub")
    cursor = history_col.find(
        {"username": username},
        {"_id": 0}  # exclude MongoDB _id from response
    ).sort("analyzed_at", -1).limit(50)  # newest first, max 50 records

    records = []
    async for doc in cursor:
        # Convert datetime to ISO string for JSON serialization
        if isinstance(doc.get("analyzed_at"), datetime):
            doc["analyzed_at"] = doc["analyzed_at"].isoformat()
        records.append(doc)

    return {"history": records, "total": len(records)}


@app.delete("/history")
async def clear_history(token_payload=Depends(get_current_token)):
    """Clear all scan history for the currently logged-in user."""
    if history_col is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    username = token_payload.get("sub")
    result = await history_col.delete_many({"username": username})
    return {"deleted": result.deleted_count, "message": "History cleared"}


# ========================================
# Set Device
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========================================
# Load Model
# ========================================
model = MyModel(num_frames=16).to(device)

try:
    checkpoint = torch.load(
        "best_model.pth",
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("⚠️  best_model.pth not found - API will start but cannot make predictions")
    model.eval()
except Exception as e:
    print(f"⚠️  Error loading model: {e}")
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

    def preprocess_faces(self, faces):
        """
        Convert numpy array of faces to normalized tensor.
        Args:
            faces: numpy array of shape (num_frames, H, W, 3) with values in [0, 255]
        Returns:
            torch tensor of shape (1, num_frames, 3, 224, 224)
        """
        if faces is None or len(faces) == 0:
            return None

        # Normalize to [0, 1]
        faces = faces.astype(np.float32) / 255.0

        # (N, H, W, 3) → (N, 3, H, W)
        faces = np.transpose(faces, (0, 3, 1, 2))

        processed_frames = []
        for face in faces:
            face_tensor = torch.from_numpy(face)
            processed_frames.append(face_tensor)

        frames_tensor = torch.stack(processed_frames, dim=0)
        frames_tensor = frames_tensor.unsqueeze(0)  # add batch dim
        return frames_tensor


# ========================================
# Prediction Function
# ========================================
def predict_deepfake(frames_tensor):
    with torch.no_grad():
        frames_tensor = frames_tensor.to(device)
        output = model(frames_tensor)
        score = torch.sigmoid(output).item()
        label = "DEEPFAKE" if score > 0.5 else "REAL"
        confidence = score if score > 0.5 else (1 - score)
        return {
            "label": label,
            "raw_score": float(score),
            "confidence": float(confidence),
            "is_deepfake": score > 0.5,
        }


# ========================================
# Predict Video Endpoint  (saves history)
# ========================================
@app.post("/predict_video")
async def predict_video(
    file: UploadFile = File(...),
    token_payload: dict = Depends(get_current_token),
):
    """
    Predict if uploaded video is deepfake.
    Saves the result to scan_history in MongoDB for the logged-in user.
    """
    temp_path = None

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        print(f"📹 Processing video: {file.filename}")

        # Extract faces
        print("🔍 Extracting faces from video...")
        faces = face_extractor.extract_faces_from_video(
            temp_path, num_frames=16, device=device
        )

        if faces is None:
            return JSONResponse(
                status_code=400,
                content={"error": "No faces detected in video", "filename": file.filename},
            )

        print(f"✅ Extracted {len(faces)} faces")

        # Preprocess
        preprocessor = VideoPreprocessor(num_frames=16)
        frames_tensor = preprocessor.preprocess_faces(faces)

        if frames_tensor is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Failed to preprocess frames", "filename": file.filename},
            )

        # Predict
        print("🤖 Running inference...")
        prediction = predict_deepfake(frames_tensor)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"✅ Prediction: {prediction['label']} (confidence: {prediction['confidence']:.2%})")

        # ── Save to MongoDB scan_history ──────────────────────────
        if history_col is not None:
            username = token_payload.get("sub")
            try:
                await history_col.insert_one({
                    "username": username,
                    "filename": file.filename,
                    "label": prediction["label"],
                    "confidence": prediction["confidence"],
                    "raw_score": prediction["raw_score"],
                    "is_deepfake": prediction["is_deepfake"],
                    "analyzed_at": datetime.now(timezone.utc),
                })
                print(f"✅ Scan saved to history for user: {username}")
            except Exception as db_err:
                # Non-fatal — still return the prediction result
                print(f"⚠️  Failed to save scan history: {db_err}")
        # ─────────────────────────────────────────────────────────

        return {
            "filename": file.filename,
            "prediction": prediction,
            "status": "success",
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "filename": file.filename, "status": "error"},
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ========================================
# Health Check
# ========================================
@app.get("/health")
async def health_check():
    mongo_status = "connected" if db is not None else "disconnected"
    return {
        "status": "healthy",
        "device": str(device),
        "model": "MyModel",
        "num_frames": 16,
        "face_size": 224,
        "mongodb": mongo_status,
    }


# ========================================
# Info Endpoint
# ========================================
@app.get("/info")
async def get_info():
    return {
        "name": "Deepfake Video Detection API",
        "version": "2.0",
        "description": "Detects deepfake videos using facial extraction and temporal analysis",
        "endpoints": {
            "POST /auth/login": "Get JWT access token",
            "POST /auth/signup": "Create user and get JWT access token",
            "GET /auth/me": "Validate current JWT and get user profile",
            "POST /predict_video": "Upload video file for deepfake detection",
            "GET /history": "Get scan history for the logged-in user",
            "DELETE /history": "Clear scan history for the logged-in user",
            "GET /health": "Check API health",
            "GET /info": "Get API information",
        },
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }


# ========================================
# Root Endpoint
# ========================================
@app.get("/")
async def root():
    return {
        "message": "Welcome to Deepfake Video Detection API",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
