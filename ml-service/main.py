"""
main.py — FastAPI ML Service for Guitar Chord Detection
========================================================
Loads the trained chord model and exposes a /predict endpoint.

Endpoints:
  GET  /           → health check
  GET  /classes    → list of supported chord names
  POST /predict    → accepts {landmarks: [63 floats]}, returns {chord, confidence, all_probs}

Usage:
  uvicorn main:app --reload --port 8001
"""

import logging
import os
import pickle
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "model", "chord_model.pkl")

# ── Global model state ─────────────────────────────────────────────────────────
model_data: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model_data
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Run train.py first.")
    else:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        classes = model_data.get("classes", [])
        acc     = model_data.get("accuracy", 0.0)
        name    = model_data.get("model_name", "unknown")
        logger.info(f"Loaded model: {name} | Accuracy: {acc*100:.1f}% | Classes: {classes}")
    yield
    model_data.clear()


app = FastAPI(
    title="Guitar Chord Detector API",
    description="Real-time guitar chord recognition from MediaPipe hand landmarks.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow frontend and Node backend ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────

class LandmarkPayload(BaseModel):
    landmarks: list[float]

    @field_validator("landmarks")
    @classmethod
    def check_length(cls, v):
        if len(v) != 63:
            raise ValueError(f"Expected 63 landmark values (21 pts × x,y,z), got {len(v)}")
        return v


class PredictionResponse(BaseModel):
    chord:      str
    confidence: float
    all_probs:  dict[str, float]
    model_loaded: bool


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    loaded = bool(model_data)
    classes = model_data.get("classes", [])
    return {
        "status": "ok",
        "model_loaded": loaded,
        "model_name": model_data.get("model_name", "none"),
        "accuracy": model_data.get("accuracy", None),
        "supported_chords": classes,
        "message": "POST /predict with {landmarks: [63 floats]}",
    }


@app.get("/classes", tags=["Info"])
async def get_classes():
    """Return list of chord labels the model can recognize."""
    if not model_data:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    return {"classes": model_data["classes"]}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(payload: LandmarkPayload):
    """
    Accept 63 normalized landmark floats and return the predicted chord name
    along with confidence scores for all classes.
    """
    if not model_data:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first, then restart the service."
        )

    clf     = model_data["model"]
    classes = model_data["classes"]

    X = np.array(payload.landmarks, dtype=np.float32).reshape(1, -1)

    # Predict probabilities
    try:
        proba = clf.predict_proba(X)[0]
    except AttributeError:
        # Fallback if classifier doesn't support predict_proba
        pred_idx = clf.predict(X)[0]
        proba = np.zeros(len(classes))
        proba[pred_idx] = 1.0

    best_idx    = int(np.argmax(proba))
    chord       = classes[best_idx]
    confidence  = float(proba[best_idx])
    all_probs   = {c: round(float(p), 4) for c, p in zip(classes, proba)}

    logger.debug(f"Predicted: {chord} ({confidence*100:.1f}%)")

    return PredictionResponse(
        chord=chord,
        confidence=round(confidence, 4),
        all_probs=all_probs,
        model_loaded=True,
    )


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "model_loaded": bool(model_data)}
