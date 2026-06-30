"""
dashboard/backend/main.py
FastAPI backend for the Credit Card Fraud Detection dashboard.

Endpoints
---------
GET  /health           — Health check (always available, even if model not loaded)
POST /predict          — Predict fraud probability for a single transaction
GET  /explain/{index}  — Return SHAP explanation for a transaction by index
WS   /ws/stream        — WebSocket stream of live transaction predictions
"""

from __future__ import annotations

import logging
import os
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap
import joblib
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (resolved relative to the repo root — adjust as needed)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "results", "stacking_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "results", "stacking_model.pkl")  # scaler embedded in model bundle
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

# ---------------------------------------------------------------------------
# Application state — populated in lifespan, may be None if files are missing
# ---------------------------------------------------------------------------
class AppState:
    model: Optional[Any] = None
    scaler: Optional[Any] = None
    data: Optional[pd.DataFrame] = None
    explainer: Optional[Any] = None
    model_loaded: bool = False


state = AppState()

# ---------------------------------------------------------------------------
# Lifespan — graceful degradation if artefacts are absent
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artefacts at startup; degrade gracefully if any are missing."""
    # --- Model ---
    if os.path.exists(MODEL_PATH):
        try:
            state.model = joblib.load(MODEL_PATH)
            logger.info("Model loaded from %s", MODEL_PATH)
        except Exception as exc:
            logger.warning("Failed to load model from %s: %s", MODEL_PATH, exc)
            state.model = None
    else:
        logger.warning(
            "Model file not found at %s. Prediction endpoints will return 503. "
            "Run scripts/create_demo_model.py to generate it.",
            MODEL_PATH,
        )

    # --- Scaler ---
    if os.path.exists(SCALER_PATH):
        try:
            state.scaler = joblib.load(SCALER_PATH)
            logger.info("Scaler loaded from %s", SCALER_PATH)
        except Exception as exc:
            logger.warning("Failed to load scaler from %s: %s", SCALER_PATH, exc)
            state.scaler = None
    else:
        logger.warning("Scaler file not found at %s.", SCALER_PATH)

    # --- Data ---
    if os.path.exists(DATA_PATH):
        try:
            state.data = pd.read_csv(DATA_PATH)
            logger.info("Dataset loaded: %d rows from %s", len(state.data), DATA_PATH)
        except Exception as exc:
            logger.warning("Failed to load dataset from %s: %s", DATA_PATH, exc)
            state.data = None
    else:
        logger.warning(
            "Dataset not found at %s. Explain endpoint will return 503.", DATA_PATH
        )

    # --- SHAP explainer (only if model is present) ---
    if state.model is not None:
        try:
            # Use a small background dataset when data is available, else None
            if state.data is not None:
                feature_cols = _get_feature_columns(state.data)
                background = state.data[feature_cols].sample(
                    min(100, len(state.data)), random_state=42
                )
                state.explainer = shap.Explainer(state.model, background)
            else:
                state.explainer = shap.Explainer(state.model)
            logger.info("SHAP explainer initialised.")
        except Exception as exc:
            logger.warning("Could not initialise SHAP explainer: %s", exc)
            state.explainer = None

    state.model_loaded = state.model is not None
    logger.info("Startup complete. model_loaded=%s", state.model_loaded)

    yield  # ← application runs here

    # Shutdown
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS: List[str] = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(
    title="Fraud Detection API",
    description="Ensemble ML-based credit card fraud detection backend.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return feature columns (all except 'Class')."""
    return [c for c in df.columns if c != "Class"]


def _require_model() -> None:
    """Raise 503 if the model is not loaded."""
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                "Run scripts/create_demo_model.py and restart the service."
            ),
        )


def _safe_expected_value(explainer: Any) -> float:
    """
    Safely extract the scalar expected value from a SHAP explainer.

    XGBoost binary classifiers may return:
      - a float                   -> use directly
      - a list / np.ndarray       -> take the last element (positive class)
      - a nested list/array       -> flatten and take the last element

    Returns a plain Python float in all cases.
    """
    ev = explainer.expected_value

    if isinstance(ev, (int, float)):
        return float(ev)

    # Convert to a flat numpy array for uniform handling
    ev_arr = np.asarray(ev).flatten()

    if ev_arr.size == 0:
        return 0.0
    if ev_arr.size == 1:
        return float(ev_arr[0])

    # Multi-class or binary: return the last element (positive class)
    return float(ev_arr[-1])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TransactionFeatures(BaseModel):
    """Input schema for /predict.

    Accepts a dictionary of feature names to values.  The canonical feature
    set is V1–V28 plus Amount and Time (30 features total).
    """
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: int  # 0 = legitimate, 1 = fraud
    model_loaded: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.  Always returns 200.
    Reports whether the model is currently loaded.
    """
    return HealthResponse(status="ok", model_loaded=state.model_loaded)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(transaction: TransactionFeatures) -> PredictionResponse:
    """
    Predict the fraud probability for a single transaction.

    The ``features`` dict should contain all 30 feature keys
    (V1–V28, Amount, Time).  Feature order is inferred from the model's
    ``feature_names_in_`` attribute when available.
    """
    _require_model()

    try:
        # Build a DataFrame so column ordering is handled correctly
        feature_dict = transaction.features
        df_input = pd.DataFrame([feature_dict])

        # Align columns to model's expected order if possible
        if hasattr(state.model, "feature_names_in_"):
            expected_cols = list(state.model.feature_names_in_)
            # Fill any missing cols with 0
            for col in expected_cols:
                if col not in df_input.columns:
                    df_input[col] = 0.0
            df_input = df_input[expected_cols]

        # Scale Amount/Time if scaler is available
        if state.scaler is not None:
            scale_cols = [c for c in ["Amount", "Time"] if c in df_input.columns]
            if scale_cols:
                df_input[scale_cols] = state.scaler.transform(df_input[scale_cols])

        proba = state.model.predict_proba(df_input)[0, 1]
        pred = int(proba >= 0.5)

        return PredictionResponse(
            fraud_probability=float(proba),
            prediction=pred,
            model_loaded=True,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


@app.get("/explain/{index}", tags=["inference"])
async def explain(index: int) -> Dict[str, Any]:
    """
    Return a SHAP explanation for a transaction at the given dataset index.
    """
    _require_model()

    if state.data is None:
        raise HTTPException(
            status_code=503,
            detail="Dataset is not loaded; explanation unavailable.",
        )
    if state.explainer is None:
        raise HTTPException(
            status_code=503,
            detail="SHAP explainer is not initialised.",
        )
    if index < 0 or index >= len(state.data):
        raise HTTPException(
            status_code=404,
            detail=f"Index {index} out of range (dataset has {len(state.data)} rows).",
        )

    try:
        feature_cols = _get_feature_columns(state.data)
        row = state.data.iloc[[index]][feature_cols]

        shap_values = state.explainer(row)

        # shap_values.values shape may be (1, n_features) or (1, n_features, n_classes)
        sv = shap_values.values
        if sv.ndim == 3:
            # Multi-class / binary output: take positive class (last)
            sv = sv[:, :, -1]
        shap_list = sv[0].tolist()

        expected_value = _safe_expected_value(state.explainer)

        return {
            "index": index,
            "features": feature_cols,
            "shap_values": shap_list,
            "expected_value": expected_value,
            "prediction": int(state.model.predict(row)[0]),
            "fraud_probability": float(state.model.predict_proba(row)[0, 1]),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Explain failed for index %d: %s", index, exc)
        raise HTTPException(
            status_code=500, detail=f"Explanation error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# WebSocket — live transaction stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint that streams live (synthetic) transaction predictions.

    The client connects; the server sends one JSON message per second with a
    randomly sampled transaction and its fraud probability.  The connection
    stays open until the client disconnects.
    """
    await websocket.accept()
    logger.info("WebSocket client connected.")

    try:
        while True:
            if state.model is None or state.data is None:
                await websocket.send_json(
                    {"error": "Model or data not loaded", "model_loaded": False}
                )
                await asyncio.sleep(2)
                continue

            try:
                feature_cols = _get_feature_columns(state.data)
                sample = state.data[feature_cols].sample(1, random_state=None)
                proba = float(state.model.predict_proba(sample)[0, 1])
                prediction = int(proba >= 0.5)
                true_label = int(
                    state.data.iloc[sample.index[0]]["Class"]
                ) if "Class" in state.data.columns else None

                payload: Dict[str, Any] = {
                    "transaction_index": int(sample.index[0]),
                    "fraud_probability": proba,
                    "prediction": prediction,
                    "features": {
                        col: float(sample[col].iloc[0]) for col in feature_cols[:5]
                    },
                }
                if true_label is not None:
                    payload["true_label"] = true_label

                await websocket.send_json(payload)
            except Exception as exc:
                logger.warning("Stream error: %s", exc)
                await websocket.send_json({"error": str(exc)})

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.exception("WebSocket stream crashed: %s", exc)
