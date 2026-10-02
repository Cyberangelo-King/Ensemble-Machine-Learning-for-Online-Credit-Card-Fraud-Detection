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
import time
import hashlib
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import shap
import joblib
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Header
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
            if MODEL_SHA256:
                with open(MODEL_PATH, "rb") as model_file:
                    digest = hashlib.sha256(model_file.read()).hexdigest()
                if digest != MODEL_SHA256:
                    raise RuntimeError("MODEL_SHA256 integrity check failed")
            state.model = joblib.load(MODEL_PATH)
            if not isinstance(state.model, dict) or "base_learners" not in state.model or "meta_learner" not in state.model or "scaler" not in state.model:
                raise RuntimeError("Invalid model bundle schema")
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

    state.scaler = state.model.get("scaler") if isinstance(state.model, dict) else None

    # --- Dashboard data ---
    if os.path.exists(DATA_PATH):
        try:
            state.data = pd.read_csv(DATA_PATH)
            logger.info("Dataset loaded: %d rows from %s", len(state.data), DATA_PATH)
        except Exception:
            state.data = _generate_dashboard_data()
    else:
        state.data = _generate_dashboard_data()
        logger.info("Research dataset absent; using synthetic dashboard data.")

    # Build explanations against the exact engineered feature space expected by the bundle.
    if state.model is not None and state.data is not None:
        try:
            background = _prepare_raw_dataframe(state.data).sample(
                min(50, len(state.data)), random_state=42
            )
            state.explainer = shap.Explainer(
                lambda matrix: _predict_matrix(np.asarray(matrix)),
                background.to_numpy(dtype=float),
                feature_names=list(background.columns),
                algorithm="permutation",
            )
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
ALLOWED_ORIGINS: List[str] = [x.strip() for x in os.environ.get("ALLOWED_ORIGINS", "").split(",") if x.strip()]
API_KEY = os.environ.get("FRAUD_API_KEY", "").strip()
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
MODEL_SHA256 = os.environ.get("MODEL_SHA256", "").strip().lower()
_rate_buckets: Dict[str, List[float]] = {}

app = FastAPI(
    title="Fraud Detection API",
    description="Ensemble ML-based credit card fraud detection backend.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
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
                "Model is not loaded. Supply a verified trained model bundle before live use."
            ),
        )


def _require_api_key(provided_key: Optional[str]) -> None:
    if API_KEY and provided_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _check_rate_limit(request: Request) -> None:
    now = time.monotonic()
    key = request.client.host if request.client else "unknown"
    bucket = [t for t in _rate_buckets.get(key, []) if now - t < 60]
    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    bucket.append(now)
    _rate_buckets[key] = bucket
    if len(_rate_buckets) > 10000:
        oldest = min(_rate_buckets, key=lambda k: _rate_buckets[k][-1])
        _rate_buckets.pop(oldest, None)


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
# Model/data adapters
# ---------------------------------------------------------------------------

def _generate_dashboard_data(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    values = rng.standard_normal((n_samples, 28))
    amount = rng.lognormal(3.5, 1.4, n_samples).clip(0, 25000)
    times = rng.uniform(0, 172792, n_samples)
    score = -0.8 * values[:, 13] - 0.6 * values[:, 11] - 0.5 * values[:, 9]
    labels = (score + rng.normal(0, 1, n_samples) >= np.quantile(score, 0.95)).astype(int)
    frame = pd.DataFrame(values, columns=[f"V{i}" for i in range(1, 29)])
    frame["Amount"] = amount
    frame["Time"] = times
    frame["Class"] = labels
    return frame


def _prepare_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    out = df[required].copy()
    out["Amount_log"] = np.log1p(out["Amount"].astype(float))
    out["Hour"] = (out["Time"].astype(float) % 86400) / 3600
    feature_names = list(state.model.get("feature_names", []))
    return out[feature_names]


def _predict_matrix(matrix: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(matrix, columns=list(state.model["feature_names"]))
    scaled = state.model["scaler"].transform(frame)
    meta = np.column_stack([
        clf.predict_proba(scaled)[:, 1] for clf in state.model["base_learners"]
    ])
    return state.model["meta_learner"].predict_proba(meta)[:, 1]


def _predict_raw(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probabilities = _predict_matrix(_prepare_raw_dataframe(df).to_numpy(dtype=float))
    threshold = float(state.model.get("optimal_threshold", 0.5))
    return probabilities, (probabilities >= threshold).astype(int)


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
    prediction: int
    threshold: float
    model_loaded: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    readiness: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.  Always returns 200.
    Reports whether the model is currently loaded.
    """
    version = "unknown"
    if isinstance(state.model, dict):
        version = str(state.model.get("model_version", "unknown"))
    return HealthResponse(
        status="ok",
        model_loaded=state.model_loaded,
        model_version=version,
        readiness=state.model_loaded,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(transaction: TransactionFeatures, request: Request, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> PredictionResponse:
    """
    Predict the fraud probability for a single transaction.

    The ``features`` dict should contain all 30 feature keys
    (V1–V28, Amount, Time).  Feature order is inferred from the model's
    ``feature_names_in_`` attribute when available.
    """
    _require_model()
    _require_api_key(x_api_key)
    _check_rate_limit(request)

    try:
        feature_dict = transaction.features
        required = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
        missing = [name for name in required if name not in feature_dict]
        if missing:
            raise HTTPException(status_code=422, detail={"missing_features": missing})
        unknown = sorted(set(feature_dict) - set(required))
        if unknown:
            raise HTTPException(status_code=422, detail={"unknown_features": unknown})
        if not all(np.isfinite(float(v)) for v in feature_dict.values()):
            raise HTTPException(status_code=422, detail="All feature values must be finite numbers.")
        if float(feature_dict["Amount"]) < 0:
            raise HTTPException(status_code=422, detail="Amount must be non-negative.")

        row = dict(feature_dict)
        row["Amount_log"] = float(np.log1p(row["Amount"]))
        row["Hour"] = float((row["Time"] % 86400) / 3600)
        expected_cols = list(state.model.get("feature_names", []))
        df_input = pd.DataFrame([{col: row[col] for col in expected_cols}], columns=expected_cols)
        x_sc = state.model["scaler"].transform(df_input)
        meta_x = np.column_stack([clf.predict_proba(x_sc)[:, 1] for clf in state.model["base_learners"]])
        proba = float(state.model["meta_learner"].predict_proba(meta_x)[0, 1])
        threshold = float(state.model.get("optimal_threshold", 0.5))
        pred = int(proba >= threshold)

        return PredictionResponse(
            fraud_probability=proba,
            prediction=pred,
            threshold=threshold,
            model_loaded=True,
        )
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
    """Return a SHAP explanation using the same model-bundle inference path as /predict."""
    _require_model()
    if state.data is None:
        raise HTTPException(status_code=503, detail="Dashboard dataset is unavailable.")
    if state.explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer is not initialised.")
    if index < 0 or index >= len(state.data):
        raise HTTPException(status_code=404, detail=f"Index {index} out of range.")

    try:
        raw_row = state.data.iloc[[index]]
        engineered = _prepare_raw_dataframe(raw_row)
        shap_values = state.explainer(engineered.to_numpy(dtype=float))
        sv = np.asarray(shap_values.values)
        if sv.ndim > 2:
            sv = sv[..., -1]
        base_values = np.asarray(shap_values.base_values).reshape(-1)
        expected_value = float(base_values[-1]) if base_values.size else 0.0
        probabilities, predictions = _predict_raw(raw_row)
        return {
            "index": index,
            "features": list(engineered.columns),
            "shap_values": sv[0].tolist(),
            "expected_value": expected_value,
            "prediction": int(predictions[0]),
            "fraud_probability": float(probabilities[0]),
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Explain failed for index %d", index)
        raise HTTPException(status_code=500, detail="Explanation generation failed.")


# ---------------------------------------------------------------------------
# WebSocket — live transaction stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """Stream dashboard transactions through the same bundle used by /predict."""
    await websocket.accept()
    logger.info("WebSocket client connected.")
    try:
        while True:
            if state.model is None or state.data is None:
                await websocket.send_json({"error": "Service is not ready", "model_loaded": False})
                await asyncio.sleep(2)
                continue
            try:
                sample = state.data.sample(1, random_state=None)
                probabilities, predictions = _predict_raw(sample)
                payload: Dict[str, Any] = {
                    "transaction_index": int(sample.index[0]),
                    "fraud_probability": float(probabilities[0]),
                    "prediction": int(predictions[0]),
                    "features": {
                        col: float(sample[col].iloc[0])
                        for col in [f"V{i}" for i in range(1, 6)]
                    },
                }
                if "Class" in sample.columns:
                    payload["true_label"] = int(sample["Class"].iloc[0])
                await websocket.send_json(payload)
            except Exception:
                logger.exception("Stream error")
                await websocket.send_json({"error": "Prediction failed"})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception:
        logger.exception("WebSocket stream crashed.")
