import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("artifacts/stacking_model.joblib")
DATA_PATH = Path("data/creditcard.csv")

# Global state
state = {
    "model": None,
    "data": None,
    "test_data": None,
    "explainer": None
}

class PredictionRequest(BaseModel):
    features: dict

class PredictionResponse(BaseModel):
    probability: float
    prediction: int
    risk_level: str
    latency_ms: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    if not DATA_PATH.exists():
        logger.error(f"Data not found at {DATA_PATH}")
        raise RuntimeError(f"Data not found at {DATA_PATH}")

    logger.info("Loading model...")
    state["model"] = joblib.load(MODEL_PATH)

    logger.info("Loading data...")
    state["data"] = pd.read_csv(DATA_PATH)

    # Simple split to mimic unseen data
    state["test_data"] = state["data"].sample(frac=0.2, random_state=42)

    logger.info("Initializing SHAP explainer...")
    # Robustly find a tree-based learner for SHAP
    tree_model = None
    if hasattr(state["model"], "named_estimators_"):
        for name in ["xgb", "rf", "random_forest"]:
            if name in state["model"].named_estimators_:
                tree_model = state["model"].named_estimators_[name]
                logger.info(f"Using {name} for SHAP explanations")
                break

    if tree_model:
        state["explainer"] = shap.TreeExplainer(tree_model)
    else:
        logger.warning("No tree-based base learner found for SHAP. Falling back to KernelExplainer (slower).")
        # In a real app, you might want to sample data here for KernelExplainer

    yield
    # Shutdown logic
    state.clear()

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

# CORS setup for production
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_risk_level(prob: float) -> str:
    if prob < 0.2: return "Low"
    if prob < 0.6: return "Medium"
    return "High"

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    df = pd.DataFrame([request.features])
    t0 = time.perf_counter()
    prob = float(state["model"].predict_proba(df)[:, 1][0])
    latency = (time.perf_counter() - t0) * 1000

    return {
        "probability": prob,
        "prediction": int(prob >= 0.5),
        "risk_level": get_risk_level(prob),
        "latency_ms": latency
    }

@app.get("/explain/{index}")
async def explain(index: int):
    if state["data"] is None or index not in state["data"].index:
         # Try locating by position if index is not label
         if index < 0 or index >= len(state["data"]):
            raise HTTPException(status_code=404, detail="Transaction not found")
         row = state["data"].iloc[[index]].drop(columns=["Class"])
    else:
         row = state["data"].loc[[index]].drop(columns=["Class"])

    if state["explainer"] is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not initialized")

    shap_values = state["explainer"].shap_values(row)

    # Handle multi-output (classification)
    if isinstance(shap_values, list):
        # For binary classification, typically index 1 is the positive class
        vals = shap_values[1][0].tolist() if len(shap_values) > 1 else shap_values[0][0].tolist()
    else:
        vals = shap_values[0].tolist()

    return {
        "base_value": float(state["explainer"].expected_value[1]) if isinstance(state["explainer"].expected_value, (list, np.ndarray)) else float(state["explainer"].expected_value),
        "shap_values": vals,
        "feature_names": row.columns.tolist(),
        "feature_values": row.iloc[0].tolist()
    }

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")

    stream_task = None
    try:
        while True:
            # Wait for control messages
            data_raw = await websocket.receive_text()
            control = json.loads(data_raw)

            action = control.get("action")
            interval = control.get("interval", 1.0)

            if action == "start":
                if stream_task:
                    stream_task.cancel()
                stream_task = asyncio.create_task(stream_transactions(websocket, interval))
            elif action == "stop":
                if stream_task:
                    stream_task.cancel()
                    stream_task = None
            elif action == "speed":
                if stream_task:
                    stream_task.cancel()
                    stream_task = asyncio.create_task(stream_transactions(websocket, interval))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        if stream_task:
            stream_task.cancel()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if stream_task:
            stream_task.cancel()

async def stream_transactions(websocket: WebSocket, interval: float):
    # Use test_data for streaming
    indices = state["test_data"].index.tolist()
    np.random.shuffle(indices)

    for idx in indices:
        row = state["test_data"].loc[idx]
        features = row.drop("Class").to_dict()
        actual = int(row["Class"])

        # Predict
        df_row = pd.DataFrame([features])
        t0 = time.perf_counter()
        prob = float(state["model"].predict_proba(df_row)[:, 1][0])
        latency = (time.perf_counter() - t0) * 1000

        message = {
            "index": int(idx),
            "features": features,
            "actual": actual,
            "prediction": int(prob >= 0.5),
            "probability": prob,
            "risk_level": get_risk_level(prob),
            "latency_ms": latency,
            "timestamp": time.time()
        }

        await websocket.send_json(message)
        await asyncio.sleep(interval)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
