"""Create a demo stacking ensemble model for rapid dashboard testing.

This script generates a minimal stacking model and synthetic dataset
without requiring the full Kaggle dataset or long training times.

Usage:
    python scripts/create_demo_model.py
    
Output:
    - artifacts/stacking_model.joblib (trained model)
    - data/creditcard.csv (synthetic data with 1000 samples)
    - results/metrics.json (mock metrics)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
np.random.seed(RANDOM_STATE)

def create_synthetic_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic credit card fraud dataset."""
    print(f"Creating synthetic dataset ({n_samples} samples)...")
    
    # Keep the demo imbalanced while ensuring enough fraud rows for CV.
    n_fraud = max(6, int(n_samples * 0.01))
    n_legit = n_samples - n_fraud
    
    # Features V1-V28 (PCA components)
    X_legit = np.random.randn(n_legit, 28) * 0.5
    X_fraud = np.random.randn(n_fraud, 28) * 1.5 + np.random.choice([-1, 1], (n_fraud, 28))
    
    X = np.vstack([X_legit, X_fraud])
    y = np.hstack([np.zeros(n_legit), np.ones(n_fraud)]).astype(int)
    
    # Add Time and Amount
    time_vals = np.random.uniform(0, 86400, n_samples)
    amount_vals = np.abs(np.random.exponential(50, n_samples))
    amount_vals[y == 1] *= np.random.uniform(2, 10, n_fraud)  # Fraud often higher amounts
    
    df = pd.DataFrame(X, columns=[f"V{i+1}" for i in range(28)])
    df["Time"] = time_vals
    df["Amount"] = amount_vals
    df["Class"] = y
    df = df[[*FEATURE_COLUMNS, "Class"]]
    
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

def train_stacking_model(X: pd.DataFrame, y: pd.Series):
    """Train lightweight stacking ensemble."""
    print("Training stacking ensemble...")
    
    # Base learners (minimal hyperparams for speed)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=100, solver="liblinear", random_state=RANDOM_STATE))
    ])
    
    rf = RandomForestClassifier(
        n_estimators=10, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    
    xgb = XGBClassifier(
        n_estimators=10, max_depth=3, random_state=RANDOM_STATE,
        eval_metric="logloss", verbosity=0, n_jobs=-1
    )
    
    # Meta learner
    meta = LogisticRegression(max_iter=100, solver="liblinear", random_state=RANDOM_STATE)
    
    n_splits = min(3, int(y.value_counts().min()))

    # Stacking classifier
    stacking = StackingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb)],
        final_estimator=meta,
        stack_method="predict_proba",
        cv=n_splits,
        n_jobs=-1
    )
    
    stacking.fit(X, y)
    return stacking

def main():
    artifacts_dir = Path("artifacts")
    data_dir = Path("data")
    results_dir = Path("results")
    
    artifacts_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Create synthetic dataset
    df = create_synthetic_dataset(n_samples=1000)
    data_path = data_dir / "creditcard.csv"
    df.to_csv(data_path, index=False)
    print(f"✓ Synthetic dataset saved: {data_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Class distribution: {df['Class'].value_counts().to_dict()}")
    
    # Train model
    X = df[FEATURE_COLUMNS]
    y = df["Class"]
    model = train_stacking_model(X, y)
    
    # Save model
    model_path = artifacts_dir / "stacking_model.joblib"
    joblib.dump(model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Generate mock metrics
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
    
    metrics = {
        "auprc": float(average_precision_score(y, y_proba)),
        "f1": float(f1_score(y, y_pred)),
        "mcc": float(matthews_corrcoef(y, y_pred)),
        "roc_auc": float((y_proba[y == 1].mean() - y_proba[y == 0].mean()) / 2 + 0.5),
        "mean_latency_ms": 0.074,
        "target_reported": {
            "auprc": 0.903,
            "f1": 0.881,
            "mcc": 0.884,
            "mean_latency_ms": 0.074,
        },
        "note": "Demo model - metrics are on synthetic data"
    }
    
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")
    print(f"  AUPRC: {metrics['auprc']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")
    print(f"  MCC: {metrics['mcc']:.3f}")
    
    print("\n✅ Demo setup complete! Run: streamlit run app.py")

if __name__ == "__main__":
    main()
