"""End-to-end fraud detection experiment pipeline.

Implements a stacking ensemble:
- Base learners: LogisticRegression, RandomForestClassifier, XGBClassifier
- Meta learner: LogisticRegression

Includes:
- Dataset loading
- Train/test split with stratification
- Preprocessing
- SMOTE at 1:10 minority:majority on training fold only
- RandomizedSearchCV (50 iterations) for each base learner
- Evaluation metrics: AUPRC, F1, MCC, confusion matrix, ROC/PR curves
- Latency benchmarking (ms/transaction)
- SHAP TreeExplainer on tuned XGBoost
- Publication-quality figure exports (300 DPI)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET_RESULTS = {
    "auprc": 0.903,
    "f1": 0.881,
    "mcc": 0.884,
    "mean_latency_ms": 0.074,
}

@dataclass
class ExperimentArtifacts:
    """Container for trained model and metric outputs."""

    metrics: dict
    model: object


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load Kaggle credit card fraud dataset CSV.

    Expected columns include Time, Amount, V1..V28, Class.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Follow README dataset instructions."
        )
    df = pd.read_csv(data_path)
    expected = {"Class", "Amount", "Time"}
    if not expected.issubset(df.columns):
        raise ValueError("Dataset schema mismatch: required columns missing.")
    return df


def build_models() -> tuple[dict, StackingClassifier]:
    """Create base learner search spaces and stacking model."""
    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)),
        ]
    )
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
    )

    search_spaces = {
        "lr": {
            "model": lr_pipe,
            "params": {
                "clf__C": np.logspace(-2, 2, 20),
                "clf__penalty": ["l1", "l2"],
            },
        },
        "rf": {
            "model": rf,
            "params": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced"],
            },
        },
        "xgb": {
            "model": xgb,
            "params": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.3],
            },
        },
    }

    meta = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
    stacking = StackingClassifier(
        estimators=[],
        final_estimator=meta,
        stack_method="predict_proba",
        passthrough=False,
        cv=5,
        n_jobs=-1,
    )
    return search_spaces, stacking


def tune_model(name, model, params, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=50,
        scoring="average_precision",
        n_jobs=-1,
        cv=cv,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    search.fit(X, y)
    return name, search.best_estimator_


def plot_and_save(fig_path: Path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def run(args):
    set_seed()
    sns.set_theme(style="whitegrid")
    out_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.data_path))
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    smote = SMOTE(sampling_strategy=0.1, random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    search_spaces, stacking = build_models()
    tuned = {}
    for name, payload in search_spaces.items():
        learner_name, best_model = tune_model(name, payload["model"], payload["params"], X_train_res, y_train_res)
        tuned[learner_name] = best_model

    stacking.estimators = [(k, v) for k, v in tuned.items()]
    stacking.fit(X_train_res, y_train_res)

    y_proba = stacking.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auprc = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    sample = X_test.iloc[:1000]
    start = time.perf_counter()
    _ = stacking.predict_proba(sample)
    elapsed = time.perf_counter() - start
    mean_latency_ms = (elapsed / len(sample)) * 1000

    metrics = {
        "auprc": auprc,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "mean_latency_ms": mean_latency_ms,
        "target_reported": TARGET_RESULTS,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Stacking Ensemble")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plot_and_save(fig_dir / "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Stacking (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plot_and_save(fig_dir / "roc_curve.png")

    p, r, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(r, p, label=f"Stacking (AUPRC={auprc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plot_and_save(fig_dir / "pr_curve.png")

    # SHAP on tuned XGBoost base learner
    xgb_model = tuned["xgb"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test.iloc[:1000])
    shap.summary_plot(shap_values, X_test.iloc[:1000], show=False)
    plot_and_save(fig_dir / "shap_summary.png")

    # simple performance comparison plot
    comp = pd.DataFrame(
        [
            {"model": "Stacking", "AUPRC": auprc, "F1": f1, "MCC": mcc},
        ]
    )
    comp_m = comp.melt(id_vars="model", var_name="metric", value_name="value")
    plt.figure(figsize=(7, 5))
    sns.barplot(data=comp_m, x="metric", y="value", hue="model")
    plt.ylim(0, 1)
    plt.title("Model Performance Comparison")
    plot_and_save(fig_dir / "model_comparison.png")

    print("Experiment completed.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/creditcard.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--fig-dir", default="figures")
    run(parser.parse_args())
