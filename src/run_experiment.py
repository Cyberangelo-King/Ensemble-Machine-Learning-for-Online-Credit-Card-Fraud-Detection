"""
Optimized End-to-End Credit Card Fraud Detection Pipeline

Models:
- Logistic Regression
- Random Forest
- XGBoost
- Stacking Ensemble (Logistic Regression meta-model)

Key improvements:
- Reduced CV folds (3 instead of 5)
- Reduced RandomizedSearch iterations
- Light SMOTE (5%)
- No duplicate stacking training
- Optional SHAP (post-analysis only)
- Faster XGBoost config
"""

from __future__ import annotations

import os
import time
import json
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from dataclasses import dataclass

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from xgboost import XGBClassifier

import shap

# =========================
# CONFIG
# =========================
RANDOM_STATE = 42
FAST_RUN = True

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# =========================
# UTILITIES
# =========================
def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data(path="data/creditcard.csv"):
    df = pd.read_csv(path)
    return df


# =========================
# MODELS
# =========================
def build_models():

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])

    rf = RandomForestClassifier(
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    xgb = XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0
    )

    search_spaces = {
        "lr": {
            "model": lr,
            "params": {
                "clf__C": np.logspace(-2, 2, 10),
                "clf__penalty": ["l1", "l2"]
            }
        },

        "rf": {
            "model": rf,
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "class_weight": [None, "balanced"]
            }
        },

        "xgb": {
            "model": xgb,
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 4, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        }
    }

    meta = LogisticRegression(max_iter=1000, solver="liblinear")

    stacking = StackingClassifier(
        estimators=[],
        final_estimator=meta,
        cv=3,
        n_jobs=-1,
        stack_method="predict_proba"
    )

    return search_spaces, stacking


# =========================
# TRAINING FUNCTION
# =========================
def tune(model, params, X, y, name, n_iter=20):

    print(f"🚀 Tuning {name}...")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0
    )

    search.fit(X, y)

    print(f"✅ {name} best AUPRC: {search.best_score_:.4f}")
    return search.best_estimator_


# =========================
# MAIN PIPELINE
# =========================
def run():

    set_seed()

    # folders
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # load
    df = load_data()
    X = df[FEATURES]
    y = df["Class"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # SMOTE (LIGHT)
    smote = SMOTE(sampling_strategy=0.05, random_state=RANDOM_STATE)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # models
    search_spaces, stacking = build_models()

    tuned_models = {}

    n_iter = 20 if FAST_RUN else 50

    # tuning loop
    for name, obj in search_spaces.items():
        best_model = tune(
            obj["model"],
            obj["params"],
            X_train,
            y_train,
            name,
            n_iter=n_iter
        )
        tuned_models[name] = best_model

    # build stacking
    stacking.estimators = list(tuned_models.items())

    print("🚀 Training stacking model...")
    stacking.fit(X_train, y_train)

    # save model
    joblib.dump(stacking, "artifacts/stacking_model.joblib")

    # =========================
    # EVALUATION
    # =========================
    print("📊 Evaluating...")

    y_proba = stacking.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "AUPRC": average_precision_score(y_test, y_proba),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba)
    }

    print(json.dumps(metrics, indent=2))

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # =========================
    # PLOTS
    # =========================

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("figures/confusion_matrix.png", dpi=300)
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="Stacking")
    plt.plot([0,1],[0,1],"--")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("figures/roc_curve.png", dpi=300)
    plt.close()

    # PR Curve
    p, r, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(r, p, label="Stacking")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("figures/pr_curve.png", dpi=300)
    plt.close()

    # =========================
    # OPTIONAL SHAP (SLOW)
    # =========================
    print("🧠 Running SHAP (optional)...")

    explainer = shap.TreeExplainer(tuned_models["xgb"])
    shap_values = explainer.shap_values(X_test.sample(1000, random_state=42))

    shap.summary_plot(shap_values, X_test.sample(1000), show=False)
    plt.savefig("figures/shap.png", dpi=300)
    plt.close()

    print("✅ Pipeline completed!")


if __name__ == "__main__":
    run()
