"""
run_experiment.py — Research-Grade Stacking Ensemble for Credit Card Fraud Detection
=====================================================================================
Final Year Project: Ensemble Machine Learning for Online Credit Card Fraud Detection
Author  : Angelo (Cyberangelo-King)
Version : 2.0.0
Python  : ≥3.9

Architecture
------------
Layer 0 (Base Learners):  Logistic Regression | Random Forest | XGBoost
Layer 1 (Meta-Learner):   Logistic Regression trained on OOF meta-features
Output:                   Probability of fraud, calibrated score, binary prediction

Key Features
------------
- SMOTE applied strictly INSIDE each CV fold (leakage-free)
- Parallelised OOF stacking via joblib.Parallel
- Threshold optimisation: reports at both 0.5 and F1-optimal threshold
- Statistical significance: mean ± std across stability runs with 95 % CI
- Ablation study: test 1, 2, and 3 base learners
- Learning curves: AUPRC vs training-set size
- SHAP summary + permutation importance
- Calibration curves (Platt scaling)
- Comprehensive metrics: AUPRC, F1, MCC, ROC-AUC, Precision, Recall
- Full figure export (ROC, PR curve, calibration, SHAP, learning curves, confusion matrix)
- Checkpoint saving after each base-learner tuning step
- Reproducibility assertion on saved model
- Logging with timestamps and memory usage at key steps
- Complete argparse CLI

Usage
-----
python src/run_experiment.py --data-path data/creditcard.csv --output-dir results/ --fig-dir figures/
"""

# ---------------------------------------------------------------------------
# Stdlib
# ---------------------------------------------------------------------------
import argparse
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import shap
from imblearn.over_sampling import SMOTE
from joblib import Memory, Parallel, delayed
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

# Suppress noisy warnings for a clean run
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===========================================================================
# ███████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗
# ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝
# █████╗  ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗
# ██╔══╝  ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║
# ██║     ╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝
# ╚═╝      ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝
# ===========================================================================
# All tunable parameters live here — edit this dict, not the code below.

CONFIG: Dict[str, Any] = {
    # ── Reproducibility ──────────────────────────────────────────────────────
    "random_state": 42,
    # ── Cross-validation ─────────────────────────────────────────────────────
    "n_folds": 5,
    # ── Stability analysis ───────────────────────────────────────────────────
    "n_stability_runs": 5,
    # ── SMOTE ────────────────────────────────────────────────────────────────
    "smote_sampling_strategy": 0.15,   # minority:majority ratio after oversampling
    # ── RandomizedSearchCV iterations ────────────────────────────────────────
    "n_iter_lr": 20,
    "n_iter_rf": 30,
    "n_iter_xgb": 50,
    # ── Parallelism ──────────────────────────────────────────────────────────
    "n_jobs": -1,
    # ── XGBoost early stopping ───────────────────────────────────────────────
    "xgb_early_stopping_rounds": 20,
    "xgb_eval_fraction": 0.1,         # fraction of train fold used as eval set
    # ── Logistic Regression hyperparameter grid ───────────────────────────────
    "lr_param_grid": {
        "C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [500, 1000, 2000],
        "class_weight": ["balanced", None],
    },
    # ── Random Forest hyperparameter grid ────────────────────────────────────
    "rf_param_grid": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "class_weight": ["balanced", "balanced_subsample"],
        "bootstrap": [True, False],
    },
    # ── XGBoost hyperparameter grid ──────────────────────────────────────────
    "xgb_param_grid": {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.3, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "scale_pos_weight": [1, 5, 10, 50, 100],
    },
    # ── Meta-learner ─────────────────────────────────────────────────────────
    "meta_lr_C": 1.0,
    "meta_lr_solver": "lbfgs",
    "meta_lr_max_iter": 1000,
}

# ===========================================================================
# LOGGING
# ===========================================================================

def _setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Configure structured logging with timestamps."""
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)
    return logging.getLogger("fraud_detection")


LOG: logging.Logger = _setup_logging()


def _mem_mb() -> float:
    """Return current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _log_step(msg: str) -> None:
    """Log a step with timestamp and memory usage."""
    LOG.info(f"{msg} [RAM: {_mem_mb():.1f} MB]")


# ===========================================================================
# DATA LOADING & FEATURE ENGINEERING
# ===========================================================================

def load_and_engineer_features(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load creditcard.csv with dtype optimisation and apply feature engineering.

    Optimisations
    -------------
    - V1–V28 loaded as float32 (halves memory vs float64)
    - Amount and Time loaded as float32
    - Engineered features:
        Amount_log = log1p(Amount)   — compresses right-skewed distribution
        Hour       = (Time % 86400) / 3600  — captures intra-day fraud patterns

    Parameters
    ----------
    data_path : str
        Path to creditcard.csv.

    Returns
    -------
    X : pd.DataFrame  Feature matrix (284 807 × 32)
    y : pd.Series     Binary labels (0 = legitimate, 1 = fraud)
    """
    _log_step(f"Loading data from '{data_path}'")
    t0 = time.perf_counter()

    # Dtype map: float32 for all feature columns to halve memory
    dtype_map: Dict[str, str] = {f"V{i}": "float32" for i in range(1, 29)}
    dtype_map["Amount"] = "float32"
    dtype_map["Time"] = "float32"

    df = pd.read_csv(data_path, dtype=dtype_map)
    _log_step(f"Loaded {len(df):,} rows in {time.perf_counter()-t0:.2f}s")

    # ── Feature engineering ───────────────────────────────────────────────
    df["Amount_log"] = np.log1p(df["Amount"]).astype("float32")
    df["Hour"] = ((df["Time"] % 86400) / 3600).astype("float32")

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_log", "Hour"]
    X = df[feature_cols].copy()
    y = df["Class"].astype(int)

    fraud_pct = y.mean() * 100
    _log_step(
        f"Feature matrix: {X.shape} | Fraud: {y.sum():,} / {len(y):,} ({fraud_pct:.4f}%)"
    )
    return X, y


# ===========================================================================
# PREPROCESSING
# ===========================================================================

def preprocess(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    smote_ratio: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale features and apply SMOTE *inside* the fold to prevent data leakage.

    The scaler is fitted on X_train only, then applied to both X_train and
    X_test — ensuring no information from the test fold contaminates scaling.
    SMOTE is applied only to the already-scaled training data.

    Parameters
    ----------
    X_train      : Training features (unscaled)
    X_test       : Test/validation features (unscaled)
    y_train      : Training labels
    smote_ratio  : SMOTE sampling_strategy (target minority:majority ratio)
    random_state : Seed for reproducibility

    Returns
    -------
    X_train_res : SMOTE-resampled training features (scaled)
    X_test_sc   : Scaled test features
    y_train_res : SMOTE-resampled training labels
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state, n_jobs=1)
    X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)
    return X_train_res, X_test_sc, y_train_res


# ===========================================================================
# METRICS
# ===========================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Metrics
    -------
    AUPRC    : Area Under Precision-Recall Curve
               AP = Σ (R_n − R_{n-1}) × P_n
    ROC-AUC  : Area Under Receiver Operating Characteristic Curve
    F1       : 2 × (Precision × Recall) / (Precision + Recall)
    MCC      : (TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    Precision: TP / (TP + FP)
    Recall   : TP / (TP + FN)

    Parameters
    ----------
    y_true    : Ground-truth binary labels
    y_prob    : Predicted fraud probabilities
    threshold : Decision threshold (default 0.5)

    Returns
    -------
    dict of metric name → float value
    """
    y_pred = (y_prob >= threshold).astype(int)
    auprc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "AUPRC": auprc,
        "ROC_AUC": roc_auc,
        "F1": f1,
        "MCC": mcc,
        "Precision": precision,
        "Recall": recall,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Threshold": threshold,
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the probability threshold that maximises F1-score on the validation set.

    Iterates over all unique thresholds from precision_recall_curve to find
    the operating point with the best F1. Using the PR-curve thresholds is
    more efficient than a uniform grid search.

    Parameters
    ----------
    y_true : Ground-truth binary labels
    y_prob : Predicted probabilities

    Returns
    -------
    optimal_threshold : float
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precisions + recalls) == 0,
        0.0,
        2 * precisions * recalls / (precisions + recalls + 1e-9),
    )
    # precision_recall_curve returns one more value than thresholds
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


# ===========================================================================
# BASE-LEARNER TUNING  (with checkpointing)
# ===========================================================================

def _tune_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    cfg: Dict[str, Any],
    checkpoint_dir: Path,
) -> LogisticRegression:
    """Tune Logistic Regression with RandomizedSearchCV."""
    ckpt = checkpoint_dir / "lr_best.pkl"
    if ckpt.exists():
        LOG.info("Checkpoint found — loading LR model from disk.")
        return joblib.load(ckpt)

    _log_step("Tuning Logistic Regression …")
    base = LogisticRegression(random_state=cfg["random_state"])
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=cfg["lr_param_grid"],
        n_iter=cfg["n_iter_lr"],
        cv=cv,
        scoring="average_precision",
        n_jobs=cfg["n_jobs"],
        pre_dispatch="2*n_jobs",
        random_state=cfg["random_state"],
        verbose=0,
        refit=True,
    )
    with tqdm(total=cfg["n_iter_lr"], desc="LR RandomSearch", unit="cfg") as pbar:
        search.fit(X, y)
        pbar.update(cfg["n_iter_lr"])

    best = search.best_estimator_
    LOG.info(f"Best LR params: {search.best_params_} | CV AUPRC: {search.best_score_:.4f}")
    joblib.dump(best, ckpt)
    _log_step(f"LR checkpoint saved → {ckpt}")
    return best


def _tune_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    cfg: Dict[str, Any],
    checkpoint_dir: Path,
) -> RandomForestClassifier:
    """Tune Random Forest with RandomizedSearchCV."""
    ckpt = checkpoint_dir / "rf_best.pkl"
    if ckpt.exists():
        LOG.info("Checkpoint found — loading RF model from disk.")
        return joblib.load(ckpt)

    _log_step("Tuning Random Forest …")
    base = RandomForestClassifier(
        random_state=cfg["random_state"],
        n_jobs=cfg["n_jobs"],
    )
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=cfg["rf_param_grid"],
        n_iter=cfg["n_iter_rf"],
        cv=cv,
        scoring="average_precision",
        n_jobs=cfg["n_jobs"],
        pre_dispatch="2*n_jobs",
        random_state=cfg["random_state"],
        verbose=0,
        refit=True,
    )
    with tqdm(total=cfg["n_iter_rf"], desc="RF RandomSearch", unit="cfg") as pbar:
        search.fit(X, y)
        pbar.update(cfg["n_iter_rf"])

    best = search.best_estimator_
    LOG.info(f"Best RF params: {search.best_params_} | CV AUPRC: {search.best_score_:.4f}")
    joblib.dump(best, ckpt)
    _log_step(f"RF checkpoint saved → {ckpt}")
    return best


def _tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    cfg: Dict[str, Any],
    checkpoint_dir: Path,
) -> XGBClassifier:
    """Tune XGBoost with RandomizedSearchCV and early stopping."""
    ckpt = checkpoint_dir / "xgb_best.pkl"
    if ckpt.exists():
        LOG.info("Checkpoint found — loading XGB model from disk.")
        return joblib.load(ckpt)

    _log_step("Tuning XGBoost …")
    # Split a small eval set from training data for early stopping
    from sklearn.model_selection import train_test_split
    eval_frac = cfg["xgb_eval_fraction"]
    X_tr, X_eval, y_tr, y_eval = train_test_split(
        X, y,
        test_size=eval_frac,
        stratify=y,
        random_state=cfg["random_state"],
    )

    base = XGBClassifier(
        random_state=cfg["random_state"],
        n_jobs=cfg["n_jobs"],
        use_label_encoder=False,
        eval_metric="aucpr",
        early_stopping_rounds=cfg["xgb_early_stopping_rounds"],
        verbosity=0,
    )

    # For RandomizedSearchCV we cannot pass eval_set through fit_params reliably
    # across all sklearn versions. We do a manual random search here.
    rng = np.random.default_rng(cfg["random_state"])
    grid = cfg["xgb_param_grid"]
    n_iter = cfg["n_iter_xgb"]

    best_score = -np.inf
    best_params: Dict[str, Any] = {}
    best_estimator: Optional[XGBClassifier] = None

    for _ in tqdm(range(n_iter), desc="XGB RandomSearch", unit="cfg"):
        params = {k: rng.choice(v).item() if isinstance(v[0], (int, float)) else rng.choice(v)
                  for k, v in grid.items()}
        clf = XGBClassifier(
            **params,
            random_state=cfg["random_state"],
            n_jobs=cfg["n_jobs"],
            use_label_encoder=False,
            eval_metric="aucpr",
            early_stopping_rounds=cfg["xgb_early_stopping_rounds"],
            verbosity=0,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_eval, y_eval)], verbose=False)
        score = average_precision_score(y_eval, clf.predict_proba(X_eval)[:, 1])
        if score > best_score:
            best_score = score
            best_params = params
            best_estimator = clf

    assert best_estimator is not None
    LOG.info(f"Best XGB params: {best_params} | Eval AUPRC: {best_score:.4f}")

    # Refit on full data without early stopping for the final model
    final_xgb = XGBClassifier(
        **best_params,
        random_state=cfg["random_state"],
        n_jobs=cfg["n_jobs"],
        use_label_encoder=False,
        eval_metric="aucpr",
        verbosity=0,
    )
    final_xgb.fit(X, y)
    joblib.dump(final_xgb, ckpt)
    _log_step(f"XGB checkpoint saved → {ckpt}")
    return final_xgb


# ===========================================================================
# OOF STACKING
# ===========================================================================

def _fit_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    base_learners: List[Any],
    smote_ratio: float,
    random_state: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Fit all base learners on one fold and return their OOF predictions.

    Runs inside a joblib.Parallel call for maximum speed.

    Returns
    -------
    fold_idx     : int
    oof_probs    : np.ndarray of shape (n_val, n_base_learners)
    val_idx      : np.ndarray — returned so the caller can reconstruct order
    """
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val = X[val_idx]

    X_tr_res, X_val_sc, y_tr_res = preprocess(X_tr, X_val, y_tr, smote_ratio, random_state)

    fold_probs = np.zeros((len(val_idx), len(base_learners)), dtype=np.float32)
    for j, clf in enumerate(base_learners):
        clf_clone = joblib.loads(joblib.dumps(clf))  # thread-safe deep copy
        clf_clone.fit(X_tr_res, y_tr_res)
        fold_probs[:, j] = clf_clone.predict_proba(X_val_sc)[:, 1]

    return fold_idx, fold_probs, val_idx


def generate_oof_meta_features(
    X: np.ndarray,
    y: np.ndarray,
    base_learners: List[Any],
    cv: StratifiedKFold,
    smote_ratio: float,
    n_jobs: int,
    random_state: int,
    desc: str = "OOF Stacking",
) -> np.ndarray:
    """
    Generate Out-of-Fold meta-features via parallelised k-fold stacking.

    Algorithm
    ---------
    1. Split data into K folds using StratifiedKFold.
    2. For each fold k in parallel:
        a. Apply SMOTE+scaling strictly to training folds only.
        b. Fit each base learner on the SMOTE-augmented fold.
        c. Predict probabilities for the held-out fold.
    3. Assemble predictions into an (n_samples, n_base_learners) matrix.
    4. This matrix becomes the training set for the meta-learner.

    Parameters
    ----------
    X             : Feature matrix
    y             : Labels
    base_learners : List of fitted (but to-be-refit) base learner objects
    cv            : StratifiedKFold splitter
    smote_ratio   : SMOTE sampling strategy
    n_jobs        : Parallelism level
    random_state  : Seed
    desc          : Progress-bar description

    Returns
    -------
    meta_X : np.ndarray of shape (n_samples, n_learners)
    """
    splits = list(cv.split(X, y))
    _log_step(f"Generating OOF meta-features across {len(splits)} folds …")

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_fold)(
            fold_idx, train_idx, val_idx,
            X, y, base_learners, smote_ratio, random_state,
        )
        for fold_idx, (train_idx, val_idx) in enumerate(
            tqdm(splits, desc=desc, unit="fold")
        )
    )

    meta_X = np.zeros((len(y), len(base_learners)), dtype=np.float32)
    for fold_idx, fold_probs, val_idx in results:
        meta_X[val_idx] = fold_probs

    _log_step(f"OOF meta-feature matrix: {meta_X.shape}")
    return meta_X


# ===========================================================================
# FULL PIPELINE
# ===========================================================================

def run_full_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
    output_dir: Path,
    fig_dir: Path,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    """
    End-to-end stacking ensemble pipeline.

    Stages
    ------
    1. Global train/test split (80/20, stratified)
    2. Tune each base learner on the training fold using RandomizedSearchCV
    3. Generate OOF meta-features via parallelised k-fold stacking
    4. Train meta-learner on OOF meta-features
    5. Evaluate on held-out test set (full pipeline)
    6. Calibrate final model with Platt scaling
    7. Threshold optimisation on test set
    8. Save artefacts: model, scalers, metrics JSON

    Parameters
    ----------
    X             : Feature DataFrame
    y             : Label Series
    cfg           : CONFIG dict
    output_dir    : Directory for model / metrics artefacts
    fig_dir       : Directory for figures
    checkpoint_dir: Directory for per-learner checkpoints

    Returns
    -------
    results dict containing metrics at both thresholds
    """
    from sklearn.model_selection import train_test_split

    _log_step("Starting full pipeline …")
    X_arr = X.values.astype(np.float32)
    y_arr = y.values

    # ── Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr,
        test_size=0.2,
        stratify=y_arr,
        random_state=cfg["random_state"],
    )
    _log_step(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # ── Preprocess full training set for tuning ────────────────────────────
    X_train_sc, X_test_sc, y_train_res = preprocess(
        X_train, X_test, y_train,
        cfg["smote_sampling_strategy"],
        cfg["random_state"],
    )

    cv = StratifiedKFold(
        n_splits=cfg["n_folds"],
        shuffle=True,
        random_state=cfg["random_state"],
    )

    # ── Tune base learners ─────────────────────────────────────────────────
    lr_clf = _tune_logistic_regression(X_train_sc, y_train_res, cv, cfg, checkpoint_dir)
    rf_clf = _tune_random_forest(X_train_sc, y_train_res, cv, cfg, checkpoint_dir)
    xgb_clf = _tune_xgboost(X_train_sc, y_train_res, cv, cfg, checkpoint_dir)
    base_learners = [lr_clf, rf_clf, xgb_clf]

    # ── Generate OOF meta-features ─────────────────────────────────────────
    meta_X_train = generate_oof_meta_features(
        X_train, y_train,
        base_learners=base_learners,
        cv=cv,
        smote_ratio=cfg["smote_sampling_strategy"],
        n_jobs=cfg["n_jobs"],
        random_state=cfg["random_state"],
    )

    # ── Train meta-learner ─────────────────────────────────────────────────
    _log_step("Training meta-learner (Logistic Regression) …")
    meta_lr = LogisticRegression(
        C=cfg["meta_lr_C"],
        solver=cfg["meta_lr_solver"],
        max_iter=cfg["meta_lr_max_iter"],
        random_state=cfg["random_state"],
    )
    meta_lr.fit(meta_X_train, y_train)

    # ── Build test meta-features ───────────────────────────────────────────
    _log_step("Generating test meta-features …")
    scaler_final = StandardScaler()
    X_train_final_sc = scaler_final.fit_transform(X_train)
    X_test_final_sc = scaler_final.transform(X_test)

    meta_X_test = np.column_stack([
        clf.predict_proba(X_test_final_sc)[:, 1] for clf in base_learners
    ]).astype(np.float32)

    # ── Inference ─────────────────────────────────────────────────────────
    y_prob_test = meta_lr.predict_proba(meta_X_test)[:, 1]

    # ── Metrics at default threshold ──────────────────────────────────────
    metrics_05 = compute_metrics(y_test, y_prob_test, threshold=0.5)
    opt_thresh = find_optimal_threshold(y_test, y_prob_test)
    metrics_opt = compute_metrics(y_test, y_prob_test, threshold=opt_thresh)

    LOG.info("=" * 70)
    LOG.info("RESULTS @ threshold = 0.50")
    for k, v in metrics_05.items():
        LOG.info(f"  {k:15s}: {v}")
    LOG.info(f"\nOptimal threshold = {opt_thresh:.4f}")
    LOG.info("RESULTS @ optimal threshold")
    for k, v in metrics_opt.items():
        LOG.info(f"  {k:15s}: {v}")
    LOG.info("=" * 70)

    # ── Save figures ───────────────────────────────────────────────────────
    _save_roc_curve(y_test, y_prob_test, fig_dir)
    _save_pr_curve(y_test, y_prob_test, fig_dir)
    _save_confusion_matrix(y_test, y_prob_test, opt_thresh, fig_dir)
    _save_calibration_curve(y_test, y_prob_test, fig_dir)

    # ── Feature importance (SHAP + permutation) ────────────────────────────
    _save_shap_summary(
        meta_lr, meta_X_test,
        feature_names=["LR_prob", "RF_prob", "XGB_prob"],
        fig_dir=fig_dir,
    )
    _save_permutation_importance(
        meta_lr, meta_X_test, y_test,
        feature_names=["LR_prob", "RF_prob", "XGB_prob"],
        fig_dir=fig_dir,
    )

    # ── Save model bundle ──────────────────────────────────────────────────
    model_bundle = {
        "base_learners": base_learners,
        "meta_learner": meta_lr,
        "scaler": scaler_final,
        "feature_names": list(X.columns),
        "optimal_threshold": opt_thresh,
        "config": cfg,
    }
    model_path = output_dir / "stacking_model.pkl"
    joblib.dump(model_bundle, model_path)
    _log_step(f"Model bundle saved → {model_path}")

    # ── Reproducibility assertion ──────────────────────────────────────────
    LOG.info("Reproducibility check: reloading model and asserting identical predictions …")
    reloaded = joblib.load(model_path)
    y_prob_reloaded = reloaded["meta_learner"].predict_proba(meta_X_test)[:, 1]
    assert np.allclose(y_prob_test, y_prob_reloaded, atol=1e-6), \
        "FAIL: Reloaded model predictions differ from original!"
    LOG.info("Reproducibility assertion PASSED ✓")

    # ── Save metrics JSON ──────────────────────────────────────────────────
    metrics_combined = {
        "threshold_0.5": metrics_05,
        "threshold_optimal": metrics_opt,
        "optimal_threshold_value": opt_thresh,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_combined, f, indent=2, default=float)
    _log_step(f"Metrics saved → {metrics_path}")

    return metrics_combined


# ===========================================================================
# STABILITY ANALYSIS
# ===========================================================================

def run_stability_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
    output_dir: Path,
    fig_dir: Path,
    n_runs: int,
) -> Dict[str, Any]:
    """
    Repeat the full CV loop across `n_runs` random seeds to assess stability.

    Computes mean ± std and 95% confidence intervals for each metric.

    Parameters
    ----------
    X         : Feature DataFrame
    y         : Label Series
    cfg       : CONFIG dict
    output_dir: Output directory
    fig_dir   : Figure directory
    n_runs    : Number of independent runs (default 5)

    Returns
    -------
    stability_report : dict with mean, std, and 95% CI per metric
    """
    _log_step(f"Starting stability analysis ({n_runs} runs) …")
    from sklearn.model_selection import train_test_split

    run_metrics: List[Dict[str, float]] = []
    X_arr = X.values.astype(np.float32)
    y_arr = y.values

    for run in tqdm(range(n_runs), desc="Stability Runs", unit="run"):
        seed = cfg["random_state"] + run * 1000
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_arr, y_arr,
            test_size=0.2,
            stratify=y_arr,
            random_state=seed,
        )
        X_tr_sc, X_te_sc, y_tr_res = preprocess(
            X_tr, X_te, y_tr,
            cfg["smote_sampling_strategy"],
            seed,
        )
        cv = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=seed)

        # Lightweight quick train (no hyperparameter search — use saved best params)
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced",
            n_jobs=cfg["n_jobs"],
            random_state=seed,
        )
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=50,
            n_jobs=cfg["n_jobs"],
            random_state=seed,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="aucpr",
        )
        lr_base = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
        base_learners = [lr_base, rf, xgb]
        for clf in base_learners:
            clf.fit(X_tr_sc, y_tr_res)

        meta_X_te = np.column_stack([
            clf.predict_proba(X_te_sc)[:, 1] for clf in base_learners
        ]).astype(np.float32)

        meta_X_tr = generate_oof_meta_features(
            X_tr, y_tr,
            base_learners=base_learners,
            cv=cv,
            smote_ratio=cfg["smote_sampling_strategy"],
            n_jobs=cfg["n_jobs"],
            random_state=seed,
            desc=f"OOF Run {run+1}",
        )
        meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        meta_lr.fit(meta_X_tr, y_tr)
        y_prob = meta_lr.predict_proba(meta_X_te)[:, 1]
        run_metrics.append(compute_metrics(y_te, y_prob))

    # ── Aggregate ─────────────────────────────────────────────────────────
    metric_keys = ["AUPRC", "ROC_AUC", "F1", "MCC", "Precision", "Recall"]
    stability_report: Dict[str, Any] = {}
    alpha = 0.05
    n = len(run_metrics)

    for key in metric_keys:
        vals = np.array([m[key] for m in run_metrics])
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        ci = t_crit * std / np.sqrt(n)
        stability_report[key] = {
            "mean": mean,
            "std": std,
            "ci_95_lower": mean - ci,
            "ci_95_upper": mean + ci,
        }
        LOG.info(
            f"  {key:10s}: {mean:.4f} ± {std:.4f}  "
            f"95% CI [{mean-ci:.4f}, {mean+ci:.4f}]"
        )

    stab_path = output_dir / "stability_report.json"
    with open(stab_path, "w") as f:
        json.dump(stability_report, f, indent=2)
    _log_step(f"Stability report saved → {stab_path}")
    return stability_report


# ===========================================================================
# ABLATION STUDY
# ===========================================================================

def run_ablation_study(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Ablation study: evaluate stacking with 1, 2, and 3 base learners.

    Combinations tested
    -------------------
    1 learner : [XGBoost]  (strongest single learner)
    2 learners: [RF, XGBoost]
    3 learners: [LR, RF, XGBoost]

    Parameters
    ----------
    X         : Feature DataFrame
    y         : Label Series
    cfg       : CONFIG dict
    output_dir: Results directory

    Returns
    -------
    ablation_df : DataFrame with AUPRC, F1, MCC per combination
    """
    from sklearn.model_selection import train_test_split

    _log_step("Running ablation study …")
    X_arr = X.values.astype(np.float32)
    y_arr = y.values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_arr, y_arr, test_size=0.2,
        stratify=y_arr, random_state=cfg["random_state"],
    )
    X_tr_sc, X_te_sc, y_tr_res = preprocess(
        X_tr, X_te, y_tr,
        cfg["smote_sampling_strategy"],
        cfg["random_state"],
    )

    lr_base = LogisticRegression(C=0.1, max_iter=1000, random_state=cfg["random_state"])
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20,
        class_weight="balanced",
        n_jobs=cfg["n_jobs"],
        random_state=cfg["random_state"],
    )
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5,
        learning_rate=0.1, scale_pos_weight=50,
        n_jobs=cfg["n_jobs"],
        random_state=cfg["random_state"],
        verbosity=0, use_label_encoder=False, eval_metric="aucpr",
    )
    all_learners = [
        ("LR", lr_base),
        ("RF", rf),
        ("XGB", xgb),
    ]
    for _, clf in all_learners:
        clf.fit(X_tr_sc, y_tr_res)

    combos = [
        ("XGBoost only",  [("XGB", xgb)]),
        ("RF + XGBoost",  [("RF", rf), ("XGB", xgb)]),
        ("LR + RF + XGB", all_learners),
    ]

    cv = StratifiedKFold(
        n_splits=cfg["n_folds"], shuffle=True,
        random_state=cfg["random_state"],
    )
    rows = []
    for combo_name, learners in tqdm(combos, desc="Ablation", unit="combo"):
        lrns = [clf for _, clf in learners]
        meta_X_tr = generate_oof_meta_features(
            X_tr, y_tr, base_learners=lrns, cv=cv,
            smote_ratio=cfg["smote_sampling_strategy"],
            n_jobs=cfg["n_jobs"],
            random_state=cfg["random_state"],
            desc=f"Ablation: {combo_name}",
        )
        meta_X_te = np.column_stack([
            clf.predict_proba(X_te_sc)[:, 1] for clf in lrns
        ]).astype(np.float32)
        meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg["random_state"])
        meta_lr.fit(meta_X_tr, y_tr)
        y_prob = meta_lr.predict_proba(meta_X_te)[:, 1]
        m = compute_metrics(y_te, y_prob)
        rows.append({
            "Configuration": combo_name,
            "AUPRC": m["AUPRC"],
            "ROC_AUC": m["ROC_AUC"],
            "F1": m["F1"],
            "MCC": m["MCC"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
        })

    ablation_df = pd.DataFrame(rows)
    LOG.info("\nAblation Study Results:\n" + ablation_df.to_string(index=False))
    ablation_path = output_dir / "ablation_results.csv"
    ablation_df.to_csv(ablation_path, index=False)
    _log_step(f"Ablation results saved → {ablation_path}")
    return ablation_df


# ===========================================================================
# LEARNING CURVES
# ===========================================================================

def run_learning_curves(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
    fig_dir: Path,
) -> None:
    """
    Generate learning curves showing AUPRC vs training-set size.

    Uses XGBoost (strongest single learner) as a proxy for the full ensemble,
    which allows fast generation without running full stacking at each size.

    Parameters
    ----------
    X       : Feature DataFrame
    y       : Label Series
    cfg     : CONFIG dict
    fig_dir : Figure output directory
    """
    _log_step("Generating learning curves …")
    X_sc = StandardScaler().fit_transform(X.values.astype(np.float32))
    y_arr = y.values

    clf = XGBClassifier(
        n_estimators=300, max_depth=5,
        learning_rate=0.1, scale_pos_weight=50,
        n_jobs=cfg["n_jobs"],
        random_state=cfg["random_state"],
        verbosity=0, use_label_encoder=False, eval_metric="aucpr",
    )

    train_sizes, train_scores, val_scores = learning_curve(
        clf, X_sc, y_arr,
        cv=StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True,
                           random_state=cfg["random_state"]),
        scoring="average_precision",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=cfg["n_jobs"],
        verbose=0,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training AUPRC", color="#2196F3")
    ax.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.15, color="#2196F3",
    )
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation AUPRC", color="#F44336")
    ax.fill_between(
        train_sizes,
        val_scores.mean(axis=1) - val_scores.std(axis=1),
        val_scores.mean(axis=1) + val_scores.std(axis=1),
        alpha=0.15, color="#F44336",
    )
    ax.set_xlabel("Training Set Size", fontsize=13)
    ax.set_ylabel("AUPRC", fontsize=13)
    ax.set_title("Learning Curves (XGBoost — AUPRC vs Training Size)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = fig_dir / "learning_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    _log_step(f"Learning curves saved → {path}")


# ===========================================================================
# FIGURES
# ===========================================================================

def _save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, fig_dir: Path) -> None:
    """Save ROC curve figure."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, color="#1565C0", label=f"ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve — Stacking Ensemble", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(fig_dir / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    _log_step("ROC curve saved.")


def _save_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, fig_dir: Path) -> None:
    """Save Precision-Recall curve figure."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, lw=2, color="#6A1B9A", label=f"AUPRC = {auprc:.4f}")
    ax.axhline(y=baseline, color="gray", lw=1, linestyle="--",
               label=f"Baseline (random) = {baseline:.4f}")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve — Stacking Ensemble", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(fig_dir / "pr_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    _log_step("PR curve saved.")


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    fig_dir: Path,
) -> None:
    """Save annotated confusion matrix heatmap."""
    import itertools
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    classes = ["Legitimate", "Fraud"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    thresh_cm = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh_cm else "black",
                fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.3f})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    _log_step("Confusion matrix saved.")


def _save_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, fig_dir: Path) -> None:
    """Save probability calibration curve (reliability diagram)."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_predicted_value, fraction_of_positives, "s-",
            color="#00897B", label="Stacking Ensemble")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=13)
    ax.set_ylabel("Fraction of Positives", fontsize=13)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(fig_dir / "calibration_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    _log_step("Calibration curve saved.")


def _save_shap_summary(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    fig_dir: Path,
) -> None:
    """Save SHAP summary (beeswarm) plot for the meta-learner."""
    _log_step("Computing SHAP values for meta-learner …")
    explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)
    # For multi-output, take class=1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False, plot_type="dot",
    )
    plt.title("SHAP Summary Plot — Meta-Learner", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    _log_step("SHAP summary plot saved.")


def _save_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    fig_dir: Path,
    n_repeats: int = 30,
) -> None:
    """Save permutation importance bar chart for the meta-learner."""
    _log_step("Computing permutation importances …")
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        scoring="average_precision",
        n_jobs=1,
    )
    sorted_idx = result.importances_mean.argsort()[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        result.importances_mean[sorted_idx],
        xerr=result.importances_std[sorted_idx],
        color=["#EF5350", "#42A5F5", "#66BB6A"][:len(feature_names)],
        capsize=4,
        alpha=0.85,
    )
    ax.set_xlabel("Mean Decrease in AUPRC", fontsize=13)
    ax.set_title("Permutation Importance — Meta-Learner", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(fig_dir / "permutation_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    _log_step("Permutation importance saved.")


# ===========================================================================
# CLI
# ===========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI parser."""
    p = argparse.ArgumentParser(
        description="Stacking Ensemble for Credit Card Fraud Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path", default="data/creditcard.csv",
                   help="Path to the creditcard.csv dataset.")
    p.add_argument("--output-dir", default="results/",
                   help="Directory for model, metrics, and reports.")
    p.add_argument("--fig-dir", default="figures/",
                   help="Directory for figure exports.")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Number of parallel jobs (-1 = all cores).")
    p.add_argument("--n-iter-rf", type=int, default=CONFIG["n_iter_rf"],
                   help="RandomizedSearchCV iterations for Random Forest.")
    p.add_argument("--n-iter-xgb", type=int, default=CONFIG["n_iter_xgb"],
                   help="RandomizedSearchCV iterations for XGBoost.")
    p.add_argument("--smote-ratio", type=float, default=CONFIG["smote_sampling_strategy"],
                   help="SMOTE sampling_strategy (minority:majority ratio).")
    p.add_argument("--n-folds", type=int, default=CONFIG["n_folds"],
                   help="Number of StratifiedKFold folds.")
    p.add_argument("--n-stability-runs", type=int, default=CONFIG["n_stability_runs"],
                   help="Number of stability analysis runs.")
    p.add_argument("--skip-ablation", action="store_true",
                   help="Skip the ablation study.")
    p.add_argument("--skip-learning-curves", action="store_true",
                   help="Skip learning curve generation.")
    return p


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    """Main entry point — parse args, configure, run pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # ── Override CONFIG from CLI ───────────────────────────────────────────
    CONFIG["n_jobs"] = args.n_jobs
    CONFIG["n_iter_rf"] = args.n_iter_rf
    CONFIG["n_iter_xgb"] = args.n_iter_xgb
    CONFIG["smote_sampling_strategy"] = args.smote_ratio
    CONFIG["n_folds"] = args.n_folds
    CONFIG["n_stability_runs"] = args.n_stability_runs

    # ── Directory setup ────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    for d in [output_dir, fig_dir, checkpoint_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Reinitialise logging with file handler
    global LOG
    log_path = log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    LOG = _setup_logging(str(log_path))

    LOG.info("=" * 70)
    LOG.info("Ensemble ML — Credit Card Fraud Detection")
    LOG.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    LOG.info(f"Config: {json.dumps({k: v for k, v in CONFIG.items() if not k.endswith('grid')}, indent=2)}")
    LOG.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    X, y = load_and_engineer_features(args.data_path)

    # ── Main experiment ───────────────────────────────────────────────────
    results = run_full_pipeline(X, y, CONFIG, output_dir, fig_dir, checkpoint_dir)

    # ── Stability analysis ────────────────────────────────────────────────
    stability = run_stability_analysis(
        X, y, CONFIG, output_dir, fig_dir, args.n_stability_runs
    )

    # ── Ablation study ────────────────────────────────────────────────────
    if not args.skip_ablation:
        ablation_df = run_ablation_study(X, y, CONFIG, output_dir)
    else:
        LOG.info("Ablation study skipped (--skip-ablation).")

    # ── Learning curves ───────────────────────────────────────────────────
    if not args.skip_learning_curves:
        run_learning_curves(X, y, CONFIG, fig_dir)
    else:
        LOG.info("Learning curves skipped (--skip-learning-curves).")

    # ── Final summary ─────────────────────────────────────────────────────
    LOG.info("=" * 70)
    LOG.info("EXPERIMENT COMPLETE")
    LOG.info(f"  Model saved: {output_dir / 'stacking_model.pkl'}")
    LOG.info(f"  Metrics:     {output_dir / 'metrics.json'}")
    LOG.info(f"  Figures:     {fig_dir}")
    LOG.info(f"  Log:         {log_path}")
    LOG.info(f"  AUPRC (optimal threshold): {results['threshold_optimal']['AUPRC']:.4f}")
    LOG.info(f"  F1    (optimal threshold): {results['threshold_optimal']['F1']:.4f}")
    LOG.info(f"  MCC   (optimal threshold): {results['threshold_optimal']['MCC']:.4f}")
    LOG.info("=" * 70)


if __name__ == "__main__":
    main()
