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
Output:                   Fraud probability score and thresholded prediction

Key Features
------------
- SMOTE applied strictly INSIDE each CV fold (leakage-free)
- Parallelised OOF stacking via joblib.Parallel
- Threshold optimisation: reports at both 0.5 and F1-optimal threshold
- Statistical significance: mean ± std across stability runs with 95 % CI
- Ablation study: test 1, 2, and 3 base learners
- Learning curves: AUPRC vs training-set size
- SHAP summary + permutation importance
- Reliability curves (probability calibration diagnostics; no calibrator is fitted)
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
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import Memory, Parallel, delayed
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.base import clone
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
    train_test_split,
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
    "n_jobs": -1,  # CV-level parallelism; individual estimators use one thread
    "split_strategy": "temporal",
    # ── XGBoost uses CV-selected estimator count; no shared eval_set ──────────
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
    Load the ULB credit-card dataset and engineer a chronological feature table.

    The source dataset's Time column is elapsed seconds from the first
    transaction.  It is therefore converted to elapsed hours (0–~48), not
    interpreted as a wall-clock hour.  Rows are sorted by Time so the default
    train/validation/test protocol is chronological.
    """
    _log_step(f"Loading data from '{data_path}'")
    started = time.perf_counter()

    dtype_map: Dict[str, str] = {f"V{i}": "float32" for i in range(1, 29)}
    dtype_map.update({"Amount": "float32", "Time": "float32"})
    df = pd.read_csv(data_path, dtype=dtype_map)

    required = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Class"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns.")
    if not set(df["Class"].unique()).issubset({0, 1}) or df["Class"].nunique() != 2:
        raise ValueError("Class must contain both binary labels 0 and 1.")
    if (df["Amount"] < 0).any() or (df["Time"] < 0).any():
        raise ValueError("Amount and Time must be non-negative.")

    df = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    df["Amount_log"] = np.log1p(df["Amount"]).astype("float32")
    df["Hour"] = (df["Time"] / 3600.0).astype("float32")

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_log", "Hour"]
    X = df[feature_cols].copy()
    y = df["Class"].astype("int8")
    X.attrs["ordered_by_time"] = True

    _log_step(
        f"Loaded {len(df):,} chronological rows in {time.perf_counter()-started:.2f}s; "
        f"features={X.shape}; fraud={int(y.sum()):,} ({y.mean():.4%})"
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
    Legacy preprocessing helper retained for notebook compatibility.

    New experiments must use an imbalanced-learn Pipeline so scaling and
    resampling are fitted independently inside every cross-validation fold.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    sampler = SMOTE(
        sampling_strategy=smote_ratio,
        random_state=random_state,
        k_neighbors=min(5, max(1, int(np.sum(y_train)) - 1)),
    )
    X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
    return X_resampled, X_test_scaled, y_resampled

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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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

def _pipeline(estimator: Any, cfg: Dict[str, Any], random_state: int) -> ImbPipeline:
    """Build a fold-safe preprocessing + sampling + classifier pipeline."""
    return ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(
            sampling_strategy=cfg["smote_sampling_strategy"],
            random_state=random_state,
            k_neighbors=5,
        )),
        ("classifier", estimator),
    ])


def _search_pipeline(
    pipeline: ImbPipeline,
    param_grid: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    n_iter: int,
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    label: str,
) -> ImbPipeline:
    """Tune a complete pipeline using training-only cross-validation."""
    if checkpoint_path.exists():
        LOG.info("Checkpoint found — loading %s pipeline.", label)
        cached = joblib.load(checkpoint_path)
        if not isinstance(cached, ImbPipeline):
            raise RuntimeError(f"Invalid {label} checkpoint: expected an imbalanced-learn Pipeline.")
        return cached

    _log_step(f"Tuning {label} pipeline with fold-local scaling and SMOTE …")
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        n_jobs=cfg["n_jobs"],
        pre_dispatch="2*n_jobs",
        random_state=cfg["random_state"],
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X, y)
    LOG.info(
        "Best %s parameters: %s | CV AUPRC: %.5f",
        label, search.best_params_, search.best_score_,
    )
    joblib.dump(search.best_estimator_, checkpoint_path)
    _log_step(f"{label} pipeline checkpoint saved → {checkpoint_path}")
    return search.best_estimator_


def _tune_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    cfg: Dict[str, Any],
    checkpoint_dir: Path,
) -> ImbPipeline:
    model = LogisticRegression(random_state=cfg["random_state"], n_jobs=1)
    pipeline = _pipeline(model, cfg, cfg["random_state"])
    grid = {f"classifier__{key}": value for key, value in cfg["lr_param_grid"].items()}
    return _search_pipeline(
        pipeline, grid, X, y, cv, cfg["n_iter_lr"], cfg,
        checkpoint_dir / "lr_pipeline_v2_best.pkl", "Logistic Regression",
    )


def _tune_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    cfg: Dict[str, Any],
    checkpoint_dir: Path,
) -> ImbPipeline:
    model = RandomForestClassifier(
        random_state=cfg["random_state"], n_jobs=1,
    )
    pipeline = _pipeline(model, cfg, cfg["random_state"])
    grid = {f"classifier__{key}": value for key, value in cfg["rf_param_grid"].items()}
    return _search_pipeline(
        pipeline, grid, X, y, cv, cfg["n_iter_rf"], cfg,
        checkpoint_dir / "rf_pipeline_v2_best.pkl", "Random Forest",
    )


def _tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    cfg: Dict[str, Any],
    checkpoint_dir: Path,
) -> ImbPipeline:
    """
    Tune XGBoost using the same fold-local preprocessing/resampling protocol.

    Early stopping is deliberately not passed a globally reused eval_set:
    ordinary CV scoring selects hyperparameters without contaminating a
    repeated holdout, and the selected pipeline is refitted on the full
    development-training partition by RandomizedSearchCV(refit=True).
    """
    model = XGBClassifier(
        random_state=cfg["random_state"],
        n_jobs=1,  # parallelism is owned by RandomizedSearchCV
        eval_metric="aucpr",
        verbosity=0,
        tree_method="hist",
    )
    pipeline = _pipeline(model, cfg, cfg["random_state"])
    grid = {f"classifier__{key}": value for key, value in cfg["xgb_param_grid"].items()}
    return _search_pipeline(
        pipeline, grid, X, y, cv, cfg["n_iter_xgb"], cfg,
        checkpoint_dir / "xgb_pipeline_v2_best.pkl", "XGBoost",
    )

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
    reference_X: Optional[np.ndarray] = None,
    reference_y: Optional[np.ndarray] = None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Fit cloned pipelines for one OOF fold.

    reference_X/reference_y are an independent tuning partition that is
    available to every fold. The OOF validation rows are never used for
    hyperparameter selection or fitting.
    """
    X_train, y_train = X[train_idx], y[train_idx]
    if reference_X is not None or reference_y is not None:
        if reference_X is None or reference_y is None:
            raise ValueError("reference_X and reference_y must be supplied together.")
        X_train = np.concatenate([reference_X, X_train], axis=0)
        y_train = np.concatenate([reference_y, y_train], axis=0)

    X_valid = X[val_idx]
    fold_probs = np.zeros((len(val_idx), len(base_learners)), dtype=np.float32)
    for column, estimator in enumerate(base_learners):
        fitted = clone(estimator)
        fitted.fit(X_train, y_train)
        fold_probs[:, column] = fitted.predict_proba(X_valid)[:, 1]

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
    reference_X: Optional[np.ndarray] = None,
    reference_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate OOF meta-features using fold-safe preprocessing.

    When reference data is supplied, it must be a disjoint training partition
    used for hyperparameter tuning. Each OOF fold trains on that reference set
    plus the non-held-out rows of X, while the held-out rows are used only for
    out-of-fold prediction. Hyperparameter search must never see X/y.
    """
    if (reference_X is None) != (reference_y is None):
        raise ValueError("reference_X and reference_y must be supplied together.")
    splits = list(cv.split(X, y))
    _log_step(f"Generating OOF meta-features across {len(splits)} folds …")
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_fold)(
            fold_idx, train_idx, valid_idx, X, y, base_learners,
            smote_ratio, random_state, reference_X, reference_y,
        )
        for fold_idx, (train_idx, valid_idx) in enumerate(
            tqdm(splits, desc=desc, unit="fold")
        )
    )

    meta_features = np.zeros((len(y), len(base_learners)), dtype=np.float32)
    for _, fold_probs, valid_idx in results:
        meta_features[valid_idx] = fold_probs
    if not np.isfinite(meta_features).all():
        raise RuntimeError("OOF meta-feature matrix contains non-finite predictions.")
    _log_step(f"OOF meta-feature matrix: {meta_features.shape}")
    return meta_features

# ===========================================================================
# FULL PIPELINE
# ===========================================================================

def _split_train_validation_test(
    X: np.ndarray,
    y: np.ndarray,
    split_strategy: str,
    seed: int,
    ordered_by_time: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create disjoint 70/15/15 train, validation and final test partitions."""
    if split_strategy == "temporal":
        if not ordered_by_time:
            raise ValueError(
                "Temporal split requested, but data is not marked as sorted by source Time. "
                "Load it with load_and_engineer_features() or choose stratified_random."
            )
        n_rows = len(y)
        train_end = int(n_rows * 0.70)
        valid_end = int(n_rows * 0.85)
        X_train, X_valid, X_test = X[:train_end], X[train_end:valid_end], X[valid_end:]
        y_train, y_valid, y_test = y[:train_end], y[train_end:valid_end], y[valid_end:]
    elif split_strategy == "stratified_random":
        X_development, X_test, y_development, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=seed,
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_development, y_development,
            test_size=(0.15 / 0.85), stratify=y_development, random_state=seed,
        )
    else:
        raise ValueError("split_strategy must be 'temporal' or 'stratified_random'.")

    for name, labels in (("train", y_train), ("validation", y_valid), ("test", y_test)):
        if len(np.unique(labels)) != 2:
            raise ValueError(
                f"The {name} partition contains only one class. "
                "Use a different split or a dataset with more fraud examples."
            )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def _predict_base_meta_features(base_learners: List[Any], X: np.ndarray) -> np.ndarray:
    """Run raw feature rows through fitted preprocessing/model pipelines."""
    return np.column_stack([
        learner.predict_proba(X)[:, 1] for learner in base_learners
    ]).astype(np.float32)


def run_full_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
    output_dir: Path,
    fig_dir: Path,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    """
    Train and evaluate a stacking ensemble without using the final test set
    for tuning, threshold selection, preprocessing or model selection.

    Default protocol: chronological 70% train / 15% validation / 15% test.
    A stratified random split is available for comparison, not as the online
    deployment evidence.
    """
    _log_step("Starting full pipeline …")
    X_array = X.to_numpy(dtype=np.float32)
    y_array = y.to_numpy(dtype=np.int8)
    strategy = cfg.get("split_strategy", "temporal")
    X_train, X_valid, X_test, y_train, y_valid, y_test = _split_train_validation_test(
        X_array, y_array, strategy, cfg["random_state"],
        ordered_by_time=bool(X.attrs.get("ordered_by_time", False)),
    )
    _log_step(
        f"Split ({strategy}): train={X_train.shape}, validation={X_valid.shape}, "
        f"test={X_test.shape}"
    )

    # Isolate hyperparameter selection from the rows used to train the
    # stacking meta-learner. Otherwise, selected hyperparameters would have
    # indirectly seen the labels of every OOF validation row.
    X_tune, X_stack, y_tune, y_stack = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train,
        random_state=cfg["random_state"],
    )
    cv = StratifiedKFold(
        n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["random_state"],
    )
    learners = [
        _tune_logistic_regression(X_tune, y_tune, cv, cfg, checkpoint_dir),
        _tune_random_forest(X_tune, y_tune, cv, cfg, checkpoint_dir),
        _tune_xgboost(X_tune, y_tune, cv, cfg, checkpoint_dir),
    ]

    _log_step("Generating OOF meta-features on a hyperparameter-disjoint stack partition …")
    meta_X_train = generate_oof_meta_features(
        X_stack, y_stack, learners, cv,
        cfg["smote_sampling_strategy"], cfg["n_jobs"], cfg["random_state"],
        reference_X=X_tune, reference_y=y_tune,
    )
    meta_learner = LogisticRegression(
        C=cfg["meta_lr_C"], solver=cfg["meta_lr_solver"],
        max_iter=cfg["meta_lr_max_iter"], random_state=cfg["random_state"],
    )
    meta_learner.fit(meta_X_train, y_stack)

    # Final base learners are refitted on the complete development-training
    # partition only after the meta-learner has been fitted from OOF predictions.
    for learner in learners:
        learner.fit(X_train, y_train)

    meta_X_valid = _predict_base_meta_features(learners, X_valid)
    meta_X_test = _predict_base_meta_features(learners, X_test)
    valid_prob = meta_learner.predict_proba(meta_X_valid)[:, 1]
    test_prob = meta_learner.predict_proba(meta_X_test)[:, 1]

    # The operating threshold is selected on the validation partition only.
    optimal_threshold = find_optimal_threshold(y_valid, valid_prob)
    validation_metrics = compute_metrics(y_valid, valid_prob, optimal_threshold)
    metrics_05 = compute_metrics(y_test, test_prob, threshold=0.5)
    metrics_opt = compute_metrics(y_test, test_prob, threshold=optimal_threshold)
    base_model_test_metrics = {
        name: compute_metrics(y_test, learner.predict_proba(X_test)[:, 1], threshold=0.5)
        for name, learner in zip(
            ["Logistic Regression", "Random Forest", "XGBoost"], learners
        )
    }

    LOG.info("Validation-selected operating threshold: %.6f", optimal_threshold)
    LOG.info("Final untouched test metrics (threshold 0.50): %s", metrics_05)
    LOG.info("Final untouched test metrics (validation threshold): %s", metrics_opt)

    _save_roc_curve(y_test, test_prob, fig_dir)
    _save_pr_curve(y_test, test_prob, fig_dir)
    _save_confusion_matrix(y_test, test_prob, optimal_threshold, fig_dir)
    _save_calibration_curve(y_test, test_prob, fig_dir)
    try:
        _save_shap_summary(
            meta_learner, meta_X_test,
            feature_names=["LR_prob", "RF_prob", "XGB_prob"], fig_dir=fig_dir,
        )
        _save_permutation_importance(
            meta_learner, meta_X_test, y_test,
            feature_names=["LR_prob", "RF_prob", "XGB_prob"], fig_dir=fig_dir,
        )
    except Exception as exc:
        LOG.warning("Optional meta-feature explainability plots were skipped: %s", exc)

    model_bundle = {
        "model_schema_version": "2.0",
        "base_learners": learners,
        "meta_learner": meta_learner,
        # New bundles preprocess inside each base learner pipeline.
        "scaler": None,
        "preprocessing_in_estimators": True,
        "feature_names": list(X.columns),
        "optimal_threshold": float(optimal_threshold),
        "model_version": "stacking-2.0-" + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "training_provenance": {
            "dataset": "Kaggle ULB Credit Card Fraud Detection",
            "split_strategy": strategy,
            "training_rows": int(len(y_train)),
            "base_hyperparameter_tuning_rows": int(len(y_tune)),
            "meta_oof_rows": int(len(y_stack)),
            "validation_rows": int(len(y_valid)),
            "test_rows": int(len(y_test)),
            "threshold_selected_on": "validation partition",
            "test_set_used_for_threshold_selection": False,
            "hyperparameter_tuning_rows_disjoint_from_oof_meta_rows": True,
        },
        "evaluation_protocol": {
            "threshold_selected_on": "held-out validation partition",
            "test_set_used_for_threshold_selection": False,
            "temporal_validation": strategy == "temporal",
            "resampling": "SMOTE inside every training fold via imbalanced-learn Pipeline",
            "scaling": "StandardScaler inside every training fold via imbalanced-learn Pipeline",
        },
        "config": {key: value for key, value in cfg.items() if not key.endswith("_grid")},
    }
    model_path = output_dir / "stacking_model.pkl"
    joblib.dump(model_bundle, model_path)
    _log_step(f"Model bundle saved → {model_path}")

    # Verify the serialized bundle follows the same full inference path.
    reloaded = joblib.load(model_path)
    reloaded_meta = _predict_base_meta_features(reloaded["base_learners"], X_test)
    reloaded_prob = reloaded["meta_learner"].predict_proba(reloaded_meta)[:, 1]
    if not np.allclose(test_prob, reloaded_prob, atol=1e-6):
        raise RuntimeError("Reloaded model bundle produced different predictions.")

    metrics = {
        "threshold_0.5": metrics_05,
        "threshold_optimal": metrics_opt,
        "optimal_threshold_value": float(optimal_threshold),
        "validation_threshold_metrics": validation_metrics,
        "base_model_test_metrics": base_model_test_metrics,
        "threshold_selection": {
            "method": "F1 maximisation",
            "selection_data": "held-out validation partition",
            "test_set_used": False,
        },
        "split_protocol": {
            "strategy": strategy,
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "test_partition_used_for": "one-time final evaluation only",
        },
        "evaluation_limitations": {
            "external_dataset_validation": "not performed",
            "live_production_validation": "not performed",
        },
        "model_version": model_bundle["model_version"],
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, default=float)
    _log_step(f"Metrics saved → {output_dir / 'metrics.json'}")
    return metrics

# ===========================================================================
# STABILITY ANALYSIS
# ===========================================================================

def _build_default_learners(cfg: Dict[str, Any], seed: int) -> List[ImbPipeline]:
    """Compact fixed-parameter learners for supplementary robustness studies."""
    local_cfg = dict(cfg)
    return [
        _pipeline(
            LogisticRegression(C=0.1, solver="liblinear", max_iter=1000, random_state=seed),
            local_cfg, seed,
        ),
        _pipeline(
            RandomForestClassifier(
                n_estimators=200, max_depth=20, class_weight="balanced",
                n_jobs=1, random_state=seed,
            ),
            local_cfg, seed,
        ),
        _pipeline(
            XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                scale_pos_weight=1, n_jobs=1, random_state=seed,
                eval_metric="aucpr", tree_method="hist", verbosity=0,
            ),
            local_cfg, seed,
        ),
    ]


def run_stability_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
    output_dir: Path,
    fig_dir: Path,
    n_runs: int,
) -> Dict[str, Any]:
    """
    Supplementary repeated-stratified-split robustness check.

    These intervals describe sensitivity to random partitioning on this
    benchmark. They are not a substitute for chronological or external
    validation and are labeled as such in the output.
    """
    if n_runs < 2:
        raise ValueError("At least two runs are required for a stability interval.")
    _log_step(f"Starting supplementary stability analysis ({n_runs} runs) …")
    X_array = X.to_numpy(dtype=np.float32)
    y_array = y.to_numpy(dtype=np.int8)
    run_metrics: List[Dict[str, float]] = []

    for run in tqdm(range(n_runs), desc="Stability Runs", unit="run"):
        seed = cfg["random_state"] + run * 1000
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array, test_size=0.2, stratify=y_array, random_state=seed,
        )
        learners = _build_default_learners(cfg, seed)
        cv = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=seed)
        meta_train = generate_oof_meta_features(
            X_train, y_train, learners, cv,
            cfg["smote_sampling_strategy"], cfg["n_jobs"], seed,
            desc=f"Stability OOF {run + 1}",
        )
        for learner in learners:
            learner.fit(X_train, y_train)
        meta_test = _predict_base_meta_features(learners, X_test)
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        meta.fit(meta_train, y_train)
        probabilities = meta.predict_proba(meta_test)[:, 1]
        run_metrics.append(compute_metrics(y_test, probabilities, threshold=0.5))

    metric_keys = ["AUPRC", "ROC_AUC", "F1", "MCC", "Precision", "Recall"]
    report: Dict[str, Any] = {
        "protocol": "Repeated stratified random 80/20 splits; supplementary only",
        "n_runs": n_runs,
        "threshold": 0.5,
        "metrics": {},
    }
    t_critical = stats.t.ppf(0.975, df=n_runs - 1)
    for key in metric_keys:
        values = np.asarray([item[key] for item in run_metrics], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        margin = float(t_critical * std / np.sqrt(n_runs))
        report["metrics"][key] = {
            "mean": mean, "std": std,
            "ci_95_lower": mean - margin, "ci_95_upper": mean + margin,
        }
        LOG.info("%s: %.4f ± %.4f; 95%% t interval [%.4f, %.4f]",
                 key, mean, std, mean-margin, mean+margin)

    with open(output_dir / "stability_report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    _log_step(f"Stability report saved → {output_dir / 'stability_report.json'}")
    return report

# ===========================================================================
# ABLATION STUDY
# ===========================================================================

def run_ablation_study(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    """Compare XGBoost, RF+XGBoost and the full stack on a held-out split."""
    X_array = X.to_numpy(dtype=np.float32)
    y_array = y.to_numpy(dtype=np.int8)
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y_array, test_size=0.2, stratify=y_array,
        random_state=cfg["random_state"],
    )
    learners = _build_default_learners(cfg, cfg["random_state"])
    names = ["LR", "RF", "XGB"]
    combinations = [
        ("XGBoost only", [learners[2]]),
        ("RF + XGBoost", [learners[1], learners[2]]),
        ("LR + RF + XGBoost", learners),
    ]

    cv = StratifiedKFold(
        n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["random_state"],
    )
    rows = []
    for combination_name, selected in tqdm(combinations, desc="Ablation", unit="combo"):
        meta_train = generate_oof_meta_features(
            X_train, y_train, selected, cv,
            cfg["smote_sampling_strategy"], cfg["n_jobs"], cfg["random_state"],
            desc=f"Ablation: {combination_name}",
        )
        for learner in selected:
            learner.fit(X_train, y_train)
        meta_test = _predict_base_meta_features(selected, X_test)
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg["random_state"])
        meta.fit(meta_train, y_train)
        probabilities = meta.predict_proba(meta_test)[:, 1]
        metrics = compute_metrics(y_test, probabilities, threshold=0.5)
        rows.append({
            "Configuration": combination_name,
            "AUPRC": metrics["AUPRC"],
            "ROC_AUC": metrics["ROC_AUC"],
            "F1_at_0.5": metrics["F1"],
            "MCC_at_0.5": metrics["MCC"],
            "Precision_at_0.5": metrics["Precision"],
            "Recall_at_0.5": metrics["Recall"],
        })

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ablation_results.csv", index=False)
    LOG.info("Ablation results (threshold=0.5):\n%s", frame.to_string(index=False))
    return frame

# ===========================================================================
# LEARNING CURVES
# ===========================================================================

def run_learning_curves(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Dict[str, Any],
    fig_dir: Path,
) -> None:
    """Generate fold-safe XGBoost learning curves using average precision."""
    _log_step("Generating fold-safe learning curves …")
    estimator = _pipeline(
        XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            scale_pos_weight=1, n_jobs=1, random_state=cfg["random_state"],
            eval_metric="aucpr", tree_method="hist", verbosity=0,
        ),
        cfg, cfg["random_state"],
    )
    cv = StratifiedKFold(
        n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["random_state"],
    )
    sizes, train_scores, validation_scores = learning_curve(
        estimator,
        X.to_numpy(dtype=np.float32),
        y.to_numpy(dtype=np.int8),
        cv=cv,
        scoring="average_precision",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=cfg["n_jobs"],
        error_score="raise",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, train_scores.mean(axis=1), "o-", label="Training AUPRC")
    ax.fill_between(
        sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.15,
    )
    ax.plot(sizes, validation_scores.mean(axis=1), "o-", label="Validation AUPRC")
    ax.fill_between(
        sizes,
        validation_scores.mean(axis=1) - validation_scores.std(axis=1),
        validation_scores.mean(axis=1) + validation_scores.std(axis=1),
        alpha=0.15,
    )
    ax.set_xlabel("Training rows")
    ax.set_ylabel("Average precision (AUPRC)")
    ax.set_title("Fold-safe XGBoost learning curves")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

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
                   help="Number of supplementary stability analysis runs.")
    p.add_argument("--split-strategy", choices=["temporal", "stratified_random"], default="temporal",
                   help="Default: chronological 70/15/15. Random split is supplementary.")
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
    CONFIG["split_strategy"] = args.split_strategy

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
