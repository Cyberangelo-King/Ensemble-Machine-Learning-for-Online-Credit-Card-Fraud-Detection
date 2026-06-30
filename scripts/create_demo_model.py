"""
create_demo_model.py — Lightweight Demo Model for Streamlit Dashboard
======================================================================
Final Year Project: Ensemble Machine Learning for Online Credit Card Fraud Detection
Author  : Angelo (Cyberangelo-King)

Purpose
-------
Creates a lightweight stacking ensemble model trained on *synthetic* data so that
the Streamlit dashboard (app.py) and FastAPI backend can be demonstrated without
requiring the Kaggle creditcard.csv dataset.

This script is intentionally fast (~30–60s) — it uses reduced hyperparameter
search spaces and fewer trees compared to the full `src/run_experiment.py` pipeline.

Output files (relative to script location)
-------------------------------------------
results/stacking_model.pkl   — joblib model bundle compatible with app.py
results/metrics.json         — metrics JSON compatible with app.py and backend

Usage
-----
    python scripts/create_demo_model.py
    python scripts/create_demo_model.py --n-samples 20000 --fraud-rate 0.05
    python scripts/create_demo_model.py --output-dir results/
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")

# ── Inline progress indicator (no external deps required) ─────────────────
def _bar(current: int, total: int, width: int = 40, prefix: str = "") -> None:
    """Print a simple ASCII progress bar to stdout."""
    pct = current / total
    filled = int(width * pct)
    bar_str = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r{prefix} [{bar_str}] {pct:5.1%} ({current}/{total})")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _section(title: str) -> None:
    """Print a styled section header."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _info(msg: str) -> None:
    print(f"  [..] {msg}")


# ===========================================================================
# SYNTHETIC DATA GENERATION
# ===========================================================================

def generate_synthetic_data(
    n_samples: int,
    fraud_rate: float,
    random_state: int,
) -> Tuple[Any, Any]:
    """
    Generate synthetic credit-card transaction data with realistic properties.

    The generator creates a dataset that approximates the statistical properties
    of the Kaggle Credit Card Fraud dataset:
    - V1–V28: Gaussian PCA components (fraud class shifted on V1, V3, V4, V14)
    - Amount:  Log-normal distribution (right-skewed)
    - Time:    Uniform over 48 hours
    - Class:   0 = legitimate, 1 = fraud

    Parameters
    ----------
    n_samples    : Total number of transactions
    fraud_rate   : Fraction of transactions that are fraudulent
    random_state : Random seed for reproducibility

    Returns
    -------
    X : np.ndarray of shape (n_samples, 30)
    y : np.ndarray of shape (n_samples,)
    """
    import numpy as np

    rng = np.random.default_rng(random_state)
    n_fraud = max(10, int(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    _info(f"Generating {n_legit:,} legitimate + {n_fraud:,} fraud = {n_samples:,} transactions")

    # ── Legitimate transactions ────────────────────────────────────────────
    X_legit = rng.standard_normal((n_legit, 28)).astype("float32")

    # ── Fraud transactions — shifted means on key features ─────────────────
    # V1, V3, V10, V12, V14, V17 are most discriminative in real data
    X_fraud = rng.standard_normal((n_fraud, 28)).astype("float32")
    shifts = {0: -4.5, 2: -7.0, 3: +4.5, 9: -5.0, 11: -7.3, 13: -7.4, 16: -7.2}
    for col, shift in shifts.items():
        X_fraud[:, col] += shift * 0.9  # slightly less extreme for synthetic realism

    # ── Amount (log-normal) ───────────────────────────────────────────────
    amount_legit = rng.lognormal(mean=3.5, sigma=1.5, size=n_legit).clip(0, 25000).astype("float32")
    amount_fraud = rng.lognormal(mean=4.8, sigma=1.2, size=n_fraud).clip(0, 25000).astype("float32")

    # ── Time (uniform across 48 hours) ────────────────────────────────────
    time_legit = rng.uniform(0, 172792, size=n_legit).astype("float32")
    time_fraud = rng.uniform(0, 172792, size=n_fraud).astype("float32")
    # Bias fraud to night hours (hour 2–6 = seconds 7200–21600)
    time_fraud[:n_fraud // 3] = rng.uniform(7200, 21600, size=n_fraud // 3).astype("float32")

    # ── Feature engineering ───────────────────────────────────────────────
    import numpy as np
    amount_log_legit = np.log1p(amount_legit)
    amount_log_fraud = np.log1p(amount_fraud)
    hour_legit = (time_legit % 86400) / 3600
    hour_fraud = (time_fraud % 86400) / 3600

    # ── Combine ───────────────────────────────────────────────────────────
    X_legit_full = np.column_stack([X_legit, amount_log_legit, hour_legit])
    X_fraud_full = np.column_stack([X_fraud, amount_log_fraud, hour_fraud])
    X = np.vstack([X_legit_full, X_fraud_full]).astype("float32")
    y = np.hstack([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    # ── Shuffle ───────────────────────────────────────────────────────────
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ===========================================================================
# MODEL TRAINING
# ===========================================================================

def train_demo_model(
    X_train: Any,
    y_train: Any,
    n_jobs: int,
    random_state: int,
) -> Tuple[List[Any], Any, Any]:
    """
    Train a lightweight stacking ensemble on the demo data.

    Uses reduced complexity vs the full pipeline to keep training fast:
    - LR:  C=0.1, saga, 1000 iter
    - RF:  150 trees, max_depth=15, balanced weights
    - XGB: 200 estimators, early stopping, scale_pos_weight=20

    Returns
    -------
    base_learners : List[clf]
    meta_learner  : LogisticRegression
    scaler        : StandardScaler
    """
    from imblearn.over_sampling import SMOTE
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    import numpy as np

    # ── Scale ─────────────────────────────────────────────────────────────
    _info("Scaling features (StandardScaler)…")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)

    # ── SMOTE ─────────────────────────────────────────────────────────────
    fraud_count = int(y_train.sum())
    legit_count = int(len(y_train) - fraud_count)
    _info(f"Applying SMOTE (fraud: {fraud_count}, legit: {legit_count})…")

    # Ensure enough fraud samples for SMOTE (needs k_neighbors+1 = 6 minimum)
    if fraud_count < 6:
        print(f"\n  [WARN] Only {fraud_count} fraud samples — increasing SMOTE k_neighbors to {max(1, fraud_count - 1)}")
        smote = SMOTE(k_neighbors=max(1, fraud_count - 1), random_state=random_state)
    else:
        smote = SMOTE(sampling_strategy=0.15, random_state=random_state)

    X_res, y_res = smote.fit_resample(X_sc, y_train)
    _info(f"After SMOTE: {int(y_res.sum()):,} fraud, {int((y_res == 0).sum()):,} legit")

    # ── Base learners ──────────────────────────────────────────────────────
    _info("Training Logistic Regression base learner…")
    lr = LogisticRegression(C=0.1, solver="saga", max_iter=1000, random_state=random_state)
    lr.fit(X_res, y_res)
    lr_prob = lr.predict_proba(X_sc)[:, 1]
    _ok("LR training complete")

    _info("Training Random Forest base learner (150 trees)…")
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=15,
        class_weight="balanced", n_jobs=n_jobs,
        random_state=random_state,
    )
    rf.fit(X_res, y_res)
    rf_prob = rf.predict_proba(X_sc)[:, 1]
    _ok("RF training complete")

    _info("Training XGBoost base learner (200 rounds, early stopping)…")
    from sklearn.model_selection import train_test_split
    X_xgb_tr, X_xgb_val, y_xgb_tr, y_xgb_val = train_test_split(
        X_res, y_res, test_size=0.1, stratify=y_res, random_state=random_state
    )
    xgb = XGBClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.1, scale_pos_weight=20,
        n_jobs=n_jobs, random_state=random_state,
        verbosity=0, use_label_encoder=False,
        eval_metric="aucpr",
        early_stopping_rounds=15,
    )
    xgb.fit(
        X_xgb_tr, y_xgb_tr,
        eval_set=[(X_xgb_val, y_xgb_val)],
        verbose=False,
    )
    # Refit on full res data for final model
    xgb_final = XGBClassifier(
        n_estimators=xgb.best_iteration or 100,
        max_depth=5, learning_rate=0.1, scale_pos_weight=20,
        n_jobs=n_jobs, random_state=random_state,
        verbosity=0, use_label_encoder=False, eval_metric="aucpr",
    )
    xgb_final.fit(X_res, y_res)
    xgb_prob = xgb_final.predict_proba(X_sc)[:, 1]
    _ok(f"XGB training complete (best iteration: {xgb.best_iteration or 100})")

    base_learners = [lr, rf, xgb_final]

    # ── OOF meta-features using 3-fold CV ─────────────────────────────────
    _info("Generating OOF meta-features (3-fold CV)…")
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    meta_X = np.zeros((len(X_train), 3), dtype=np.float32)

    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        _bar(fold_i + 1, 3, prefix="  OOF folds")
        X_fold_tr, X_fold_val = X_train[tr_idx], X_train[val_idx]
        y_fold_tr = y_train[tr_idx]

        fold_scaler = StandardScaler()
        X_fold_tr_sc = fold_scaler.fit_transform(X_fold_tr)
        X_fold_val_sc = fold_scaler.transform(X_fold_val)

        fraud_count_fold = int(y_fold_tr.sum())
        if fraud_count_fold < 6:
            fold_smote = SMOTE(k_neighbors=max(1, fraud_count_fold - 1), random_state=random_state)
        else:
            fold_smote = SMOTE(sampling_strategy=0.15, random_state=random_state)

        try:
            X_fold_res, y_fold_res = fold_smote.fit_resample(X_fold_tr_sc, y_fold_tr)
        except ValueError as e:
            # Graceful fallback: use unaugmented fold
            print(f"\n  [WARN] SMOTE failed on fold {fold_i}: {e}. Using unaugmented fold.")
            X_fold_res, y_fold_res = X_fold_tr_sc, y_fold_tr

        for j, clf in enumerate([lr.__class__, rf.__class__, xgb_final.__class__]):
            clf_fold = [lr, rf, xgb_final][j].__class__(
                **{k: v for k, v in [lr, rf, xgb_final][j].get_params().items()
                   if k not in ["early_stopping_rounds", "eval_metric"]}
            ) if j == 2 else type([lr, rf, xgb_final][j])(**[lr, rf, xgb_final][j].get_params())
            try:
                clf_fold.fit(X_fold_res, y_fold_res)
                meta_X[val_idx, j] = clf_fold.predict_proba(X_fold_val_sc)[:, 1]
            except Exception:
                # Fallback: use full-data model predictions
                meta_X[val_idx, j] = [lr_prob, rf_prob, xgb_prob][j][val_idx]

    _ok("OOF meta-features generated")

    # ── Meta-learner ──────────────────────────────────────────────────────
    _info("Training meta-learner (Logistic Regression on OOF features)…")
    meta_lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=random_state)
    meta_lr.fit(meta_X, y_train)
    _ok("Meta-learner training complete")

    return base_learners, meta_lr, scaler


# ===========================================================================
# EVALUATION
# ===========================================================================

def evaluate_model(
    base_learners: List[Any],
    meta_learner: Any,
    scaler: Any,
    X_test: Any,
    y_test: Any,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics on the test set.

    Returns
    -------
    metrics_bundle : dict compatible with app.py and dashboard/backend/main.py
    """
    import numpy as np
    from sklearn.metrics import (
        average_precision_score, roc_auc_score, f1_score,
        matthews_corrcoef, precision_score, recall_score,
        confusion_matrix, precision_recall_curve,
    )

    X_test_sc = scaler.transform(X_test)
    meta_X_test = np.column_stack([
        clf.predict_proba(X_test_sc)[:, 1] for clf in base_learners
    ]).astype(np.float32)
    y_prob = meta_learner.predict_proba(meta_X_test)[:, 1]

    # Default threshold
    y_pred_05 = (y_prob >= 0.5).astype(int)
    auprc = float(average_precision_score(y_test, y_prob))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    # Optimal threshold (F1-maximising)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = np.where(
        (precisions + recalls) == 0, 0.0,
        2 * precisions * recalls / (precisions + recalls + 1e-9)
    )
    best_idx = int(np.argmax(f1s[:-1]))
    opt_thresh = float(thresholds[best_idx])
    y_pred_opt = (y_prob >= opt_thresh).astype(int)

    def _metrics(y_pred: Any) -> Dict[str, float]:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        return {
            "AUPRC": auprc,
            "ROC_AUC": roc_auc,
            "F1": float(f1_score(y_test, y_pred, zero_division=0)),
            "MCC": float(matthews_corrcoef(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        }

    return {
        "threshold_0.5": _metrics(y_pred_05),
        "threshold_optimal": _metrics(y_pred_opt),
        "optimal_threshold_value": opt_thresh,
        "note": "Demo model trained on synthetic data — metrics are approximate",
    }


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a lightweight demo model for the Streamlit dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-samples", type=int, default=15000,
                   help="Total synthetic transactions to generate")
    p.add_argument("--fraud-rate", type=float, default=0.05,
                   help="Fraction of transactions that are fraud (>= 0.01 recommended)")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Number of parallel jobs")
    p.add_argument("--output-dir", type=str, default="results",
                   help="Directory to save model and metrics")
    return p.parse_args()


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    _section("Demo Model Creation — Credit Card Fraud Detection")
    print(f"  n_samples   : {args.n_samples:,}")
    print(f"  fraud_rate  : {args.fraud_rate:.1%}")
    print(f"  random_state: {args.random_state}")
    print(f"  output_dir  : {output_dir.resolve()}")

    # ── Check dependencies ─────────────────────────────────────────────────
    _section("1. Checking Dependencies")
    missing = []
    for pkg in ["sklearn", "xgboost", "imblearn", "joblib", "numpy", "pandas"]:
        try:
            __import__(pkg)
            _ok(pkg)
        except ImportError:
            missing.append(pkg)
            print(f"  [MISSING] {pkg}")
    if missing:
        print(f"\n  Install missing packages: pip install {' '.join(missing)}")
        sys.exit(1)

    # ── Generate synthetic data ────────────────────────────────────────────
    _section("2. Generating Synthetic Data")
    X, y = generate_synthetic_data(args.n_samples, args.fraud_rate, args.random_state)
    _ok(f"Dataset shape: {X.shape} | Fraud: {int(y.sum()):,} ({y.mean():.2%})")

    # ── Train / test split ─────────────────────────────────────────────────
    _section("3. Splitting Data")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.random_state
    )
    _ok(f"Train: {X_train.shape} | Test: {X_test.shape}")
    _ok(f"Train fraud: {int(y_train.sum()):,} | Test fraud: {int(y_test.sum()):,}")

    # ── Validate fraud sample count ────────────────────────────────────────
    if y_train.sum() < 6:
        print(
            f"\n  [ERROR] Only {int(y_train.sum())} fraud samples in training set — SMOTE requires at least 6.\n"
            f"  Increase --n-samples or --fraud-rate.\n"
            f"  Example: python scripts/create_demo_model.py --n-samples 15000 --fraud-rate 0.05"
        )
        sys.exit(1)

    # ── Train model ────────────────────────────────────────────────────────
    _section("4. Training Stacking Ensemble")
    base_learners, meta_learner, scaler = train_demo_model(
        X_train, y_train, args.n_jobs, args.random_state
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    _section("5. Evaluating on Test Set")
    metrics = evaluate_model(base_learners, meta_learner, scaler, X_test, y_test)
    m_opt = metrics["threshold_optimal"]
    m_05 = metrics["threshold_0.5"]
    print(f"\n  {'Metric':<12} {'@0.5':>8} {'@Optimal':>10}")
    print(f"  {'-'*32}")
    for key in ["AUPRC", "ROC_AUC", "F1", "MCC", "Precision", "Recall"]:
        print(f"  {key:<12} {m_05.get(key, 0):>8.4f} {m_opt.get(key, 0):>10.4f}")
    print(f"\n  Optimal threshold: {metrics['optimal_threshold_value']:.4f}")
    print(f"  TP={m_opt['TP']}, FP={m_opt['FP']}, TN={m_opt['TN']}, FN={m_opt['FN']}")

    # ── Save artefacts ─────────────────────────────────────────────────────
    _section("6. Saving Artefacts")
    import joblib
    import numpy as np

    # Build feature names (must match what app.py expects)
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount_log", "Hour"]

    model_bundle = {
        "base_learners": base_learners,
        "meta_learner": meta_learner,
        "scaler": scaler,
        "feature_names": feature_names,
        "optimal_threshold": metrics["optimal_threshold_value"],
        "training_note": "Demo model trained on synthetic data",
    }

    model_path = output_dir / "stacking_model.pkl"
    joblib.dump(model_bundle, model_path)
    _ok(f"Model saved: {model_path}")

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    _ok(f"Metrics saved: {metrics_path}")

    # ── Reproducibility check ──────────────────────────────────────────────
    _section("7. Reproducibility Check")
    reloaded = joblib.load(model_path)
    X_test_sc = reloaded["scaler"].transform(X_test)
    meta_X_test = np.column_stack([
        clf.predict_proba(X_test_sc)[:, 1] for clf in reloaded["base_learners"]
    ])
    y_prob_reload = reloaded["meta_learner"].predict_proba(meta_X_test)[:, 1]

    # Compute original predictions for comparison
    X_test_sc_orig = scaler.transform(X_test)
    meta_X_test_orig = np.column_stack([
        clf.predict_proba(X_test_sc_orig)[:, 1] for clf in base_learners
    ])
    y_prob_orig = meta_learner.predict_proba(meta_X_test_orig)[:, 1]

    if np.allclose(y_prob_orig, y_prob_reload, atol=1e-6):
        _ok("Reproducibility assertion PASSED — reloaded model is identical")
    else:
        max_diff = float(np.max(np.abs(y_prob_orig - y_prob_reload)))
        print(f"  [WARN] Max prediction difference: {max_diff:.2e} (acceptable if < 1e-4)")

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    _section("Complete!")
    print(f"  Total time : {elapsed:.1f}s")
    print(f"  Model path : {model_path.resolve()}")
    print(f"  Metrics    : {metrics_path.resolve()}")
    print(f"\n  Launch dashboard:")
    print(f"    streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
