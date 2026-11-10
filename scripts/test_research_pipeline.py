"""Focused unit tests for the revised research experiment pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_experiment import (  # noqa: E402
    CONFIG,
    _pipeline,
    _split_train_validation_test,
    _get_supplementary_training_data,
    _build_cross_validator,
    _oof_validation_indices,
    compute_metrics,
    generate_oof_meta_features,
    load_and_engineer_features,
)


def test_feature_engineering_uses_elapsed_hours_and_sorts_time(tmp_path: Path) -> None:
    rows = 60
    frame = pd.DataFrame({
        **{f"V{i}": np.linspace(-1, 1, rows) for i in range(1, 29)},
        "Amount": np.linspace(0, 100, rows),
        "Time": np.arange(rows, dtype=float)[::-1] * 3600.0,
        "Class": np.tile([0, 1], rows // 2),
    })
    csv_path = tmp_path / "creditcard.csv"
    frame.to_csv(csv_path, index=False)

    features, labels = load_and_engineer_features(str(csv_path))

    assert features.attrs["ordered_by_time"] is True
    assert len(features) == rows
    assert len(labels) == rows
    assert np.all(np.diff(features["Hour"].to_numpy()) >= 0)
    np.testing.assert_allclose(
        features["Hour"].to_numpy(),
        np.arange(rows, dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_loader_rejects_infinite_feature_values(tmp_path: Path) -> None:
    rows = 60
    frame = pd.DataFrame({
        **{f"V{i}": np.linspace(-1, 1, rows) for i in range(1, 29)},
        "Amount": np.linspace(0, 100, rows),
        "Time": np.arange(rows, dtype=float),
        "Class": np.tile([0, 1], rows // 2),
    })
    frame.loc[0, "V7"] = np.inf
    csv_path = tmp_path / "creditcard_nonfinite.csv"
    frame.to_csv(csv_path, index=False)

    try:
        load_and_engineer_features(str(csv_path))
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("The loader must reject infinite feature values.")


def test_temporal_split_is_disjoint_and_keeps_order() -> None:
    X = np.arange(400, dtype=np.float32).reshape(200, 2)
    y = np.tile(np.array([0, 1], dtype=np.int8), 100)

    X_train, X_valid, X_test, y_train, y_valid, y_test = _split_train_validation_test(
        X, y, "temporal", seed=42, ordered_by_time=True
    )

    assert len(X_train) == 140
    assert len(X_valid) == 30
    assert len(X_test) == 30
    assert X_train[:, 0].max() < X_valid[:, 0].min()
    assert X_valid[:, 0].max() < X_test[:, 0].min()
    assert all(np.unique(part).tolist() == [0, 1] for part in (y_train, y_valid, y_test))


def test_supplementary_analyses_receive_training_partition_only() -> None:
    X = pd.DataFrame({
        "feature_a": np.arange(200, dtype=np.float32),
        "feature_b": np.arange(200, dtype=np.float32) * 2,
    })
    X.attrs["ordered_by_time"] = True
    y = pd.Series(np.tile([0, 1], 100), name="Class", dtype="int8")

    X_train, y_train = _get_supplementary_training_data(X, y, dict(CONFIG))

    assert len(X_train) == 140
    assert len(y_train) == 140
    assert X_train["feature_a"].max() == 139
    assert X_train.attrs["ordered_by_time"] is True
    assert set(y_train.unique()) == {0, 1}


def test_temporal_split_requires_chronological_loader_marker() -> None:
    X = np.arange(400, dtype=np.float32).reshape(200, 2)
    y = np.tile(np.array([0, 1], dtype=np.int8), 100)

    try:
        _split_train_validation_test(X, y, "temporal", seed=42, ordered_by_time=False)
    except ValueError as exc:
        assert "not marked as sorted" in str(exc)
    else:
        raise AssertionError("Temporal split must reject unverified input ordering.")


def test_temporal_cross_validation_is_forward_chaining_and_covers_only_future_blocks() -> None:
    y = np.tile(np.array([0, 1], dtype=np.int8), 300)
    cfg = dict(CONFIG)
    cfg["n_folds"] = 5

    cv = _build_cross_validator(y, cfg, "temporal", seed=42)
    splits = list(cv.split(np.arange(len(y))))
    validation_indices = _oof_validation_indices(cv, np.zeros((len(y), 1)), y)

    assert len(splits) == 5
    for train_idx, valid_idx in splits:
        assert train_idx.max() < valid_idx.min()
        assert np.unique(y[train_idx]).size == 2
        assert np.unique(y[valid_idx]).size == 2
    assert validation_indices.min() > 0
    assert validation_indices.max() == len(y) - 1
    assert len(np.unique(validation_indices)) == len(validation_indices)


def test_oof_meta_features_fit_complete_pipeline_and_disjoint_reference() -> None:
    rng = np.random.default_rng(42)
    n_meta = 120
    n_reference = 80
    y_meta = np.zeros(n_meta, dtype=np.int8)
    y_meta[:24] = 1
    X_meta = rng.normal(size=(n_meta, 5)).astype(np.float32)
    X_meta[y_meta == 1, 0] += 1.5

    y_reference = np.zeros(n_reference, dtype=np.int8)
    y_reference[:20] = 1
    X_reference = rng.normal(size=(n_reference, 5)).astype(np.float32)
    X_reference[y_reference == 1, 0] += 1.5

    cfg = dict(CONFIG)
    cfg["smote_sampling_strategy"] = 0.5
    pipeline = _pipeline(
        LogisticRegression(max_iter=500, solver="liblinear"),
        cfg,
        random_state=42,
    )

    oof = generate_oof_meta_features(
        X_meta,
        y_meta,
        [pipeline],
        StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        smote_ratio=0.5,
        n_jobs=1,
        random_state=42,
        reference_X=X_reference,
        reference_y=y_reference,
    )

    assert oof.shape == (n_meta, 1)
    assert np.isfinite(oof).all()
    assert np.all((oof >= 0.0) & (oof <= 1.0))
    assert pipeline.named_steps["smote"].sampling_strategy == 0.5


def test_metrics_report_primary_and_threshold_dependent_scores() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    probabilities = np.array([0.05, 0.2, 0.35, 0.55, 0.8, 0.95])

    result = compute_metrics(y_true, probabilities, threshold=0.5)

    assert 0.0 <= result["AUPRC"] <= 1.0
    assert 0.0 <= result["ROC_AUC"] <= 1.0
    assert result["TP"] == 3
    assert result["TN"] == 3
    assert result["FP"] == 0
    assert result["FN"] == 0
