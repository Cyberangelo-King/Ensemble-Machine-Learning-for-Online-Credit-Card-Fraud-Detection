"""
scripts/generate_synthetic_data.py
Generates a synthetic creditcard.csv for local development and CI.

The file mimics the structure of the Kaggle Credit Card Fraud Detection
dataset (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
  - Columns: Time, V1–V28, Amount, Class
  - 10,000 rows total
  - ~1% fraud (Class=1) — increased from 0.17% so that SMOTE has
    sufficient minority-class samples during model training.

No real cardholder data is used; all values are sampled from statistical
distributions chosen to resemble the real dataset at a high level.
"""

from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
TOTAL_ROWS: int = 10_000
FRAUD_RATIO: float = 0.01          # 1% fraud — up from 0.17% for SMOTE robustness
OUTPUT_PATH: str = "data/creditcard.csv"

# ---------------------------------------------------------------------------


def generate(
    n_rows: int = TOTAL_ROWS,
    fraud_ratio: float = FRAUD_RATIO,
    random_seed: int = RANDOM_SEED,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    """Generate and save the synthetic creditcard.csv.

    Parameters
    ----------
    n_rows:
        Total number of transactions.
    fraud_ratio:
        Fraction of rows that are fraudulent (Class=1).
    random_seed:
        NumPy random seed for reproducibility.
    output_path:
        Destination path relative to the working directory (usually repo root).

    Returns
    -------
    pd.DataFrame
        The generated dataframe (also written to *output_path*).
    """
    rng = np.random.default_rng(random_seed)

    n_fraud = max(int(n_rows * fraud_ratio), 50)  # guarantee at least 50 fraud rows
    n_legit = n_rows - n_fraud

    logger.info(
        "Generating %d transactions (%d legitimate, %d fraudulent — %.2f%%).",
        n_rows, n_legit, n_fraud, 100 * n_fraud / n_rows,
    )

    # ------------------------------------------------------------------
    # Time column — seconds elapsed from the first transaction
    # ------------------------------------------------------------------
    time_legit = rng.uniform(0, 172_800, n_legit)   # up to 48 h
    time_fraud = rng.uniform(0, 172_800, n_fraud)
    time_all = np.concatenate([time_legit, time_fraud])

    # ------------------------------------------------------------------
    # V1–V28 — PCA-like anonymous features
    # Legitimate transactions: standard normal
    # Fraudulent transactions: shifted means to make them separable
    # ------------------------------------------------------------------
    n_v_features = 28
    v_legit = rng.standard_normal((n_legit, n_v_features))

    # Shift several components for fraud to mimic real-world patterns
    v_fraud = rng.standard_normal((n_fraud, n_v_features))
    fraud_shifts = np.zeros(n_v_features)
    fraud_shifts[0] = -3.0   # V1  — strong negative shift in real data
    fraud_shifts[1] = 2.5    # V2
    fraud_shifts[2] = -2.0   # V3
    fraud_shifts[3] = 1.5    # V4
    fraud_shifts[9] = -2.0   # V10
    fraud_shifts[11] = 3.5   # V12
    fraud_shifts[13] = -2.5  # V14
    fraud_shifts[15] = 2.0   # V16
    v_fraud += fraud_shifts

    v_all = np.vstack([v_legit, v_fraud])

    # ------------------------------------------------------------------
    # Amount — log-normal, fraud amounts tend to be smaller
    # ------------------------------------------------------------------
    amount_legit = rng.lognormal(mean=4.5, sigma=1.5, size=n_legit)  # ~$90 median
    amount_legit = np.clip(amount_legit, 0.01, 25_691.16)            # real dataset max
    amount_fraud = rng.lognormal(mean=3.0, sigma=1.2, size=n_fraud)  # ~$20 median
    amount_fraud = np.clip(amount_fraud, 0.01, 2_125.87)
    amount_all = np.concatenate([amount_legit, amount_fraud])

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    labels = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    # ------------------------------------------------------------------
    # Assemble DataFrame
    # ------------------------------------------------------------------
    v_cols = {f"V{i}": v_all[:, i - 1] for i in range(1, n_v_features + 1)}
    df = pd.DataFrame(
        {"Time": time_all, **v_cols, "Amount": amount_all, "Class": labels}
    )

    # Shuffle rows so fraud is not all at the end
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Write to disk
    # ------------------------------------------------------------------
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logger.info("Saved %d rows to %s.", len(df), out_path)
    logger.info(
        "Class distribution: %s",
        df["Class"].value_counts().to_dict(),
    )

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generate()
