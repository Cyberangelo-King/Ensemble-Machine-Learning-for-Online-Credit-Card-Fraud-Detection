# Ensemble Machine Learning for Online Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection/ci.yml?label=CI&logo=github)](https://github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection/actions)
[![AUPRC](https://img.shields.io/badge/AUPRC-0.903-orange)](RESULTS.md)
[![F1](https://img.shields.io/badge/F1-0.881-red)](RESULTS.md)

A **research-grade, production-deployable stacking ensemble** for real-time credit card fraud detection, achieving AUPRC 0.903 and MCC 0.884 on the Kaggle Credit Card Fraud dataset. The system combines Logistic Regression, Random Forest, and XGBoost base learners with a Logistic Regression meta-learner, using leakage-free SMOTE-inside-CV oversampling and full statistical validation across 5 stability runs.

---

## Architecture Diagram

```
                   ┌─────────────────────────────────────────────────┐
                   │           INPUT: Transaction Features            │
                   │  V1–V28 (PCA) + Amount_log + Hour (engineered)  │
                   └──────────────────────┬──────────────────────────┘
                                          │
                              StandardScaler (fit on train only)
                                          │
                              SMOTE (applied INSIDE each CV fold)
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
   ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
   │  BASE LEARNER 1 │        │  BASE LEARNER 2 │        │  BASE LEARNER 3 │
   │ Logistic Regr.  │        │  Random Forest  │        │    XGBoost      │
   │ C=best, saga    │        │  500 trees      │        │  early stopping │
   └────────┬────────┘        └────────┬────────┘        └────────┬────────┘
            │  P(fraud|x)              │  P(fraud|x)              │  P(fraud|x)
            └───────────────────┬──────┘──────────────────────────┘
                                │
              OOF Meta-Features: [lr_prob, rf_prob, xgb_prob]
                                │
                     ┌──────────▼──────────┐
                     │  META-LEARNER        │
                     │  Logistic Regression │
                     │  trained on OOF probs│
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │  Threshold Opt.      │
                     │  F1-maximising on    │
                     │  validation set      │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │   OUTPUT: P(fraud)   │
                     │   Binary prediction  │
                     └─────────────────────┘

   Layer 0: Base Learners  →  Layer 1: Meta-Learner  →  Calibrated Output
```

---

## Results

```
┌─────────────────────────────────────────────────────────────────┐
│                   KEY RESULTS SUMMARY                           │
├──────────────┬─────────────────────────────────────────────────┤
│  AUPRC       │  0.9030  (vs. XGBoost alone: 0.890)            │
│  F1-Score    │  0.8810  (optimal threshold: ~0.38)             │
│  MCC         │  0.8840  (near-perfect agreement)               │
│  ROC-AUC     │  0.9791                                         │
│  Precision   │  0.9012                                         │
│  Recall      │  0.8621                                         │
│  Latency     │  <2 ms per transaction (CPU)                    │
└──────────────┴─────────────────────────────────────────────────┘
```

See [RESULTS.md](RESULTS.md) for the full results including ablation study, stability analysis, and baseline comparisons.

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection.git
cd Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection
bash setup.sh

# 2. Generate demo data and model (no Kaggle required)
python scripts/create_demo_model.py

# 3. Launch the Streamlit dashboard
streamlit run app.py
```

For the full research experiment with the real Kaggle dataset:
```bash
# Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place it in data/creditcard.csv, then:
python src/run_experiment.py --data-path data/creditcard.csv --output-dir results/ --fig-dir figures/
```

---

## Full Experiment CLI Reference

The main pipeline supports extensive configuration via command-line flags:

```bash
python src/run_experiment.py \
    --data-path data/creditcard.csv \   # Path to creditcard.csv
    --output-dir results/ \             # Model, metrics, checkpoints
    --fig-dir figures/ \                # Exported figures
    --n-jobs -1 \                       # Parallelism (-1 = all cores)
    --n-iter-rf 30 \                    # RF RandomizedSearch iterations
    --n-iter-xgb 50 \                   # XGBoost RandomizedSearch iterations
    --smote-ratio 0.15 \                # SMOTE minority:majority ratio
    --n-folds 5 \                       # StratifiedKFold folds
    --n-stability-runs 5 \              # Stability analysis runs
    --skip-ablation \                   # Optional: skip ablation study
    --skip-learning-curves              # Optional: skip learning curves
```

**Outputs produced:**
| Artefact | Path | Description |
|---|---|---|
| Model bundle | `results/stacking_model.pkl` | Full pipeline (scalers + all models) |
| Metrics | `results/metrics.json` | JSON with all metrics at both thresholds |
| Stability | `results/stability_report.json` | Mean ± std across runs |
| Ablation | `results/ablation_results.csv` | Per-configuration metrics |
| ROC curve | `figures/roc_curve.png` | ROC with AUC annotation |
| PR curve | `figures/pr_curve.png` | Precision-Recall curve |
| Confusion matrix | `figures/confusion_matrix.png` | Heatmap at optimal threshold |
| Calibration | `figures/calibration_curve.png` | Reliability diagram |
| SHAP | `figures/shap_summary.png` | SHAP beeswarm (meta-learner) |
| Permutation | `figures/permutation_importance.png` | Permutation importance |
| Learning curves | `figures/learning_curves.png` | AUPRC vs training size |
| Experiment log | `results/logs/experiment_*.log` | Timestamped execution log |

---

## Dashboard

The interactive Streamlit dashboard requires no Kaggle data — it uses the demo model.

```bash
# Ensure demo model is created first
python scripts/create_demo_model.py

# Launch dashboard
streamlit run app.py
```

**Dashboard Pages:**
- **Home** — Project overview, key metrics summary cards, and quick-start guide
- **Transaction Simulator** — Enter transaction features and get real-time fraud probability
- **SHAP Explanations** — SHAP waterfall chart showing per-feature contribution for any transaction
- **Metrics** — Model performance visualisations (ROC, PR curve, confusion matrix)
- **Dataset** — Dataset statistics, class distribution, feature distributions
- **Model Comparison** — Side-by-side comparison of all base learners vs ensemble

**Full-stack deployment** (FastAPI + React frontend):
```bash
# Terminal 1: Start FastAPI backend
cd dashboard/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start React frontend
cd dashboard/frontend
npm install && npm run dev
```

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 |
| Fraud cases | 492 |
| Fraud rate | 0.172% (highly imbalanced) |
| Time span | 2 days of European cardholder transactions |
| Features | V1–V28: PCA-anonymised components; Amount; Time |
| Engineered | `Amount_log = log1p(Amount)`, `Hour = (Time % 86400) / 3600` |
| Labels | Class: 0 = legitimate, 1 = fraud |

**Class imbalance handling:** SMOTE (Synthetic Minority Over-sampling Technique) is applied with `sampling_strategy=0.15` strictly **inside** each cross-validation fold to prevent data leakage. See [METHODOLOGY.md](METHODOLOGY.md) for a detailed explanation.

---

## Project Structure

```
Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection/
│
├── src/
│   └── run_experiment.py          # Main ML pipeline (~850 lines, full research pipeline)
│
├── dashboard/
│   ├── backend/
│   │   └── main.py                # FastAPI: POST /predict, GET /explain, WS /ws/stream
│   └── frontend/
│       ├── src/
│       │   ├── App.jsx            # React entry: dark mode, WebSocket, SHAP waterfall
│       │   └── components/        # TransactionForm, MetricsCard, SHAPChart, etc.
│       ├── package.json
│       └── vite.config.js
│
├── scripts/
│   ├── create_demo_model.py       # Train lightweight demo model on synthetic data
│   ├── generate_synthetic_data.py # Generate 10k-row synthetic creditcard.csv
│   └── test_backend.py            # pytest: REST + WebSocket smoke tests
│
├── data/                          # (gitignored) Place creditcard.csv here
│   └── .gitkeep
│
├── results/                       # (gitignored) Model, metrics, checkpoints
│   ├── stacking_model.pkl
│   ├── metrics.json
│   ├── stability_report.json
│   ├── ablation_results.csv
│   └── checkpoints/
│       ├── lr_best.pkl
│       ├── rf_best.pkl
│       └── xgb_best.pkl
│
├── figures/                       # (gitignored) Exported plots
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── confusion_matrix.png
│   ├── calibration_curve.png
│   ├── shap_summary.png
│   ├── permutation_importance.png
│   └── learning_curves.png
│
├── app.py                         # Streamlit dashboard (standalone, no Kaggle needed)
├── requirements.txt               # All Python dependencies, dual-versioned
├── setup.sh                       # One-command environment setup
├── render.yaml                    # Render.com deployment config
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline
├── README.md
├── METHODOLOGY.md                 # Detailed methodology (CRISP-DM, maths, algorithms)
├── RESULTS.md                     # Full experimental results
├── DEPLOYMENT.md                  # Deployment guide (local + Render)
└── LICENSE
```

---

## Technical Architecture

### Stacking Ensemble

Stacking (stacked generalisation) trains a **meta-learner** on the out-of-fold (OOF) predictions of base learners rather than directly on the raw features. This prevents the meta-learner from simply memorising the strongest base learner's predictions.

**Why three diverse learners?**

| Learner | Strength | Weakness |
|---|---|---|
| Logistic Regression | Calibrated probabilities, fast | Assumes linearity |
| Random Forest | Robust, low variance, handles nonlinearity | Slower, less precise probability calibration |
| XGBoost | Highest raw performance, handles interactions | Prone to overfitting without tuning |

Stacking these three gives the meta-learner access to complementary "views" of the data, allowing it to learn which learner to trust in which regions of feature space.

### SMOTE-Inside-CV (Leakage Prevention)

Applying SMOTE before splitting into folds would allow synthetic minority samples — derived from training data — to appear in validation folds. This creates **data leakage**, inflating validation AUPRC by up to ~0.04 points.

The correct approach (implemented here):

```
for fold in k_folds:
    X_train_fold, y_train_fold = ...
    X_val_fold,   y_val_fold   = ...
    
    scaler.fit(X_train_fold)              # fit on training fold ONLY
    X_train_scaled = scaler.transform(X_train_fold)
    X_val_scaled   = scaler.transform(X_val_fold)   # no leakage
    
    X_train_smote, y_train_smote = SMOTE().fit_resample(X_train_scaled, y_train_fold)
    # SMOTE never sees X_val_fold
    
    base_learner.fit(X_train_smote, y_train_smote)
    oof_preds[val_idx] = base_learner.predict_proba(X_val_scaled)[:, 1]
```

### OOF Meta-Feature Generation

```
Algorithm: OOF Meta-Feature Generation
Input:  X (n×p), y (n,), base_learners L, K folds
Output: meta_X (n × |L|)

meta_X ← zeros(n, |L|)
for k = 1 to K:
    train_idx, val_idx ← fold_k(X, y)
    X_tr_res, X_val_sc, y_tr_res ← preprocess(X[train_idx], y[train_idx])
    for j, learner in enumerate(L):
        learner_k ← clone(learner)
        learner_k.fit(X_tr_res, y_tr_res)
        meta_X[val_idx, j] ← learner_k.predict_proba(X_val_sc)[:, 1]

meta_learner.fit(meta_X, y)
```

---

## Citation

If you use this project in your research or coursework, please cite:

```bibtex
@misc{angelo2024fraud,
  author    = {Angelo, Cyberangelo-King},
  title     = {Ensemble Machine Learning for Online Credit Card Fraud Detection},
  year      = {2024},
  publisher = {GitHub},
  url       = {https://github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection}
}
```

**Dataset citation:**
```
Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015).
Calibrating probability with undersampling for unbalanced classification.
In 2015 IEEE Symposium Series on Computational Intelligence (SSCI).
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **Dataset**: Machine Learning Group — ULB (Université Libre de Bruxelles), available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **SHAP library**: Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
- **imbalanced-learn**: Lemaître et al. (2017), JMLR
- **XGBoost**: Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System"
- **scikit-learn**: Pedregosa et al. (2011), JMLR

---

*Final Year Project — [Your University Name] | Academic Year 2024–2025*
