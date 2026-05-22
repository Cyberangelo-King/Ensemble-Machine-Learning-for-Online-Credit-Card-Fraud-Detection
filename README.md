# Ensemble Machine Learning for Online Credit Card Fraud Detection

> Academic implementation of a stacking ensemble for highly imbalanced fraud detection, based on the research project by **Boyejo Faith Akinola (CSC/20/4843, FUTA)** under the supervision of **Dr. E. Ajulo**.

## Executive Summary
This repository provides a production-ready and reproducible implementation of an ensemble fraud detection pipeline using Logistic Regression, Random Forest, and XGBoost base learners with a Logistic Regression meta-classifier. The project targets online fraud detection constraints and emphasizes robust imbalanced-learning metrics (AUPRC, F1, MCC) over raw accuracy.

## Table of Contents
1. [Research Objectives](#research-objectives)
2. [Methodology Overview](#methodology-overview)
3. [Repository Structure](#repository-structure)
4. [Dataset Access](#dataset-access)
5. [Quickstart](#quickstart)
6. [Reproducibility](#reproducibility)
7. [Results and Targets](#results-and-targets)
8. [Visualizations](#visualizations)
9. [Attribution](#attribution)
10. [License](#license)

## Research Objectives
- Detect fraudulent transactions in a severely imbalanced setting.
- Optimize for meaningful fraud metrics (AUPRC/F1/MCC).
- Benchmark real-time inference latency against operational targets (100–500 ms budget).
- Provide interpretable model behavior using SHAP.

## Methodology Overview
- Data split: stratified train/test.
- Leakage-safe imbalance handling: SMOTE only on training partitions at 1:10 minority-to-majority ratio.
- Hyperparameter optimization: RandomizedSearchCV with 50 sampled configurations per base learner.
- Ensemble learning: stacking classifier.
- Evaluation: AUPRC, F1-score, MCC, confusion matrix, ROC/PR curves, and per-transaction latency.

See [METHODOLOGY.md](METHODOLOGY.md) for full details.

## Repository Structure
- `src/run_experiment.py`: End-to-end experiment runner.
- `scripts/download_data_instructions.md`: Dataset retrieval instructions.
- `METHODOLOGY.md`: Detailed research methodology and CRISP-DM alignment.
- `RESULTS.md`: Reported and reproducibility-focused performance summary.
- `figures/`: Programmatically generated plots (300 DPI).
- `results/`: Machine-readable experiment outputs.
- `notebooks/`: Optional walkthrough notebook scaffolding.

## Dataset Access
Dataset: **Credit Card Fraud Detection** (284,807 European card transactions).
1. Create a Kaggle account.
2. Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
3. Place `creditcard.csv` into `data/`.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py --data-path data/creditcard.csv --output-dir results --fig-dir figures
```


## How Another Researcher Can Implement This Project
A practical, replication-focused guide is provided in [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md), including:
- environment setup with Python 3.10,
- exact dataset placement and verification,
- full experiment execution,
- reproducibility checks,
- interpretability workflow,
- extension ideas for follow-on publications.

## Reproducibility
- Global random seed is fixed (`RANDOM_STATE=42`).
- Version-pinned dependencies in `requirements.txt`.
- Deterministic splits and CV folds.
- Metrics are exported to `results/metrics.json`.

## Results and Targets
Reported benchmark targets from the project report:
- AUPRC: **0.903**
- F1-score: **0.881**
- MCC: **0.884**
- Mean latency: **0.074 ms/transaction**

Reproduce with the provided script and compare to target values in output JSON.

## Visualizations
The pipeline generates publication-quality (300 DPI):
- Confusion matrix heatmap
- ROC curve
- Precision-Recall curve
- SHAP summary plot
- Model performance comparison chart

## Attribution
See [ATTRIBUTION.md](ATTRIBUTION.md).

## License
This project is released under the MIT License.
