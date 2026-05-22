# Researcher Implementation Guide

This guide explains how another researcher can clone, run, validate, and extend this project.

## 1) Clone and Environment Setup
```bash
git clone https://github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection.git
cd Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Data Acquisition and Placement
1. Download the CCFD dataset from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place the file at `data/creditcard.csv`.
3. Confirm expected shape (~284,807 rows, 31 columns):
```bash
python - <<'PY'
import pandas as pd
p='data/creditcard.csv'
df=pd.read_csv(p)
print(df.shape)
print(df['Class'].value_counts())
PY
```

## 3) Run the Full Experiment
```bash
python src/run_experiment.py \
  --data-path data/creditcard.csv \
  --output-dir results \
  --fig-dir figures
```

Outputs:
- `results/metrics.json`: observed metrics and reference targets.
- `figures/*.png`: 300-DPI figures.

## 4) Validate Reproducibility
- The code fixes random seeds (`RANDOM_STATE=42`) and uses deterministic split/CV settings.
- Re-run the same command at least 3 times and compare `results/metrics.json`.
- If results drift, pin platform details (Python, OS, BLAS/OpenMP versions) in your lab notes.

## 5) Interpretability Workflow (SHAP)
- SHAP TreeExplainer is run on the tuned XGBoost learner.
- Use `figures/shap_summary.png` to inspect dominant features.
- Compare whether top contributors align with published findings (e.g., V17, V14, V12, V10).

## 6) How to Extend for New Research
- Swap meta learner or add calibrated decision thresholding.
- Add temporal split validation to mimic real deployment chronology.
- Add probability calibration (Platt/Isotonic) and cost-sensitive thresholding.
- Add external datasets for cross-domain robustness.

## 7) Reporting Checklist for Publication
- Include AUPRC, F1, MCC, confusion matrix, and latency.
- Report class distribution and leakage controls.
- Describe SMOTE usage strictly on training folds.
- Provide exact package versions and random seed.
- Publish figure-generation scripts and raw metric JSON.
