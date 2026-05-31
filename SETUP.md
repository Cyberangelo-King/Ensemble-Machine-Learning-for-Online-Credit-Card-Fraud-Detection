# Setup & Deployment Guide

This guide walks through preparing the Streamlit dashboard for deployment.

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Obtain the Dataset
The dataset is **NOT included** in the repository (too large for GitHub).

**Option A: From Kaggle (Recommended)**
1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Sign in and download `creditcard.csv`
3. Place at: `data/creditcard.csv`

**Option B: Using Kaggle CLI**
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

### 3. Train the Stacking Ensemble Model
```bash
python src/run_experiment.py \
  --data-path data/creditcard.csv \
  --output-dir results \
  --fig-dir figures
```

**Expected Output:**
- `artifacts/stacking_model.joblib` - Trained model (~15 MB)
- `results/metrics.json` - Evaluation metrics
- `figures/*.png` - Performance plots

**Training Time:** ~10-20 minutes (depends on hardware)

### 4. Verify Dataset Structure
```bash
python - <<'EOF'
import pandas as pd
df = pd.read_csv('data/creditcard.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Class distribution:\n{df['Class'].value_counts()}")
EOF
```

### 5. Launch Streamlit Dashboard
```bash
streamlit run app.py
```

The app will open at: `http://localhost:8501`

---

## Directory Structure
```
├── app.py                          # Streamlit dashboard (entry point)
├── requirements.txt                # Python dependencies
├── data/
│   └── creditcard.csv             # [DOWNLOAD] Credit card fraud dataset (284KB)
├── artifacts/
│   └── stacking_model.joblib      # [GENERATED] Trained model (15 MB)
├── results/
│   └── metrics.json               # [GENERATED] Evaluation metrics
├── figures/
│   ├── confusion_matrix.png       # [GENERATED] Performance plots
│   ├── pr_curve.png
│   ├── roc_curve.png
│   └── shap_summary.png
├── src/
│   └── run_experiment.py          # Model training pipeline
├── scripts/
│   └── download_data_instructions.md
└── notebooks/                      # Jupyter notebooks (optional)
```

---

## Troubleshooting

### Issue: "Model file missing: artifacts/stacking_model.joblib"
**Solution:** Run the training pipeline first:
```bash
python src/run_experiment.py --data-path data/creditcard.csv
```

### Issue: "Dataset missing: data/creditcard.csv"
**Solution:** Download from Kaggle and place at `data/creditcard.csv`

### Issue: Streamlit port already in use
**Solution:** Run on a different port:
```bash
streamlit run app.py --server.port 8502
```

### Issue: SHAP computations are slow
**Solution:** This is normal for SHAP TreeExplainer on large samples. 
- First run computes and caches; subsequent runs are fast
- To reduce computation time, modify `app.py` line 120:
  ```python
  sample = df[FEATURES].sample(min(500, len(df)), random_state=42)  # Change 1000 to 500
  ```

---

## Environment Variables (Optional)

For production Streamlit deployments, create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[logger]
level = "info"

[client]
toolbarMode = "minimal"
```

---

## Performance Notes

**Inference Latency:**
- Single transaction: ~0.074 ms
- Batch (1000 transactions): ~74 ms

**Memory Usage:**
- Loaded model: ~50 MB RAM
- Full dashboard with SHAP: ~500 MB RAM

---

## Next Steps

1. ✅ Deploy to Streamlit Cloud: https://streamlit.io/cloud
2. ✅ Add authentication (Streamlit Secrets)
3. ✅ Connect to live transaction data
4. ✅ Export predictions to database

See `IMPLEMENTATION_GUIDE.md` for research-oriented extensions.