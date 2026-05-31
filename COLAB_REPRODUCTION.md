# Reproducing Results on Google Colab

This guide provides a detailed step-by-step approach to reproduce the fraud detection ensemble results using Google Colaboratory (Colab).

## Prerequisites
- Google account (free tier is sufficient)
- Kaggle account (to download the dataset)
- ~30-45 minutes for the full pipeline

---

## Step 1: Set Up Google Colab Environment

### 1.1 Create a New Notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **New notebook**
3. Name it: `Fraud_Detection_Reproduction`

### 1.2 Mount Google Drive (Optional but Recommended)
Add this cell to save artifacts and results to your Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Create a working directory
import os
os.makedirs('/content/drive/My Drive/fraud_detection', exist_ok=True)
work_dir = '/content/drive/My Drive/fraud_detection'
```

**Why?** Colab sessions timeout; Drive preserves your trained model.

---

## Step 2: Install Dependencies

Add this cell:

```python
# Install required packages
!pip install --upgrade pip
!pip install -q scikit-learn==1.3.0 xgboost==2.0.0 imbalanced-learn==0.11.0 shap==0.44.0 pandas==2.0.3 numpy==1.24.4 scipy==1.11.2 matplotlib==3.7.2 seaborn==0.12.2 joblib==1.3.2 pyyaml==6.0.1

print("✅ All dependencies installed!")
```

**Time:** ~2-3 minutes

---

## Step 3: Authenticate with Kaggle and Download Dataset

### 3.1 Get Kaggle API Credentials
1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings/account)
2. Click **Create New API Token** → Downloads `kaggle.json`
3. Keep this file safe (contains your credentials)

### 3.2 Upload and Configure Kaggle Credentials
Add this cell:

```python
from google.colab import files
print("Upload your kaggle.json file:")
uploaded = files.upload()

# Move credentials to proper location
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

print("✅ Kaggle credentials configured")
```

### 3.3 Download Dataset
Add this cell:

```python
!kaggle datasets download -d mlg-ulb/creditcardfraud
!unzip -q creditcardfraud.zip

# Verify dataset
import pandas as pd
df = pd.read_csv('creditcard.csv')
print(f"✅ Dataset loaded!")
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"   Class distribution:\n{df['Class'].value_counts()}")
```

**Expected output:**
```
✅ Dataset loaded!
   Shape: (284807, 31)
   Columns: ['Time', 'V1', 'V2', ..., 'V28', 'Amount', 'Class']
   Class distribution:
   0    284315
   1      492
```

---

## Step 4: Clone the Repository

Add this cell:

```python
!git clone https://github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection.git
%cd Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection

# Verify structure
!ls -la
```

---

## Step 5: Set Up Project Structure

Add this cell:

```python
import os
from pathlib import Path

# Create necessary directories
Path('data').mkdir(exist_ok=True)
Path('artifacts').mkdir(exist_ok=True)
Path('results').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)

# Move dataset to correct location
!mv ../creditcard.csv data/

# Verify
!ls -la data/
print("✅ Project structure ready")
```

---

## Step 6: Run the Full Experiment Pipeline

Add this cell:

```python
import sys
sys.path.insert(0, '/content/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection')

# Run the experiment
!python src/run_experiment.py \
  --data-path data/creditcard.csv \
  --output-dir results \
  --fig-dir figures
```

**What happens:**
- Loads 284,807 transactions
- Splits into train/test (80/20, stratified)
- Applies SMOTE to training data
- Tunes 3 base learners (RandomizedSearchCV, 50 iterations each)
- Trains stacking meta-learner
- Generates metrics and plots

**Expected time:** 15-25 minutes (depending on Colab GPU)

---

## Step 7: Verify Results

Add this cell:

```python
import json

# Load and display metrics
with open('results/metrics.json', 'r') as f:
    metrics = json.load(f)

print("=" * 50)
print("FRAUD DETECTION EXPERIMENT RESULTS")
print("=" * 50)
print(f"AUPRC:           {metrics['auprc']:.4f}")
print(f"F1-Score:        {metrics['f1']:.4f}")
print(f"MCC:             {metrics['mcc']:.4f}")
print(f"ROC-AUC:         {metrics['roc_auc']:.4f}")
print(f"Mean Latency:    {metrics['mean_latency_ms']:.4f} ms/txn")
print("\nTarget Benchmarks:")
print(f"AUPRC Target:    {metrics['target_reported']['auprc']:.4f}")
print(f"F1 Target:       {metrics['target_reported']['f1']:.4f}")
print(f"MCC Target:      {metrics['target_reported']['mcc']:.4f}")
print("=" * 50)

# Check differences
diff_auprc = abs(metrics['auprc'] - metrics['target_reported']['auprc'])
diff_f1 = abs(metrics['f1'] - metrics['target_reported']['f1'])
diff_mcc = abs(metrics['mcc'] - metrics['target_reported']['mcc'])

print(f"\n✅ Reproducibility Check:")
print(f"   AUPRC Δ:  {diff_auprc:.4f} ({diff_auprc/metrics['target_reported']['auprc']*100:.2f}%)")
print(f"   F1 Δ:     {diff_f1:.4f} ({diff_f1/metrics['target_reported']['f1']*100:.2f}%)")
print(f"   MCC Δ:    {diff_mcc:.4f} ({diff_mcc/metrics['target_reported']['mcc']*100:.2f}%)")
```

---

## Step 8: Visualize Generated Plots

Add this cell:

```python
import matplotlib.pyplot as plt
from PIL import Image
import os

# List all generated figures
fig_dir = 'figures'
figures = sorted([f for f in os.listdir(fig_dir) if f.endswith('.png')])

print(f"Generated {len(figures)} figures:")
for fig in figures:
    print(f"  - {fig}")

# Display each figure
for fig_name in figures:
    fig_path = os.path.join(fig_dir, fig_name)
    img = Image.open(fig_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(fig_name, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
```

**Expected plots:**
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- SHAP Summary Plot
- Model Comparison

---

## Step 9: Save Artifacts to Google Drive

Add this cell (if you mounted Drive):

```python
import shutil

work_dir = '/content/drive/My Drive/fraud_detection'

# Copy results and figures
!cp -r results/* {work_dir}/
!cp -r figures/* {work_dir}/
!cp artifacts/stacking_model.joblib {work_dir}/

print(f"✅ Artifacts saved to Google Drive: {work_dir}")
print("\nSaved files:")
!ls -lh {work_dir}/
```

---

## Step 10: Optional - Deploy Streamlit Dashboard

If you want to test the Streamlit app:

### 10.1 Install Streamlit
```python
!pip install -q streamlit pyngrok
```

### 10.2 Create App Script
```python
# Create a simplified app for Colab
with open('app_colab.py', 'w') as f:
    f.write("""
import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Fraud Detection Results", layout="wide")
st.title("Credit Card Fraud Detection - Results")

# Load metrics
with open('results/metrics.json') as f:
    metrics = json.load(f)

col1, col2, col3, col4 = st.columns(4)
col1.metric("AUPRC", f"{metrics['auprc']:.4f}")
col2.metric("F1-Score", f"{metrics['f1']:.4f}")
col3.metric("MCC", f"{metrics['mcc']:.4f}")
col4.metric("Latency", f"{metrics['mean_latency_ms']:.3f} ms")

st.success("✅ Experiment successfully reproduced!")
    """)
```

---

## Complete Colab Notebook Template

Here's the complete notebook in one block:

```python
# ============================================
# FRAUD DETECTION REPRODUCTION - GOOGLE COLAB
# ============================================

# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Install Dependencies
!pip install --upgrade pip -q
!pip install -q scikit-learn==1.3.0 xgboost==2.0.0 imbalanced-learn==0.11.0 shap==0.44.0 pandas==2.0.3 numpy==1.24.4 scipy==1.11.2 matplotlib==3.7.2 seaborn==0.12.2 joblib==1.3.2

# Step 3: Authenticate Kaggle
from google.colab import files
print("Upload kaggle.json:")
uploaded = files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Step 4: Download Dataset
!kaggle datasets download -d mlg-ulb/creditcardfraud -q
!unzip -q creditcardfraud.zip
print("✅ Dataset ready")

# Step 5: Clone Repository
!git clone -q https://github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection.git
%cd Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection

# Step 6: Setup Directories
from pathlib import Path
for d in ['data', 'artifacts', 'results', 'figures']:
    Path(d).mkdir(exist_ok=True)
!mv ../creditcard.csv data/

# Step 7: Run Pipeline
!python src/run_experiment.py --data-path data/creditcard.csv --output-dir results --fig-dir figures

# Step 8: Display Results
import json
with open('results/metrics.json') as f:
    metrics = json.load(f)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
for key in ['auprc', 'f1', 'mcc']:
    print(f"{key.upper()}: {metrics[key]:.4f} (Target: {metrics['target_reported'][key]:.4f})")
print("="*50)

# Step 9: Display Plots
from PIL import Image
import matplotlib.pyplot as plt
import os

for fig in sorted(os.listdir('figures')):
    if fig.endswith('.png'):
        plt.figure(figsize=(10, 6))
        plt.imshow(Image.open(f'figures/{fig}'))
        plt.axis('off')
        plt.title(fig)
        plt.show()

# Step 10: Save to Drive
!cp -r results/* /content/drive/My\ Drive/
!cp artifacts/stacking_model.joblib /content/drive/My\ Drive/
print("\n✅ All artifacts saved to Google Drive")
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Kaggle auth fails** | Ensure `kaggle.json` is uploaded; check credentials format |
| **Out of memory** | Reduce dataset size or use Colab Pro for more RAM |
| **SHAP is slow** | Normal behavior; first run caches results (~5 min) |
| **RandomizedSearchCV takes too long** | Colab's CPU is slower than local; be patient (~20 min) |
| **Timeout after 12 hours** | Save artifacts to Google Drive frequently |

---

## Expected Timeline

| Step | Time |
|------|------|
| Setup & install | 5 min |
| Download dataset | 2 min |
| Clone repo | 1 min |
| Run pipeline | **20-25 min** |
| Visualize & save | 5 min |
| **Total** | **~35-40 min** |

---

## Verification Checklist

- [ ] All dependencies installed successfully
- [ ] Dataset downloaded (284,807 rows)
- [ ] Repository cloned
- [ ] Pipeline executed without errors
- [ ] Metrics within 2% of targets (AUPRC, F1, MCC)
- [ ] 5 figures generated (confusion matrix, ROC, PR, SHAP, comparison)
- [ ] Model saved to `artifacts/stacking_model.joblib`
- [ ] Results JSON written to `results/metrics.json`
- [ ] Artifacts backed up to Google Drive

---

## Next Steps

After reproduction:
1. **Analyze SHAP plots** to understand feature importance
2. **Experiment with hyperparameters** in `src/run_experiment.py`
3. **Test the Streamlit dashboard** with the trained model
4. **Publish your results** if modifications are made
5. **Document findings** in a Colab notebook

---

## References

- [Google Colab Documentation](https://colab.research.google.com/notebooks/welcome.ipynb)
- [Kaggle API Docs](https://github.com/Kaggle/kaggle-api)
- [Repository](https://github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection)
- [METHODOLOGY.md](https://github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection/blob/main/METHODOLOGY.md)
