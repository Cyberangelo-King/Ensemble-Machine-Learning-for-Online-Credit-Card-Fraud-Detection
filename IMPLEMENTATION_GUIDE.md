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

## 8) Google Colab Setup

You can run this project directly in Google Colab for free cloud-based experimentation.

### Quick Start
1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Run the following cells:

**Cell 1: Clone Repository and Install Dependencies**
```python
import os
os.chdir('/content')
!git clone https://github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection.git
os.chdir('Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection')
!pip install -r requirements.txt
```

**Cell 2: Mount Google Drive (for persistent storage)**
```python
from google.colab import drive
drive.mount('/content/drive')

# Create a symlink to your Colab working directory
!mkdir -p /content/drive/MyDrive/fraud_detection
!ln -s /content/drive/MyDrive/fraud_detection /content/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection/colab_output
```

**Cell 3: Download Dataset from Kaggle**
```python
# Install kaggle CLI
!pip install kaggle

# Upload kaggle.json to Colab (visit https://www.kaggle.com/settings/account for API token)
# Or configure manually:
from google.colab import files
files.upload()  # Upload your kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
!kaggle datasets download -d mlg-ulb/creditcardfraud
!unzip -o creditcardfraud.zip -d data/
```

**Cell 4: Verify Data**
```python
import pandas as pd
df = pd.read_csv('data/creditcard.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['Class'].value_counts()}")
```

**Cell 5: Run the Full Experiment**
```python
!python src/run_experiment.py \
  --data-path data/creditcard.csv \
  --output-dir colab_output/results \
  --fig-dir colab_output/figures
```

**Cell 6: Download Results**
```python
# Visualize results
import matplotlib.pyplot as plt
from PIL import Image
import os

fig_dir = 'colab_output/figures'
if os.path.exists(fig_dir):
    for fig_file in sorted(os.listdir(fig_dir)):
        if fig_file.endswith('.png'):
            img = Image.open(os.path.join(fig_dir, fig_file))
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(fig_file)
            plt.tight_layout()
            plt.show()

# Download metrics
!ls -la colab_output/results/
```

### Important Notes
- **GPU**: Google Colab provides free GPU access. Enable it via `Runtime > Change runtime type > GPU`
- **Storage**: Colab instances have limited storage (~12 GB). Use Google Drive for persistent storage
- **Session Duration**: Colab instances disconnect after ~12 hours of inactivity. Save your results to Google Drive
- **Dataset Size**: The creditcard dataset is ~150 MB. Ensure you have sufficient quota on Kaggle and Google Drive
- **Timeout**: Long-running experiments may timeout. Monitor execution and save checkpoints if needed

### Troubleshooting
- **Kaggle API Error**: Ensure `kaggle.json` is properly configured and has 600 permissions
- **Memory Issues**: Use `--sample-size 0.5` flag to run on 50% of data for testing
- **Import Errors**: Restart the kernel (`Ctrl+M .`) and re-run installation cells
