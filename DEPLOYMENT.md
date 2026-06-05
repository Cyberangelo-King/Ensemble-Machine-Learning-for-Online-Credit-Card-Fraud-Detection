# Streamlit Deployment Guide

This document explains how to deploy the fraud detection dashboard to production.

## Local Development

### Prerequisites
- Python 3.10+
- Git

### Steps
1. Clone the repository
2. Follow `SETUP.md` to configure environment and train model
3. Run: `streamlit run app.py`
4. Open browser to `http://localhost:8501`

---

## Streamlit Cloud Deployment

### Prerequisites
- GitHub account with this repository
- Streamlit account (free tier available)

### Best Option for a Sample Presentation

Use the dashboard's built-in demo mode. Streamlit Cloud does **not** run `setup.sh`, and `data/creditcard.csv` plus `artifacts/stacking_model.joblib` are intentionally ignored by Git. The current `app.py` handles missing files by using a synthetic demo dataset and deterministic demo model, so no Kaggle download or committed model artifact is required for a presentation.

If Streamlit Cloud shows `Startup error: Model file missing: artifacts/stacking_model.joblib`, the deployed app is running older code that still required a committed model file. Push the latest branch that includes demo mode, confirm Streamlit Cloud is deploying that branch with `app.py` as the main file, then reboot or redeploy the app. The repository includes `runtime.txt` to pin Streamlit Cloud to Python 3.12 for stable dependency installation.

### Deployment Steps

1. **Push the latest demo-mode code to GitHub**
   ```bash
   git push origin main
   ```

2. **Create or redeploy the Streamlit Cloud app**
   - Visit: https://streamlit.io/cloud
   - Click "New app" or open your existing app settings
   - Select this GitHub repository and the branch containing the latest `app.py`
   - Set main file path: `app.py`
   - Click "Deploy"; for an existing app, click "Reboot" or "Redeploy" after pushing

3. **Verify demo mode**
   - The app should load without `artifacts/stacking_model.joblib`
   - A banner should say demo mode is active
   - The Simulator, Metrics, Dataset, and Model Comparison pages should work for the presentation

### Optional Production Model Deployment

For production-quality model predictions, train locally on the real Kaggle CSV and upload the resulting model through a storage strategy that fits your deployment. The repository ignores these large generated files by default, so `setup.sh` artifacts are not automatically available on Streamlit Cloud.

```bash
python src/run_experiment.py --data-path data/creditcard.csv
```

Options after training:
- keep using demo mode for presentations,
- force-add a small demo artifact only if you explicitly want it in Git, or
- store larger production artifacts externally, such as S3, GCS, Azure Blob Storage, or a model registry.

### Secrets Management
- If using API keys: add to Streamlit Cloud Secrets
- Format: `.streamlit/secrets.toml`
```toml
[api]
kaggle_api_key = "your-key-here"
```

---

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run
```bash
docker build -t fraud-detection:latest .
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  fraud-detection:latest
```

### Docker Compose (Recommended)
```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./artifacts:/app/artifacts
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

Run with: `docker-compose up`

---

## Production Considerations

### 1. Model Storage
- **Sample presentation:** use built-in demo mode; do not commit `artifacts/stacking_model.joblib`.
- **Small intentional artifact (< 50 MB):** force-add only if you explicitly want the model in Git for a production-like demo.
- **Large or production models:** use external storage.
  - AWS S3
  - Google Cloud Storage
  - Azure Blob Storage

```python
# Example: Load from S3
import boto3
s3 = boto3.client('s3')
s3.download_file('bucket-name', 'stacking_model.joblib', 'artifacts/stacking_model.joblib')
```

### 2. Authentication
Add login layer if needed:
```python
import streamlit as st

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    
    if st.session_state.password_correct:
        return True
    
    password = st.text_input("Password:", type="password")
    if password == st.secrets["password"]:
        st.session_state.password_correct = True
        return True
    return False

if not check_password():
    st.stop()
```

### 3. Performance Optimization
```python
# Use caching effectively
@st.cache_resource(ttl=3600)  # Refresh every hour
def load_model(path: Path):
    return joblib.load(path)

# Limit SHAP computation
sample = df[FEATURES].sample(min(500, len(df)), random_state=42)  # Smaller sample
```

### 4. Monitoring
Add logging to track usage:
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Prediction made: proba={proba:.3f}, latency={dt:.2f}ms")
```

### 5. Error Handling
The dashboard should not stop just because demo artifacts are absent. Keep missing-model handling on the demo fallback path used by `app.py`: load the trained model when it exists, otherwise use the deterministic demo model and show the demo-mode banner.

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Inference latency (1 transaction) | 0.074 ms |
| Batch processing (1000 transactions) | 74 ms |
| Model size | ~15 MB |
| RAM usage (app) | ~200 MB (base) + 300 MB (SHAP) |

---

## Troubleshooting Deployment

### App crashes after deploy
- Check logs in Streamlit Cloud dashboard
- Verify all dependencies in `requirements.txt`
- Test locally first: `streamlit run app.py`

### `Startup error: Model file missing: artifacts/stacking_model.joblib`
- This is the old pre-demo-mode startup behavior. The latest `app.py` falls back to demo mode instead of stopping.
- Push the latest code to GitHub, then reboot or redeploy the Streamlit Cloud app.
- Confirm Streamlit Cloud is using the intended branch and `app.py` as the main file path.
- For a sample presentation, do not commit the ignored `artifacts/stacking_model.joblib`; let demo mode run.

### Slow SHAP computations
- Reduce sample size in `app.py` line 120
- Cache results with `@st.cache_resource`

---

## Useful Links

- Streamlit Cloud: https://streamlit.io/cloud
- Deployment docs: https://docs.streamlit.io/library/deploy
- Docker docs: https://docs.docker.com/
- Model registry: https://mlflow.org/