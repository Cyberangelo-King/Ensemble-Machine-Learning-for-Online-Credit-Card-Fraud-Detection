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

### Deployment Steps

1. **Push Code to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Pre-train Model Locally**
   ```bash
   python src/run_experiment.py --data-path data/creditcard.csv
   ```
   
   Then commit the trained model:
   ```bash
   git add artifacts/stacking_model.joblib
   git commit -m "Add trained stacking model"
   git push origin main
   ```
   
   ⚠️ **Note:** The model is ~15 MB. Streamlit Cloud has storage limits; consider using external storage for larger models.

3. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Click "New app"
   - Select this GitHub repository
   - Set main file path: `app.py`
   - Click "Deploy"

4. **Secrets Management**
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
- **Local (< 50 MB):** Commit to repo ✓
- **Large models (> 50 MB):** Use external storage
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
```python
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error("Model not found. Please train the model first.")
    st.stop()
```

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

### Model file not found
- Ensure `artifacts/stacking_model.joblib` is committed
- Or download dataset and train model on deployment platform

### Slow SHAP computations
- Reduce sample size in `app.py` line 120
- Cache results with `@st.cache_resource`

---

## Useful Links

- Streamlit Cloud: https://streamlit.io/cloud
- Deployment docs: https://docs.streamlit.io/library/deploy
- Docker docs: https://docs.docker.com/
- Model registry: https://mlflow.org/