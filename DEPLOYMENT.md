# Deployment Guide: Ensemble ML for Credit Card Fraud Detection

> **Document type:** Deployment & Operations Guide  
> **Project:** Final Year Computer Science Project  
> **Version:** 2.0  

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Local Development — Streamlit Only](#2-local-development--streamlit-only)
3. [Local Development — Full Stack](#3-local-development--full-stack)
4. [Render.com Deployment](#4-rendercom-deployment)
5. [Environment Variables Reference](#5-environment-variables-reference)
6. [Troubleshooting](#6-troubleshooting)
7. [Performance Tuning](#7-performance-tuning)

---

## 1. Prerequisites

### Required Software

| Software | Version | Purpose | Install |
|---|---|---|---|
| Python | ≥ 3.9 | ML pipeline + Streamlit + FastAPI | [python.org](https://www.python.org/) |
| pip | ≥ 21.0 | Python package manager | Bundled with Python |
| Node.js | ≥ 18.0 | React frontend | [nodejs.org](https://nodejs.org/) |
| npm | ≥ 9.0 | JavaScript package manager | Bundled with Node.js |
| Git | ≥ 2.30 | Version control | [git-scm.com](https://git-scm.com/) |

### Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 16 GB |
| Disk (demo only) | 2 GB | 4 GB |
| Disk (full experiment) | 4 GB | 8 GB |
| CPU | 2 cores | 8+ cores |
| GPU | Not required | Speeds XGBoost ~5× |

### Verify Your Environment

```bash
python --version     # Should be 3.9+
node --version       # Should be 18+
npm --version        # Should be 9+
git --version
```

---

## 2. Local Development — Streamlit Only

This is the simplest setup — it runs the Streamlit dashboard with a demo model trained on synthetic data. No Kaggle account or real dataset required.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection.git
cd Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection
```

### Step 2: Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
venv\Scripts\activate.bat
```

### Step 3: Run the Setup Script

```bash
bash setup.sh
```

This script:
1. Creates `data/`, `results/`, `figures/`, `results/checkpoints/` directories
2. Installs all Python dependencies from `requirements.txt`
3. Generates synthetic data via `scripts/generate_synthetic_data.py`
4. Creates the demo model via `scripts/create_demo_model.py`

### Step 4: Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**

### Optional: Run the Full ML Experiment

If you have the Kaggle dataset (`creditcard.csv`):

```bash
# Place creditcard.csv in the data/ directory
cp /path/to/creditcard.csv data/creditcard.csv

# Run the full experiment (takes 2–3 hours)
python src/run_experiment.py \
    --data-path data/creditcard.csv \
    --output-dir results/ \
    --fig-dir figures/
```

---

## 3. Local Development — Full Stack

The full stack includes the Streamlit dashboard, FastAPI backend, and React frontend.

### Step 1: Complete Streamlit Setup First

Follow all steps from [Section 2](#2-local-development--streamlit-only).

### Step 2: Start the FastAPI Backend

```bash
# In a new terminal
source venv/bin/activate

cd dashboard/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Verify the backend is running:
```bash
curl http://localhost:8000/health
# Expected: {"status": "ok", "model_loaded": true}
```

API documentation (Swagger UI): http://localhost:8000/docs

### Step 3: Start the React Frontend

```bash
# In another new terminal
cd dashboard/frontend
npm install
npm run dev
```

The React frontend will be available at **http://localhost:5173**

### Step 4: Environment Variables for Full Stack

Create a `.env` file in `dashboard/frontend/`:
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

Create a `.env` file in `dashboard/backend/`:
```env
MODEL_PATH=../../results/stacking_model.pkl
METRICS_PATH=../../results/metrics.json
ALLOWED_ORIGINS=http://localhost:5173
```

### Step 5: Verify Full Stack

```bash
# Test the prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": -2.3, "V2": 1.2, "V3": -3.1, "Amount": 99.5, "Hour": 14.0}'

# Run the automated smoke tests
cd scripts
pytest test_backend.py -v
```

---

## 4. Render.com Deployment

Render.com provides free-tier hosting for both the Python backend and Node.js frontend. This section walks through deploying the full application.

### Step 1: Prerequisites

- GitHub account with the repository pushed
- [Render.com](https://render.com) account (free tier sufficient)
- Repository must contain `render.yaml` (included in this project)

### Step 2: Connect Repository to Render

1. Log in to [Render.com](https://render.com)
2. Click **"New +"** → **"Blueprint"**
3. Select your GitHub repository
4. Render will auto-detect `render.yaml` and show the services to be created:
   - `fraud-backend` (Python web service)
   - `fraud-frontend` (Node.js static site)
5. Click **"Apply"**

### Step 3: Review Auto-Detected Build Commands

The `render.yaml` configures the following build pipeline:

```yaml
# Backend build command (auto-generated synthetic data + demo model)
buildCommand: |
  pip install -r requirements.txt
  python scripts/generate_synthetic_data.py
  python scripts/create_demo_model.py

# Frontend build command
buildCommand: cd dashboard/frontend && npm install && npm run build
```

> ⚠️ **Important:** The build command uses synthetic data, not the Kaggle dataset, because `data/creditcard.csv` is gitignored and cannot be deployed directly. The deployed demo uses the synthetic model.

### Step 4: Set Environment Variables

In the Render dashboard for the **backend service**:

1. Go to **Environment** tab
2. Add the following variables:

| Key | Value | Notes |
|---|---|---|
| `ALLOWED_ORIGINS` | `https://your-frontend.onrender.com` | Replace with your frontend URL |
| `MODEL_PATH` | `results/stacking_model.pkl` | Path relative to repo root |
| `METRICS_PATH` | `results/metrics.json` | Path relative to repo root |
| `PORT` | `8000` | Auto-set by Render, but explicit is clearer |
| `PYTHON_VERSION` | `3.11.0` | Pin Python version |

For the **frontend service**:

| Key | Value |
|---|---|
| `VITE_API_URL` | `https://your-backend.onrender.com` |
| `VITE_WS_URL` | `wss://your-backend.onrender.com` |

### Step 5: Health Check Configuration

Render requires a health check endpoint. This project provides `GET /health`:

```json
{"status": "ok", "model_loaded": true}
```

In the Render backend service settings:
- **Health Check Path:** `/health`
- **Health Check Interval:** 30 seconds

### Step 6: Verify Deployment

1. Wait for both services to deploy (typically 5–10 minutes)
2. Check health: `curl https://your-backend.onrender.com/health`
3. Open the frontend URL in your browser
4. Test a prediction via the React frontend

### Step 7: Custom Domain (Optional)

1. In Render, go to your service → **Settings** → **Custom Domains**
2. Add your domain (e.g., `fraud-detector.yourdomain.com`)
3. Update your DNS records as instructed by Render
4. Update `ALLOWED_ORIGINS` in the backend to include your custom domain

### Streamlit-Only Deployment on Render

If you only want to deploy the Streamlit dashboard (simpler):

1. Create a **"Web Service"** (not Blueprint)
2. Connect your repository
3. Set:
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt && python scripts/create_demo_model.py`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.headless true`
4. Add health check: `/_stcore/health`

---

## 5. Environment Variables Reference

### Backend (`dashboard/backend/`)

| Variable | Default | Required | Description |
|---|---|---|---|
| `MODEL_PATH` | `results/stacking_model.pkl` | No | Path to the joblib model bundle |
| `METRICS_PATH` | `results/metrics.json` | No | Path to metrics JSON |
| `ALLOWED_ORIGINS` | `*` | No | CORS allowed origins. **Change to your frontend URL in production** — `"*"` allows any origin, which is fine for demo but should be locked down for production. Example: `https://myapp.onrender.com` |
| `PORT` | `8000` | No | Server port (auto-set by Render/Railway) |
| `LOG_LEVEL` | `info` | No | Logging level: debug, info, warning, error |

### Frontend (`dashboard/frontend/`)

| Variable | Default | Required | Description |
|---|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | Yes (prod) | FastAPI backend base URL |
| `VITE_WS_URL` | `ws://localhost:8000` | Yes (prod) | WebSocket base URL (use `wss://` in production) |

### Streamlit (`app.py`)

| Variable | Default | Required | Description |
|---|---|---|---|
| `PORT` | `8501` | No | Auto-set by Render when using `$PORT` in start command |

> **Security note on ALLOWED_ORIGINS:** In development, `ALLOWED_ORIGINS = "*"` is convenient. Before deploying to production, update it to your exact frontend URL to prevent cross-origin requests from unauthorised domains. Example: `ALLOWED_ORIGINS=https://fraud-frontend.onrender.com`

---

## 6. Troubleshooting

### Issue 1: `ModuleNotFoundError` on startup

**Symptom:**
```
ModuleNotFoundError: No module named 'xgboost'
```

**Solution:**
```bash
# Activate your virtual environment first
source venv/bin/activate

# Reinstall all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import xgboost; print(xgboost.__version__)"
```

If using Render and the build fails, check that `requirements.txt` includes all packages and that `PYTHON_VERSION` is set.

---

### Issue 2: Model not loaded / dashboard shows warning

**Symptom:**  
Streamlit shows "⚠️ No model found" in the sidebar.

**Solution:**
```bash
# Ensure you've created the demo model
python scripts/create_demo_model.py

# Verify the file exists
ls -la results/stacking_model.pkl
```

If the script itself fails, check:
```bash
# Check for sufficient fraud samples
python -c "
import pandas as pd
df = pd.read_csv('data/synthetic_creditcard.csv')
print(df['Class'].value_counts())
"

# If fraud count < 6, regenerate with higher fraud rate
python scripts/generate_synthetic_data.py --fraud-rate 0.05
python scripts/create_demo_model.py --fraud-rate 0.05
```

---

### Issue 3: FastAPI backend crashes on startup

**Symptom:**
```
RuntimeError: Model file not found at results/stacking_model.pkl
```
or
```
Application startup failed. Exiting.
```

**Solution:**  
The backend is designed to degrade gracefully — it should log a warning and return `503 Service Unavailable` on model-dependent routes, not crash. If it's still crashing:

```bash
# Create the model first
python scripts/create_demo_model.py

# Check the model path configuration
cat dashboard/backend/main.py | grep MODEL_PATH

# Set the env variable explicitly
export MODEL_PATH="$(pwd)/results/stacking_model.pkl"
uvicorn dashboard.backend.main:app --reload
```

---

### Issue 4: Render health check fails (503)

**Symptom:**  
Render shows "Service unavailable" or the deploy fails health check.

**Solution:**

1. Verify the `/health` endpoint exists and returns 200:
   ```bash
   curl https://your-backend.onrender.com/health
   ```

2. Check Render build logs for the `create_demo_model.py` step — it must succeed for the model to be present at startup.

3. Ensure the health check path in Render settings matches exactly: `/health` (not `/` or `/healthz`)

4. Check Render environment variables — missing `MODEL_PATH` won't crash but the `model_loaded` field in `/health` will be `false`.

---

### Issue 5: CORS errors in React frontend

**Symptom:**
```
Access to fetch at 'http://localhost:8000/predict' from origin 'http://localhost:5173'
has been blocked by CORS policy
```

**Solution:**  
The backend's `ALLOWED_ORIGINS` must include the frontend's origin.

For local development:
```bash
export ALLOWED_ORIGINS="http://localhost:5173"
uvicorn dashboard.backend.main:app --reload
```

For production (in Render environment variables):
```
ALLOWED_ORIGINS=https://fraud-frontend-abc123.onrender.com
```

Multiple origins can be comma-separated:
```
ALLOWED_ORIGINS=https://myapp.com,https://www.myapp.com
```

---

### Issue 6: XGBoost memory error / OOM during experiment

**Symptom:**
```
Killed  (process killed by OS due to out of memory)
```
or
```
MemoryError: Unable to allocate ... array
```

**Solution:**
```bash
# Reduce n_estimators and n_iter
python src/run_experiment.py \
    --data-path data/creditcard.csv \
    --n-iter-rf 10 \
    --n-iter-xgb 20 \
    --n-folds 3 \
    --n-stability-runs 2

# Or run in low-memory mode (disable parallelism)
python src/run_experiment.py \
    --data-path data/creditcard.csv \
    --n-jobs 1 \
    --n-iter-rf 10 \
    --n-iter-xgb 20
```

---

### Issue 7: `setup.sh` encoding error (mojibake)

**Symptom:**  
`bash setup.sh` fails with strange characters or encoding errors.

**Solution:**  
The included `setup.sh` uses ASCII-only characters to avoid encoding issues. If you encounter problems:
```bash
# Check file encoding
file setup.sh

# Fix line endings if needed
sed -i 's/\r//' setup.sh

# Run with explicit bash
/bin/bash setup.sh
```

---

## 7. Performance Tuning

### Speeding Up the Full Experiment

The full experiment (with the Kaggle dataset) takes ~2.5–3 hours on a 4-core CPU. Here are options to speed it up:

**Reduce search iterations:**
```bash
python src/run_experiment.py \
    --n-iter-rf 15 \
    --n-iter-xgb 25 \
    --n-folds 3 \
    --skip-learning-curves
# Estimated time: ~45–60 minutes
```

**Skip expensive optional steps:**
```bash
python src/run_experiment.py \
    --skip-ablation \
    --skip-learning-curves \
    --n-stability-runs 2
# Estimated time: ~1.5 hours
```

**Use checkpoints:**  
If interrupted, re-running the experiment will automatically resume from saved base-learner checkpoints in `results/checkpoints/`. No work is lost.

**GPU acceleration:**  
XGBoost supports GPU training. If you have an NVIDIA GPU:
```python
# In CONFIG dict, add:
"xgb_device": "cuda"

# And pass device='cuda' to XGBClassifier constructors
```

### Optimising the Streamlit Dashboard

**Cache the model:**  
The dashboard uses `@st.cache_resource` for the model and `@st.cache_data` for metrics — these are already optimised.

**Reduce Streamlit's overhead:**
```bash
streamlit run app.py \
    --server.maxUploadSize=10 \
    --server.enableCORS=false \
    --browser.gatherUsageStats=false
```

**Production Streamlit deployment:**
```bash
# Disable development mode for faster startup
streamlit run app.py \
    --server.headless=true \
    --server.port=$PORT \
    --server.address=0.0.0.0
```

### Optimising FastAPI

**Enable multiple workers:**
```bash
uvicorn dashboard.backend.main:app \
    --workers 4 \
    --host 0.0.0.0 \
    --port 8000
```

**Profile slow endpoints:**
```bash
pip install pyinstrument
python -m pyinstrument -m uvicorn dashboard.backend.main:app
```

**Model pre-warming:**  
On startup, the backend makes one dummy prediction to warm up the model cache, reducing first-request latency from ~300ms to ~2ms.

### Render.com Performance Tips

1. **Upgrade from free tier:** The free tier spins down after 15 minutes of inactivity, causing a 30–60s cold start. Upgrading to the Starter plan ($7/month) keeps the service always-on.

2. **Use a faster region:** Choose the Render region closest to your users (EU, US East, US West, Singapore).

3. **Build caching:** Render caches pip installations between deploys. Pin versions in `requirements.txt` to avoid unnecessary re-downloads.

4. **Set PYTHONDONTWRITEBYTECODE=1:** Prevents .pyc files from being written to disk, slightly reducing build time.

---

## Quick Reference

```bash
# Local Streamlit setup (fastest path to running)
git clone <repo> && cd <repo>
bash setup.sh
streamlit run app.py

# Full stack local dev
uvicorn dashboard.backend.main:app --reload --port 8000 &
cd dashboard/frontend && npm install && npm run dev &
streamlit run app.py &

# Full ML experiment
python src/run_experiment.py --data-path data/creditcard.csv --output-dir results/ --fig-dir figures/

# Run tests
pytest scripts/test_backend.py -v

# Create fresh demo model
python scripts/create_demo_model.py --n-samples 15000 --fraud-rate 0.05
```

---

*For methodology details, see [METHODOLOGY.md](METHODOLOGY.md). For results, see [RESULTS.md](RESULTS.md).*
