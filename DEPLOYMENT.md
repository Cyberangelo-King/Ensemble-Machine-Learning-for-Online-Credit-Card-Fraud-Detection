# Demo Deployment and Operations Guide

> **Status:** research/demo deployment only. This repository is not a certified fraud-control service and must not receive real cardholder data, payment credentials or production transaction traffic.

## 1. Understand the two model paths

The repository has two distinct workflows:

| Workflow | Purpose | Data |
|---|---|---|
| Synthetic demo | Exercise the Streamlit dashboard, FastAPI API and React replay UI | Generated synthetic rows |
| Research benchmark | Train and evaluate the ensemble on the public ULB/Kaggle dataset | `creditcard.csv` |

The synthetic model is not a substitute for the benchmark model. It must not be used to make claims about real-world fraud-detection performance. The old headline figures in `RESULTS.md` are explicitly archived and are not verified results for the revised experiment.

## 2. Local setup

### Python environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Generate a synthetic demonstration model

```bash
python scripts/generate_synthetic_data.py
python scripts/create_demo_model.py
streamlit run app.py
```

The Streamlit app is available at `http://localhost:8501`. The synthetic dataset/model are demonstration artifacts only.

### Run the FastAPI demo

From the repository root, with the synthetic model present:

```bash
export APP_ENV=development
export MODEL_SHA256="$(sha256sum results/stacking_model.pkl | cut -d' ' -f1)"
export ALLOWED_ORIGINS="http://localhost:5173"
uvicorn dashboard.backend.main:app --reload --host 0.0.0.0 --port 8000
```

PowerShell equivalent for the digest:

```powershell
$env:APP_ENV = "development"
$env:MODEL_SHA256 = (Get-FileHash results/stacking_model.pkl -Algorithm SHA256).Hash.ToLower()
$env:ALLOWED_ORIGINS = "http://localhost:5173"
uvicorn dashboard.backend.main:app --reload --host 0.0.0.0 --port 8000
```

Check `GET /health` at `http://localhost:8000/health`. The endpoint returns 503 until the model has loaded and passed its SHA-256 and feature-schema checks. API documentation is at `/docs`.

### Run the React demo

```bash
cd dashboard/frontend
npm ci
VITE_API_URL=http://localhost:8000 VITE_WS_URL=ws://localhost:8000 npm run dev
```

Open `http://localhost:5173`. The dashboard replays locally available sample rows through the research model; it is not a live payment stream. SHAP explanations are generated for selected rows when the backend explainer is available.

## 3. Run the actual research benchmark

Obtain the public ULB/Kaggle `creditcard.csv` dataset yourself and place it at `data/creditcard.csv`. Do not commit the data file.

```bash
python src/run_experiment.py \
  --data-path data/creditcard.csv \
  --output-dir results/ \
  --fig-dir figures/ \
  --split-strategy temporal
```

The current default protocol uses chronological training/validation/test partitions, forward-chaining time-series CV for temporal experiments, fold-local scaling and SMOTE, OOF stacking, validation-only threshold selection and a one-time final test evaluation.

The run can be computationally expensive. No benchmark metric is considered current until the experiment completes and its generated `results/metrics.json`, model bundle, plots, provenance and test counts have been inspected. Record the commit SHA and preserve the run artifacts. See `RESEARCH_AUDIT.md` and `METHODOLOGY.md`.

## 4. Render Blueprint deployment (public demo only)

The checked-in `render.yaml` defines:
- a Python 3.11/FastAPI web service;
- a static Vite frontend served by Render's static-site infrastructure;
- HTTPS/WSS URLs from the frontend to the API;
- a restricted CORS origin;
- `APP_ENV=demo`, explicitly distinguishing the public demo from production.

### Before creating the Blueprint

1. Train or create the exact demo/research model you intend to host.
2. Upload `results/stacking_model.pkl` to a trusted HTTPS artifact store you control.
3. Compute its SHA-256 digest:
   ```bash
   sha256sum results/stacking_model.pkl
   ```
4. In Render, create a Blueprint from this repository and supply:
   - `MODEL_URL`: the artifact's HTTPS download URL;
   - `MODEL_SHA256`: the exact 64-character digest.

The build script streams the artifact to a temporary file, enforces HTTPS, verifies the SHA-256 digest and only then atomically replaces the model file. A missing URL, failed download or digest mismatch aborts the backend build.

### Service configuration

The frontend uses:
- `VITE_API_URL=https://fraud-api.onrender.com`
- `VITE_WS_URL=wss://fraud-api.onrender.com`

The React app appends `/ws/stream` to the WebSocket base URL. Because Vite variables are compiled into public browser JavaScript, never put an API key, storage credential or other secret in a `VITE_*` variable.

The backend CORS allowlist in the Blueprint is `https://fraud-dashboard.onrender.com`. Update `ALLOWED_ORIGINS` if the assigned Render hostname or custom domain differs.

Render's `/health` check requires a valid model file and matching `MODEL_SHA256`. A service without the required model artifact will remain unready and return HTTP 503 by design.

## 5. Authentication boundary — important

The current React demo does not implement end-user authentication and browsers cannot attach an arbitrary `X-API-Key` header to the native WebSocket constructor. Therefore:

- The Blueprint sets `APP_ENV=demo` and is for non-sensitive public demonstration only.
- Do not expose real transaction/cardholder data through this deployment.
- Do not set `FRAUD_API_KEY` on the public SPA deployment unless an authenticated same-origin proxy or equivalent cookie-based/user-authentication layer is implemented and tested.
- Setting `APP_ENV=production` without `FRAUD_API_KEY` makes protected API and WebSocket paths fail closed.
- The current rate limiter is process-local and is not a distributed abuse-control system.

A genuine production design requires user/service authentication, a trusted API gateway or backend-for-frontend, durable audit trails, monitoring, privacy controls, independent security review and an operational fraud-review policy. None of those are established by deploying this demo.

## 6. Environment variables

| Variable | Component | Purpose |
|---|---|---|
| `MODEL_URL` | Backend build | Trusted HTTPS model artifact URL |
| `MODEL_SHA256` | Backend | Required SHA-256 digest checked before deserialization |
| `MODEL_MAX_BYTES` | Model fetcher | Maximum downloaded model size (default 2 GiB) |
| `APP_ENV` | Backend | `development`, `demo`, or `production`; production requires auth configuration |
| `FRAUD_API_KEY` | Backend | Optional API key; required by production-mode guarded endpoints |
| `ALLOWED_ORIGINS` | Backend | Comma-separated exact browser origins |
| `RATE_LIMIT_PER_MINUTE` | Backend | Best-effort per-process request limit |
| `VITE_API_URL` | Frontend build | HTTPS API base URL |
| `VITE_WS_URL` | Frontend build | WSS WebSocket base URL; app appends `/ws/stream` |

## 7. Operational checks

After a demo deployment:
1. Confirm the backend reports `status=ok`, `model_loaded=true` and the expected model version at `/health`.
2. Open the frontend and start the simulated replay.
3. Confirm rows show an index, amount, probability, thresholded label, ground-truth label (when available) and measured inference latency.
4. Select a row and confirm that its SHAP explanation matches the backend response.
5. Inspect Render logs for model digest, schema, startup and replay errors.

A successful deployment proves that the demo stack starts. It does **not** establish benchmark performance, production readiness, online transaction ingestion, regulatory compliance or safe use with real financial data.
