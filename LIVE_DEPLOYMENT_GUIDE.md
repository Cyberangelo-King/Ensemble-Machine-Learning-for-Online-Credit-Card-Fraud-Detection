# 🌐 Live Deployment Guide

This guide explains how to deploy the **FraudGuard AI** Command Center to the web so you can share a live link with Dr. Ajulo.

## Option 1: One-Click Deployment (Recommended for Render.com)

Render is great because it can host both the Python backend (with WebSockets) and the React frontend.

1.  **Push to GitHub:**
    Make sure all your changes (including the new `dashboard/` folder and `render.yaml`) are pushed to a GitHub repository.
2.  **Connect to Render:**
    - Go to [dashboard.render.com](https://dashboard.render.com).
    - Click **"New +"** -> **"Blueprint"**.
    - Connect your GitHub repository.
3.  **Deploy:**
    Render will read the `render.yaml` file and automatically:
    - Set up the Python environment.
    - Train the model (so it's ready for the dashboard).
    - Start the FastAPI backend.
    - Build and host the React frontend.

## Option 2: Split Deployment (Fastest Frontend Performance)

If you prefer using **Netlify** or **Vercel** for the frontend:

### 1. Backend (FastAPI on Render/Railway)
- Deploy the root folder to Render as a **Web Service**.
- Build Command: `pip install -r requirements.txt && python scripts/generate_synthetic_data.py && python src/run_experiment.py`
- Start Command: `uvicorn dashboard.backend.main:app --host 0.0.0.0 --port $PORT`
- **Copy the URL** (e.g., `https://fraud-api.onrender.com`).

### 2. Frontend (Netlify)
- Connect your GitHub repo to Netlify.
- Set the **Base Directory** to `dashboard/frontend`.
- Set the **Build Command** to `npm run build`.
- Set the **Publish Directory** to `dashboard/frontend/dist`.
- **CRITICAL:** Add Environment Variables in the Netlify UI:
    - `VITE_API_URL`: `https://your-backend-url.onrender.com`
    - `VITE_WS_URL`: `wss://your-backend-url.onrender.com/ws/stream` (Note the `wss://` for secure WebSockets).

## Important Presentation Note
Free-tier hosting (like Render's free tier) often "sleeps" after inactivity. If you're presenting to Dr. Ajulo, **open the link 2 minutes before the presentation** to ensure the backend has "woken up."
