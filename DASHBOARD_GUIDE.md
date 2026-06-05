# FraudGuard AI: Real-time Stacking Ensemble Dashboard

This project replaces a basic Streamlit app with a high-fidelity, real-time "Command Center" dashboard designed for academic defenses. It features a React frontend and a FastAPI backend.

## Key Features
- **Real-time Streaming:** Transactions are streamed via WebSockets and processed live.
- **Forensic Investigation:** Click any transaction to see a deep-dive SHAP (SHapley Additive exPlanations) breakdown.
- **Academic Metrics:** Live updates for Precision, Recall, and F1-Score.
- **Professional UI:** Dark/Light mode, high-contrast typography, and a "Command Center" aesthetic.

## How to Run

### 1. Prerequisites
Ensure you have Python 3.8+ and Node.js 18+ installed.

### 2. Setup
Install Python dependencies:
```bash
pip install -r requirements.txt
pip install fastapi uvicorn websockets joblib pandas shap xgboost scikit-learn
```

Install Frontend dependencies:
```bash
cd dashboard/frontend
npm install
```

### 3. Training the Model (Optional)
If `artifacts/stacking_model.joblib` is missing, you can generate synthetic data and train the model:
```bash
python scripts/generate_synthetic_data.py
python src/run_experiment.py
```

### 4. Start the Backend
```bash
python dashboard/backend/main.py
```
The API will be available at `http://localhost:8000`.

### 5. Start the Frontend
In a new terminal:
```bash
cd dashboard/frontend
npm run dev
```
The dashboard will be available at `http://localhost:3000`.

## Presentation Tips for Dr. E. Ajulo
1. **The "Live" Element:** Start the simulation and watch the metrics stabilize. This demonstrates the system's ability to handle high-velocity data.
2. **Explainability:** When a fraud (red) transaction appears, click it immediately. Use the SHAP bar chart to explain *why* the model made that decision (e.g., "Feature V17 was the primary driver for this high-risk classification").
3. **The Ensemble Story:** Mention that the predictions are the result of a Stacking Ensemble (LR + RF + XGB), providing better robustness than any single model.
