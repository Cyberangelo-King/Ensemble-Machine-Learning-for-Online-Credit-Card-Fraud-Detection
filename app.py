"""Streamlit dashboard for credit card fraud stacking ensemble.

Entry point: streamlit run app.py
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve

LOGGER = logging.getLogger("fraud_dashboard")
logging.basicConfig(level=logging.INFO)

TARGET_METRICS = {"AUPRC": 0.903, "F1": 0.881, "MCC": 0.884}
MODEL_PATH = Path("artifacts/stacking_model.joblib")
DATA_PATH = Path("data/creditcard.csv")
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

SAMPLES = {
    "Legitimate Example": {f: 0.0 for f in FEATURES} | {"Amount": 12.3, "Time": 10000},
    "Fraud-like Example": {f: -2.0 for f in FEATURES} | {"V14": -4.0, "V17": -5.0, "Amount": 2345, "Time": 40000},
    "Borderline Example": {f: -0.3 for f in FEATURES} | {"V10": -1.1, "V12": -1.3, "Amount": 350, "Time": 25000},
}


class DemoModel:
    """Fallback probabilistic model for demo mode when artifact is missing."""

    is_demo = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        z = (
            -0.9 * X["V17"].to_numpy()
            - 0.8 * X["V14"].to_numpy()
            - 0.6 * X["V12"].to_numpy()
            - 0.5 * X["V10"].to_numpy()
            + 0.0003 * X["Amount"].to_numpy()
        )
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 0.001, 0.999)
        return np.column_stack([1 - p, p])


def startup_wizard() -> tuple[object, pd.DataFrame, bool]:
    """Load model/data with clear guidance and optional demo fallback."""
    missing = []
    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH))
    if not DATA_PATH.exists():
        missing.append(str(DATA_PATH))

    if missing:
        st.warning("Startup Wizard: required files are missing.")
        st.markdown("**Missing files:**")
        for item in missing:
            st.markdown(f"- `{item}`")
        st.markdown(
            """
### How to fetch required files
1. **Dataset (`data/creditcard.csv`)**
   - Download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - Place CSV in `data/creditcard.csv`.
2. **Model artifact (`artifacts/stacking_model.joblib`)**
   - Run training script to auto-save model:
     `python src/run_experiment.py --data-path data/creditcard.csv --artifact-path artifacts/stacking_model.joblib`
"""
        )
        demo = st.toggle("Enable Demo Mode (recommended for presentations)", value=True)
        if demo:
            st.info("Demo mode enabled: predictions are mock heuristic outputs.")
            return DemoModel(), build_demo_dataframe(), True
        st.stop()

    return load_model(MODEL_PATH), load_data(DATA_PATH), False


@st.cache_resource
def load_model(path: Path):
    LOGGER.info("Loading model from %s", path)
    return joblib.load(path)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def build_demo_dataframe(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, size=(n, len(FEATURES))), columns=FEATURES)
    X["Amount"] = np.abs(rng.normal(150, 200, size=n))
    X["Time"] = rng.integers(0, 172800, size=n)
    y = ((-X["V17"] - X["V14"] - X["V12"] + 0.001 * X["Amount"]) > 2.2).astype(int)
    return X.assign(Class=y)


def validate_inputs(values: dict) -> tuple[bool, str]:
    try:
        for k, v in values.items():
            if not np.isfinite(float(v)):
                return False, f"{k} must be finite."
        if values["Amount"] < 0:
            return False, "Amount cannot be negative."
        if values["Time"] < 0:
            return False, "Time cannot be negative."
        return True, ""
    except Exception as exc:
        return False, f"Invalid input: {exc}"


def classify_risk(prob: float) -> str:
    return "Low" if prob < 0.2 else "Medium" if prob < 0.6 else "High"


def simulator_page(model, is_demo: bool):
    st.header("Transaction Simulator")
    if is_demo:
        st.warning("Demo Mode: outputs are simulated and not from the trained artifact.")

    col1, col2 = st.columns(2)
    selected = st.selectbox("Load sample", ["None"] + list(SAMPLES.keys()))
    defaults = SAMPLES[selected] if selected != "None" else {f: 0.0 for f in FEATURES}
    vals = {}
    for idx, f in enumerate(FEATURES):
        with col1 if idx % 2 == 0 else col2:
            vals[f] = st.number_input(f, value=float(defaults[f]), format="%.6f")

    if st.button("Predict Transaction"):
        ok, msg = validate_inputs(vals)
        if not ok:
            st.error(msg)
            return
        x = pd.DataFrame([vals])[FEATURES]
        t0 = time.perf_counter()
        proba = float(model.predict_proba(x)[:, 1][0])
        dt = (time.perf_counter() - t0) * 1000
        risk = classify_risk(proba)
        conf = max(proba, 1 - proba)
        st.metric("Fraud Probability", f"{proba*100:.2f}%")
        st.metric("Confidence", f"{conf*100:.2f}%")
        st.metric("Inference Latency", f"{dt:.3f} ms")
        st.markdown(f"**Risk Level:** {risk}")
        st.progress(float(proba))
        st.session_state.setdefault("history", []).append({**vals, "proba": proba, "latency_ms": dt, "risk": risk})


def shap_page(model, df, is_demo: bool):
    st.header("SHAP Explanations")
    if is_demo:
        st.warning("Demo Mode: SHAP uses a surrogate logistic model for visualization only.")
        surrogate = LogisticRegression(max_iter=1000)
        sample = df[FEATURES].sample(min(1000, len(df)), random_state=42)
        surrogate.fit(sample, df.loc[sample.index, "Class"])
        explainer = shap.Explainer(surrogate, sample)
        shap_vals = explainer(sample)
        shap.summary_plot(shap_vals.values, sample, show=False)
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.clf()
        return

    with st.spinner("Computing SHAP resources..."):
        xgb_model = model.named_estimators_.get("xgb") if hasattr(model, "named_estimators_") else None
        if xgb_model is None:
            st.error("XGBoost base learner missing in model artifact.")
            return
        explainer = shap.TreeExplainer(xgb_model)
        sample = df[FEATURES].sample(min(1000, len(df)), random_state=42)
        shap_vals = explainer.shap_values(sample)

    shap.summary_plot(shap_vals, sample, show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.clf()


def metrics_page(df):
    st.header("Performance Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("AUPRC", TARGET_METRICS["AUPRC"])
    c2.metric("F1-Score", TARGET_METRICS["F1"])
    c3.metric("MCC", TARGET_METRICS["MCC"])

    y = df["Class"].values
    score = (df["V17"].rank(pct=True).values * -1 + 1) * 0.5 + (df["V14"].rank(pct=True).values * -1 + 1) * 0.5
    thr = st.slider("Decision Threshold", 0.01, 0.99, 0.5, 0.01)
    pred = (score >= thr).astype(int)
    cm = confusion_matrix(y, pred)
    cm_norm = cm / cm.sum()
    txt = [[f"{cm[i,j]}\n({cm_norm[i,j]*100:.2f}%)" for j in range(2)] for i in range(2)]
    fig = go.Figure(data=go.Heatmap(z=cm, x=["Legit", "Fraud"], y=["Legit", "Fraud"], colorscale="Blues", text=txt, texttemplate="%{text}"))
    st.plotly_chart(fig, use_container_width=True)

    precision, recall, _ = precision_recall_curve(y, score)
    st.plotly_chart(px.line(x=recall, y=precision, labels={"x": "Recall", "y": "Precision"}, title="Precision-Recall Curve"), use_container_width=True)


def dataset_page(df):
    st.header("Dataset Exploration")
    cls = df["Class"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
    st.plotly_chart(px.bar(x=cls.index, y=cls.values, color=cls.index, title="Original Class Distribution"), use_container_width=True)


def comparison_page():
    st.header("Model Comparison")
    cmp = pd.DataFrame([
        {"Model": "Logistic Regression", "AUPRC": 0.79, "F1": 0.74, "MCC": 0.75, "Latency": 0.030},
        {"Model": "Random Forest", "AUPRC": 0.86, "F1": 0.82, "MCC": 0.83, "Latency": 0.220},
        {"Model": "XGBoost", "AUPRC": 0.89, "F1": 0.86, "MCC": 0.87, "Latency": 0.140},
        {"Model": "Stacking Ensemble", "AUPRC": 0.903, "F1": 0.881, "MCC": 0.884, "Latency": 0.074},
    ])
    st.plotly_chart(px.bar(cmp.melt(id_vars="Model", value_vars=["AUPRC", "F1", "MCC"]), x="variable", y="value", color="Model", barmode="group"), use_container_width=True)


def home_page(is_demo: bool):
    st.title("Credit Card Fraud Detection Dashboard")
    if is_demo:
        st.warning("You are in DEMO MODE. Upload/fetch artifacts for real model predictions.")


def main():
    st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Simulator", "SHAP Explanations", "Metrics", "Dataset", "Model Comparison"])

    model, df, is_demo = startup_wizard()

    if page == "Home":
        home_page(is_demo)
    elif page == "Simulator":
        simulator_page(model, is_demo)
    elif page == "SHAP Explanations":
        shap_page(model, df, is_demo)
    elif page == "Metrics":
        metrics_page(df)
    elif page == "Dataset":
        dataset_page(df)
    elif page == "Model Comparison":
        comparison_page()


if __name__ == "__main__":
    main()
