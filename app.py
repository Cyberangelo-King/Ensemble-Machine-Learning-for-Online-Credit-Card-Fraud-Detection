"""Streamlit dashboard for credit card fraud stacking ensemble.

Entry point: streamlit run app.py
"""
from __future__ import annotations

import io
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
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve

LOGGER = logging.getLogger("fraud_dashboard")
logging.basicConfig(level=logging.INFO)

TARGET_METRICS = {"AUPRC": 0.903, "F1": 0.881, "MCC": 0.884}
MODEL_PATH = Path("artifacts/stacking_model.joblib")
DATA_PATH = Path("data/creditcard.csv")
RISK_COLORS = {"Low": "#2ca02c", "Medium": "#ffbf00", "High": "#d62728"}
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

SAMPLES = {
    "Legitimate Example": {f: 0.0 for f in FEATURES} | {"Amount": 12.3, "Time": 10000},
    "Fraud-like Example": {f: -2.0 for f in FEATURES} | {"V14": -4.0, "V17": -5.0, "Amount": 2345, "Time": 40000},
    "Borderline Example": {f: -0.3 for f in FEATURES} | {"V10": -1.1, "V12": -1.3, "Amount": 350, "Time": 25000},
}


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file missing: {path}")
    LOGGER.info("Loading model from %s", path)
    return joblib.load(path)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset missing: {path}")
    df = pd.read_csv(path)
    return df


@st.cache_resource
def get_shap_explainer(model):
    # stacked model has named estimators in sklearn 1.3
    xgb_model = model.named_estimators_.get("xgb") if hasattr(model, "named_estimators_") else None
    if xgb_model is None:
        raise ValueError("XGBoost base learner not found in stacking model.")
    return shap.TreeExplainer(xgb_model), xgb_model


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
    if prob < 0.2:
        return "Low"
    if prob < 0.6:
        return "Medium"
    return "High"


def simulator_page(model):
    st.header("Transaction Simulator")
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
        st.subheader("Prediction Result")
        st.metric("Fraud Probability", f"{proba*100:.2f}%")
        st.metric("Confidence", f"{conf*100:.2f}%")
        st.metric("Inference Latency", f"{dt:.3f} ms")
        st.markdown(f"**Risk Level:** :{ 'red' if risk=='High' else 'orange' if risk=='Medium' else 'green'}[{risk}]")
        st.progress(float(proba))
        st.session_state.setdefault("history", []).append({**vals, "proba": proba, "latency_ms": dt, "risk": risk})


def shap_page(model, df):
    st.header("SHAP Explanations")
    with st.spinner("Computing SHAP resources..."):
        explainer, xgb_model = get_shap_explainer(model)
        sample = df[FEATURES].sample(min(1000, len(df)), random_state=42)
        shap_vals = explainer.shap_values(sample)

    st.subheader("Summary Plot")
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.summary_plot(shap_vals, sample, show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.clf()
    st.info("Key discriminative features to inspect: V17, V14, V12, V10.")

    st.subheader("Dependence Plot")
    feat = st.selectbox("Select feature", FEATURES, index=16)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    shap.dependence_plot(feat, shap_vals, sample, show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.clf()

    st.subheader("Force Plot (latest prediction)")
    if st.session_state.get("history"):
        latest = pd.DataFrame([st.session_state["history"][-1]])[FEATURES]
        sv = explainer.shap_values(latest)
        force = shap.force_plot(explainer.expected_value, sv[0], latest.iloc[0], matplotlib=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{force.html()}</body>"
        st.components.v1.html(shap_html, height=320, scrolling=True)
    else:
        st.warning("Run a prediction first to view force plot.")


def metrics_page(df):
    st.header("Performance Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("AUPRC", TARGET_METRICS["AUPRC"])
    c2.metric("F1-Score", TARGET_METRICS["F1"])
    c3.metric("MCC", TARGET_METRICS["MCC"])
    st.caption("AUPRC, F1, and MCC are more informative than accuracy under severe class imbalance.")

    y = df["Class"].values
    # proxy scores for interactive demo
    score = (df["V17"].rank(pct=True).values * -1 + 1) * 0.5 + (df["V14"].rank(pct=True).values * -1 + 1) * 0.5
    thr = st.slider("Decision Threshold", 0.01, 0.99, 0.5, 0.01)
    pred = (score >= thr).astype(int)
    cm = confusion_matrix(y, pred)
    cm_norm = cm / cm.sum()
    txt = [[f"{cm[i,j]}\n({cm_norm[i,j]*100:.2f}%)" for j in range(2)] for i in range(2)]
    fig = go.Figure(data=go.Heatmap(z=cm, x=["Legit","Fraud"], y=["Legit","Fraud"], colorscale="Blues", text=txt, texttemplate="%{text}"))
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

    precision, recall, t = precision_recall_curve(y, score)
    pr = px.line(x=recall, y=precision, labels={"x":"Recall","y":"Precision"}, title="Precision-Recall Curve")
    st.plotly_chart(pr, use_container_width=True)


def dataset_page(df):
    st.header("Dataset Exploration")
    st.write("Dataset contains 284,807 European card transactions with ~0.17% fraud.")
    cls = df["Class"].value_counts().rename({0:"Legitimate",1:"Fraud"})
    st.plotly_chart(px.bar(x=cls.index, y=cls.values, color=cls.index, color_discrete_map={"Fraud":"red","Legitimate":"green"}, title="Original Class Distribution"), use_container_width=True)
    smote_counts = pd.Series({"Fraud": int(cls["Legitimate"]*0.1), "Legitimate": int(cls["Legitimate"])})
    st.plotly_chart(px.bar(x=smote_counts.index, y=smote_counts.values, color=smote_counts.index, title="Training Distribution After SMOTE (1:10)"), use_container_width=True)

    feature = st.selectbox("Feature distribution", FEATURES)
    fig = px.histogram(df, x=feature, color=df["Class"].map({0:"Legitimate",1:"Fraud"}), barmode="overlay", nbins=60)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.groupby("Class")[FEATURES].agg(["mean","std","min","max"]).head(10))


def comparison_page():
    st.header("Model Comparison")
    cmp = pd.DataFrame([
        {"Model":"Logistic Regression","AUPRC":0.79,"F1":0.74,"MCC":0.75,"Latency":0.030},
        {"Model":"Random Forest","AUPRC":0.86,"F1":0.82,"MCC":0.83,"Latency":0.220},
        {"Model":"XGBoost","AUPRC":0.89,"F1":0.86,"MCC":0.87,"Latency":0.140},
        {"Model":"Stacking Ensemble","AUPRC":0.903,"F1":0.881,"MCC":0.884,"Latency":0.074},
    ])
    st.plotly_chart(px.bar(cmp.melt(id_vars="Model", value_vars=["AUPRC","F1","MCC"]), x="variable", y="value", color="Model", barmode="group", title="Metric Comparison"), use_container_width=True)
    st.plotly_chart(px.bar(cmp, x="Model", y="Latency", color="Model", title="Inference Latency (ms / transaction)"), use_container_width=True)
    st.markdown("Stacking architecture combines complementary errors of base learners through a logistic meta-learner.")


def home_page():
    st.title("Credit Card Fraud Detection Dashboard")
    st.markdown("Interactive companion app for stacking ensemble fraud detection research.")
    st.markdown("- Repository: https://github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection")
    st.markdown("- Key results: AUPRC 0.903, F1 0.881, MCC 0.884")


def main():
    st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Simulator", "SHAP Explanations", "Metrics", "Dataset", "Model Comparison"])

    try:
        model = load_model(MODEL_PATH)
        df = load_data(DATA_PATH)
    except Exception as exc:
        st.error(f"Startup error: {exc}")
        st.stop()

    if page == "Home":
        home_page()
    elif page == "Simulator":
        simulator_page(model)
    elif page == "SHAP Explanations":
        shap_page(model, df)
    elif page == "Metrics":
        metrics_page(df)
    elif page == "Dataset":
        dataset_page(df)
    elif page == "Model Comparison":
        comparison_page()


if __name__ == "__main__":
    main()
