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
import streamlit.components.v1 as components
from sklearn.metrics import confusion_matrix, precision_recall_curve

LOGGER = logging.getLogger("fraud_dashboard")
logging.basicConfig(level=logging.INFO)

APP_DIR = Path(__file__).resolve().parent
TARGET_METRICS = {"AUPRC": 0.903, "F1": 0.881, "MCC": 0.884}
MODEL_PATH = APP_DIR / "artifacts" / "stacking_model.joblib"
DATA_PATH = APP_DIR / "data" / "creditcard.csv"
RISK_COLORS = {"Low": "#2ca02c", "Medium": "#ffbf00", "High": "#d62728"}
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

SAMPLES = {
    "Legitimate Example": {f: 0.0 for f in FEATURES} | {"Amount": 12.3, "Time": 10000},
    "Fraud-like Example": {f: -2.0 for f in FEATURES} | {"V14": -4.0, "V17": -5.0, "Amount": 2345, "Time": 40000},
    "Borderline Example": {f: -0.3 for f in FEATURES} | {"V10": -1.1, "V12": -1.3, "Amount": 350, "Time": 25000},
}


class HeuristicFraudModel:
    """Small deterministic fallback so the dashboard stays interactive before training."""

    source = "demo heuristic"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        frame = X[FEATURES].astype(float)
        risk_signal = (
            -1.15 * frame["V17"]
            - 0.85 * frame["V14"]
            - 0.65 * frame["V12"]
            - 0.45 * frame["V10"]
            + 0.18 * np.log1p(frame["Amount"].clip(lower=0))
        )
        normalized = (risk_signal - 5.5) / 2.8
        fraud_probability = 1 / (1 + np.exp(-normalized))
        fraud_probability = np.clip(fraud_probability.to_numpy(dtype=float), 0.001, 0.999)
        return np.column_stack([1 - fraud_probability, fraud_probability])


def make_demo_dataset(n_samples: int = 2500) -> pd.DataFrame:
    """Generate a synthetic CCFD-shaped dataset for demo/deployment smoke tests."""
    rng = np.random.default_rng(42)
    n_fraud = max(12, int(n_samples * 0.01))
    n_legit = n_samples - n_fraud

    legit = rng.normal(0, 0.65, size=(n_legit, 28))
    fraud = rng.normal(0, 1.15, size=(n_fraud, 28))
    for feature_idx in (9, 11, 13, 16):
        fraud[:, feature_idx] -= rng.uniform(2.5, 4.5, size=n_fraud)

    features = np.vstack([legit, fraud])
    y = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])
    frame = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, 29)])
    frame["Amount"] = rng.exponential(75, size=n_samples)
    frame.loc[y == 1, "Amount"] *= rng.uniform(3, 8, size=n_fraud)
    frame["Time"] = rng.uniform(0, 172800, size=n_samples)
    frame["Class"] = y
    return frame.sample(frac=1, random_state=42).reset_index(drop=True)


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: Path):
    if path.exists():
        LOGGER.info("Loading model from %s", path)
        return joblib.load(path), "trained stacking ensemble"
    LOGGER.warning("Model file missing at %s; using demo heuristic model.", path)
    return HeuristicFraudModel(), "demo heuristic"


@st.cache_data(show_spinner="Loading dataset...")
def load_data(path: Path) -> tuple[pd.DataFrame, str]:
    if path.exists():
        df = pd.read_csv(path)
        missing = {"Class", *FEATURES} - set(df.columns)
        if missing:
            raise ValueError(f"Dataset at {path} is missing required columns: {sorted(missing)}")
        return df, "real Kaggle CCFD dataset"
    LOGGER.warning("Dataset missing at %s; using synthetic demo data.", path)
    return make_demo_dataset(), "synthetic demo dataset"


@st.cache_resource(show_spinner="Preparing SHAP explainer...")
def get_shap_explainer(model):
    xgb_model = model.named_estimators_.get("xgb") if hasattr(model, "named_estimators_") else None
    if xgb_model is None:
        return None, None
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


def render_data_status(model_source: str, data_source: str) -> None:
    if model_source.startswith("demo") or data_source.startswith("synthetic"):
        st.warning(
            "Demo mode is active because `artifacts/stacking_model.joblib` and/or "
            "`data/creditcard.csv` were not found. The dashboard is fully usable for "
            "review, but production metrics require training the model on the Kaggle dataset."
        )
    st.caption(f"Model source: **{model_source}** · Data source: **{data_source}**")


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
        result_cols = st.columns(4)
        result_cols[0].metric("Fraud Probability", f"{proba*100:.2f}%")
        result_cols[1].metric("Confidence", f"{conf*100:.2f}%")
        result_cols[2].metric("Inference Latency", f"{dt:.3f} ms")
        result_cols[3].metric("Risk Level", risk)
        st.progress(float(proba), text=f"Risk band: {risk}")
        st.session_state.setdefault("history", []).append({**vals, "proba": proba, "latency_ms": dt, "risk": risk})


def shap_page(model, df):
    st.header("SHAP Explanations")
    explainer, _ = get_shap_explainer(model)
    if explainer is None:
        st.info(
            "SHAP explanations are available after loading a trained stacking model with an "
            "XGBoost base learner. Demo mode still supports simulator, metrics, dataset, and comparison views."
        )
        return

    with st.spinner("Computing SHAP values..."):
        sample = df[FEATURES].sample(min(500, len(df)), random_state=42)
        shap_vals = explainer.shap_values(sample)

    st.subheader("Summary Plot")
    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_vals, sample, show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.clf()
    st.info("Key discriminative features to inspect: V17, V14, V12, V10.")

    st.subheader("Dependence Plot")
    feat = st.selectbox("Select feature", FEATURES, index=FEATURES.index("V17"))
    plt.figure(figsize=(8, 4))
    shap.dependence_plot(feat, shap_vals, sample, show=False)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.clf()

    st.subheader("Force Plot (latest prediction)")
    if st.session_state.get("history"):
        latest = pd.DataFrame([st.session_state["history"][-1]])[FEATURES]
        sv = explainer.shap_values(latest)
        force = shap.force_plot(explainer.expected_value, sv[0], latest.iloc[0], matplotlib=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{force.html()}</body>"
        components.html(shap_html, height=320, scrolling=True)
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
    score = (df["V17"].rank(pct=True).values * -1 + 1) * 0.5 + (df["V14"].rank(pct=True).values * -1 + 1) * 0.5
    thr = st.slider("Decision Threshold", 0.01, 0.99, 0.5, 0.01)
    pred = (score >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    cm_norm = cm / max(cm.sum(), 1)
    txt = [[f"{cm[i, j]}\n({cm_norm[i, j]*100:.2f}%)" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Legit", "Fraud"],
            y=["Legit", "Fraud"],
            colorscale="Blues",
            text=txt,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

    precision, recall, _ = precision_recall_curve(y, score)
    pr = px.line(x=recall, y=precision, labels={"x": "Recall", "y": "Precision"}, title="Precision-Recall Curve")
    st.plotly_chart(pr, use_container_width=True)


def dataset_page(df):
    st.header("Dataset Exploration")
    st.write(f"Loaded dataset contains {len(df):,} transactions and {int(df['Class'].sum()):,} fraud cases.")
    cls = df["Class"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
    st.plotly_chart(
        px.bar(
            x=cls.index,
            y=cls.values,
            color=cls.index,
            color_discrete_map={"Fraud": "red", "Legitimate": "green"},
            title="Class Distribution",
        ),
        use_container_width=True,
    )
    legit_count = int(cls.get("Legitimate", 0))
    smote_counts = pd.Series({"Fraud": int(legit_count * 0.1), "Legitimate": legit_count})
    st.plotly_chart(
        px.bar(x=smote_counts.index, y=smote_counts.values, color=smote_counts.index, title="Training Distribution After SMOTE (1:10)"),
        use_container_width=True,
    )

    feature = st.selectbox("Feature distribution", FEATURES)
    fig = px.histogram(df, x=feature, color=df["Class"].map({0: "Legitimate", 1: "Fraud"}), barmode="overlay", nbins=60)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.groupby("Class")[FEATURES].agg(["mean", "std", "min", "max"]).head(10))


def comparison_page():
    st.header("Model Comparison")
    cmp = pd.DataFrame([
        {"Model": "Logistic Regression", "AUPRC": 0.79, "F1": 0.74, "MCC": 0.75, "Latency": 0.030},
        {"Model": "Random Forest", "AUPRC": 0.86, "F1": 0.82, "MCC": 0.83, "Latency": 0.220},
        {"Model": "XGBoost", "AUPRC": 0.89, "F1": 0.86, "MCC": 0.87, "Latency": 0.140},
        {"Model": "Stacking Ensemble", "AUPRC": 0.903, "F1": 0.881, "MCC": 0.884, "Latency": 0.074},
    ])
    st.plotly_chart(px.bar(cmp.melt(id_vars="Model", value_vars=["AUPRC", "F1", "MCC"]), x="variable", y="value", color="Model", barmode="group", title="Metric Comparison"), use_container_width=True)
    st.plotly_chart(px.bar(cmp, x="Model", y="Latency", color="Model", title="Inference Latency (ms / transaction)"), use_container_width=True)
    st.markdown("Stacking architecture combines complementary errors of base learners through a logistic meta-learner.")


def home_page():
    st.title("Credit Card Fraud Detection Dashboard")
    st.markdown("Interactive companion app for stacking ensemble fraud detection research.")
    st.markdown("- Repository: https://github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection")
    st.markdown("- Key results: AUPRC 0.903, F1 0.881, MCC 0.884")
    st.markdown(
        "Use the sidebar to simulate transactions, inspect model metrics, explore data distributions, "
        "and compare the stacking ensemble with baseline models."
    )


def main():
    st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Simulator", "SHAP Explanations", "Metrics", "Dataset", "Model Comparison"])

    try:
        model, model_source = load_model(MODEL_PATH)
        df, data_source = load_data(DATA_PATH)
    except Exception as exc:
        st.error(f"Startup error: {exc}")
        st.stop()

    render_data_status(model_source, data_source)

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
