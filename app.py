"""
app.py — Streamlit Dashboard for Credit Card Fraud Detection
=============================================================
Final Year Project: Ensemble Machine Learning for Online Credit Card Fraud Detection
Author  : Angelo (Cyberangelo-King)
Version : 2.0.0

Pages
-----
- Home           : Project overview, key metric cards, quick-start guide
- Simulator      : Real-time transaction fraud probability
- SHAP           : SHAP waterfall + bar chart explanations
- Metrics        : ROC, PR curve, confusion matrix, calibration
- Dataset        : Feature distributions, class balance, statistics
- Model Comparison: Per-model metric comparison
- About          : Final year project context and methodology summary
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page configuration (must be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="Fraud Detection — Ensemble ML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e0e0e0 !important;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2744 100%);
        border: 1px solid #2d5a8e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 4px;
    }
    .metric-card h3 { color: #90caf9; font-size: 0.85rem; margin: 0; font-weight: 500; }
    .metric-card p  { color: #ffffff; font-size: 2rem; margin: 8px 0 0 0; font-weight: 700; }
    .metric-card span { color: #a5d6a7; font-size: 0.78rem; }
    /* Status badges */
    .badge-excellent {
        background-color: #1b5e20;
        color: #a5d6a7;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-good {
        background-color: #0d47a1;
        color: #90caf9;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    /* Fraud alert */
    .fraud-alert {
        background: linear-gradient(135deg, #7f0000, #b71c1c);
        border: 1px solid #ef9a9a;
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .legit-ok {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border: 1px solid #a5d6a7;
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    /* Section headers */
    .section-header {
        border-left: 4px solid #1976d2;
        padding-left: 12px;
        margin-bottom: 16px;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# MODEL LOADING
# ===========================================================================

MODEL_PATHS = [
    "results/stacking_model.pkl",
    "stacking_model.pkl",
    "models/stacking_model.pkl",
    "scripts/stacking_model.pkl",
]
METRICS_PATHS = [
    "results/metrics.json",
    "metrics.json",
    "scripts/metrics.json",
]


@st.cache_resource(show_spinner="Loading model…")
def load_model() -> Optional[Dict[str, Any]]:
    """Load the stacking model bundle. Returns None if not found."""
    try:
        import joblib
    except ImportError:
        return None
    for path in MODEL_PATHS:
        if Path(path).exists():
            try:
                bundle = joblib.load(path)
                required = {"base_learners", "meta_learner", "feature_names"}
                if not isinstance(bundle, dict) or not required.issubset(bundle):
                    continue
                if bundle.get("feature_schema_version") != "ulb-pca-elapsed-hours-v1":
                    # Pre-v2 artifacts used a different Time/Hour contract.
                    continue
                pipeline_mode = bool(bundle.get("preprocessing_in_estimators", False))
                if not pipeline_mode and bundle.get("scaler") is None:
                    continue
                return bundle
            except Exception:
                continue
    return None


@st.cache_data(show_spinner=False)
def load_metrics() -> Optional[Dict[str, Any]]:
    """Load saved metrics JSON. Returns None if not found."""
    for path in METRICS_PATHS:
        if Path(path).exists():
            try:
                with open(path) as f:
                    report = json.load(f)
                if report.get("benchmark_status") not in {"synthetic_demo", "experimental_benchmark"}:
                    # Hide legacy reports whose split/threshold provenance is unknown.
                    continue
                return report
            except Exception:
                continue
    return None


@st.cache_data(show_spinner=False)
def load_sample_data() -> Optional[pd.DataFrame]:
    """Load a sample of the dataset for display."""
    data_paths = [
        "data/creditcard.csv",
        "data/synthetic_creditcard.csv",
        "scripts/synthetic_creditcard.csv",
        "synthetic_creditcard.csv",
    ]
    for path in data_paths:
        if Path(path).exists():
            try:
                return pd.read_csv(path, nrows=5000)
            except Exception:
                continue
    return None


def predict_transaction(
    bundle: Dict[str, Any],
    features: Dict[str, float],
) -> Tuple[float, np.ndarray]:
    """
    Run inference on a single transaction.

    Returns
    -------
    fraud_prob   : float — probability of fraud
    base_probs   : np.ndarray — [lr_prob, rf_prob, xgb_prob]
    """
    import numpy as np
    feature_names = bundle.get("feature_names", [f"V{i}" for i in range(1, 29)] + ["Amount_log", "Hour"])
    x = np.array([[features.get(f, 0.0) for f in feature_names]], dtype=np.float32)
    if bundle.get("preprocessing_in_estimators", False):
        model_input = x
    else:
        scaler = bundle.get("scaler")
        if scaler is None:
            raise ValueError("Model bundle has no supported preprocessing strategy.")
        model_input = scaler.transform(x)
    base_probs = np.array([
        clf.predict_proba(model_input)[0, 1] for clf in bundle["base_learners"]
    ])
    meta_x = base_probs.reshape(1, -1)
    fraud_prob = float(bundle["meta_learner"].predict_proba(meta_x)[0, 1])
    return fraud_prob, base_probs


# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:
    st.markdown("## 🛡️ Fraud Detector")
    st.markdown("*Ensemble ML — Final Year Project*")
    st.divider()

    page = st.radio(
        "Navigate",
        options=[
            "🏠 Home",
            "⚡ Simulator",
            "🔍 SHAP Explanations",
            "📊 Metrics",
            "📁 Dataset",
            "🤖 Model Comparison",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Model status indicator
    bundle = load_model()
    if bundle is not None:
        st.success("✅ Model loaded")
        opt_thresh = bundle.get("optimal_threshold", 0.5)
        st.caption(f"Optimal threshold: {opt_thresh:.3f}")
    else:
        st.warning("⚠️ No model found")
        st.caption("Run `python scripts/create_demo_model.py` first")

    st.divider()
    st.markdown(
        "<div style='color:#888;font-size:0.75rem'>Research prototype · 2026</div>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# PAGE: HOME
# ===========================================================================

if page == "🏠 Home":
    st.title("🛡️ Ensemble Machine Learning for Credit Card Fraud Detection")
    st.markdown(
        "> *A research prototype exploring stacked classifiers for highly imbalanced transaction data. "
        "Benchmark metrics are shown only when a saved experiment report is available.*"
    )

    # ── Key Metrics Cards ──────────────────────────────────────────────────
    st.markdown("### Performance snapshot")
    metrics_data = load_metrics()

    if metrics_data:
        m = metrics_data.get("threshold_optimal", {})
        metric_keys = [
            ("AUPRC", "Average precision", "Primary imbalanced-data metric"),
            ("F1", "F1-score", "At the saved operating threshold"),
            ("MCC", "Matthews correlation", "Uses all confusion-matrix cells"),
            ("ROC_AUC", "ROC-AUC", "Ranking discrimination"),
            ("Precision", "Precision", "Share of alerts that are fraud"),
            ("Recall", "Recall", "Share of fraud cases detected"),
        ]
        columns = st.columns(3)
        for index, (key, label, caption) in enumerate(metric_keys):
            with columns[index % 3]:
                value = m.get(key)
                st.metric(label, f"{float(value):.4f}" if value is not None else "—")
                st.caption(caption)
        if metrics_data.get("note"):
            st.info(f"Dataset note: {metrics_data['note']}")
        if metrics_data.get("split_protocol"):
            st.caption(
                "Evaluation split: "
                + str(metrics_data["split_protocol"].get("strategy", "not recorded"))
                + ". See results/metrics.json for protocol details."
            )
    else:
        st.info(
            "No experiment metrics are loaded. The dashboard can still run in demo mode, "
            "but benchmark scores will not be invented or inferred. Run the Kaggle experiment "
            "to generate a genuine evaluation report."
        )

    st.divider()

    # ── Architecture ───────────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown('<div class="section-header"><h3>System Architecture</h3></div>', unsafe_allow_html=True)
        st.code("""
┌─────────────────────────────────────┐
│    Transaction Features (30-dim)    │
│ V1–V28 PCA + Amount_log + Hour      │
└────────────────┬────────────────────┘
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
  [LR Base]  [RF Base]  [XGB Base]
     │           │           │
     └─────┬─────┘───────────┘
           │  OOF Meta-Features
           ▼
    [Meta-Learner LR]
           │
    P(fraud | x)
""", language="text")

    with col_b:
        st.markdown('<div class="section-header"><h3>Quick Start</h3></div>', unsafe_allow_html=True)
        st.code("""# 1. Clone and setup
git clone https://github.com/Cyberangelo-King/\\
  Ensemble-Machine-Learning-for-Online-\\
  Credit-Card-Fraud-Detection.git
bash setup.sh

# 2. Create demo model
python scripts/create_demo_model.py

# 3. Launch dashboard
streamlit run app.py""", language="bash")

        st.markdown("**For full experiment (Kaggle dataset):**")
        st.code("""python src/run_experiment.py \\
    --data-path data/creditcard.csv \\
    --output-dir results/ \\
    --fig-dir figures/""", language="bash")

    st.divider()

    # ── Dataset summary ────────────────────────────────────────────────────
    st.markdown('<div class="section-header"><h3>Dataset Summary</h3></div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Transactions", "284,807")
    d2.metric("Fraud Cases", "492 (0.172%)")
    d3.metric("Imbalance Ratio", "578 : 1")
    d4.metric("Feature Dimensions", "30 (28 PCA + 2 engineered)")

# ===========================================================================
# PAGE: SIMULATOR
# ===========================================================================

elif page == "⚡ Simulator":
    st.title("⚡ Transaction Fraud Simulator")
    st.markdown(
        "Enter transaction features to get a real-time fraud probability estimate from the stacking ensemble."
    )

    if bundle is None:
        st.error(
            "❌ **Model not loaded.** Please run `python scripts/create_demo_model.py` first, "
            "then refresh this page."
        )
        st.stop()

    opt_thresh = bundle.get("optimal_threshold", 0.5)

    with st.expander("💡 About the Features", expanded=False):
        st.markdown("""
        - **V1–V14**: PCA-transformed transaction features (most discriminative)
        - **Amount**: Transaction value in Euros
        - **Hour**: Elapsed hours since the first transaction in the source dataset (0–48), not wall-clock time
        
        For a demo, use the default values or try the pre-set fraud/legitimate scenarios below.
        """)

    # Pre-set scenarios
    col_presets = st.columns(3)
    scenario = None
    with col_presets[0]:
        if st.button("🟢 Typical Legitimate", use_container_width=True):
            scenario = "legitimate"
    with col_presets[1]:
        if st.button("🔴 Suspicious Transaction", use_container_width=True):
            scenario = "suspicious"
    with col_presets[2]:
        if st.button("🟡 Borderline Case", use_container_width=True):
            scenario = "borderline"

    # Default values per scenario
    scenario_defaults = {
        "legitimate": {"V1": 1.8, "V2": 0.3, "V3": 2.1, "V4": -0.2, "Amount": 45.0, "Hour": 14.0},
        "suspicious": {"V1": -4.5, "V2": 1.8, "V3": -7.2, "V4": 4.1, "Amount": 299.0, "Hour": 3.0},
        "borderline": {"V1": -1.2, "V2": 0.8, "V3": -1.5, "V4": 1.1, "Amount": 120.0, "Hour": 22.0},
    }
    defaults = scenario_defaults.get(scenario, {}) if scenario else {}

    st.divider()
    st.markdown("### Transaction Features")

    with st.form("transaction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**PCA Features (V1–V14)**")
            v_vals = {}
            for i in range(1, 15):
                default_val = defaults.get(f"V{i}", 0.0)
                v_vals[f"V{i}"] = st.number_input(
                    f"V{i}", value=float(default_val),
                    min_value=-30.0, max_value=30.0,
                    step=0.01, format="%.4f",
                    key=f"v{i}",
                )

        with col2:
            st.markdown("**PCA Features (V15–V28)**")
            for i in range(15, 29):
                default_val = defaults.get(f"V{i}", 0.0)
                v_vals[f"V{i}"] = st.number_input(
                    f"V{i}", value=float(default_val),
                    min_value=-30.0, max_value=30.0,
                    step=0.01, format="%.4f",
                    key=f"v{i}",
                )

            st.markdown("**Derived Features**")
            amount = st.number_input(
                "Amount (€)", value=float(defaults.get("Amount", 50.0)),
                min_value=0.0, max_value=50000.0,
                step=0.01,
            )
            hour = st.slider(
                "Elapsed Hours Since First Transaction", min_value=0.0, max_value=48.0,
                value=float(defaults.get("Hour", 12.0)),
                step=0.1,
            )

        submitted = st.form_submit_button(
            "🔍 Analyse Transaction",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        with st.spinner("Running ensemble inference…"):
            import math
            features = {**v_vals}
            features["Amount_log"] = math.log1p(amount)
            features["Hour"] = hour

            fraud_prob, base_probs = predict_transaction(bundle, features)
            is_fraud = fraud_prob >= opt_thresh

        st.divider()
        col_r1, col_r2 = st.columns([1, 1])

        with col_r1:
            if is_fraud:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h2>🚨 FRAUD DETECTED</h2>
                    <p style="font-size: 2.5rem; font-weight: bold; margin: 10px 0">{fraud_prob:.1%}</p>
                    <p>Fraud Probability</p>
                    <p style="font-size: 0.9rem; opacity: 0.8">Exceeds optimal threshold of {opt_thresh:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-ok">
                    <h2>✅ LEGITIMATE</h2>
                    <p style="font-size: 2.5rem; font-weight: bold; margin: 10px 0">{fraud_prob:.1%}</p>
                    <p>Fraud Probability</p>
                    <p style="font-size: 0.9rem; opacity: 0.8">Below optimal threshold of {opt_thresh:.3f}</p>
                </div>
                """, unsafe_allow_html=True)

        with col_r2:
            st.markdown("**Base Learner Predictions**")
            learner_names = ["Logistic Regression", "Random Forest", "XGBoost"]
            for name, prob in zip(learner_names, base_probs):
                colour = "#ef5350" if prob >= 0.5 else "#66bb6a"
                st.markdown(f"**{name}:** `{prob:.4f}`")
                st.progress(float(prob))

        # Probability gauge
        st.markdown("**Fraud Probability Gauge**")
        gauge_data = pd.DataFrame({
            "Metric": ["Fraud Probability", "Legitimate Probability"],
            "Value": [fraud_prob, 1 - fraud_prob],
        })
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 1.5))
            ax.barh(["Score"], [fraud_prob], color="#ef5350" if is_fraud else "#66bb6a",
                    height=0.5, alpha=0.85)
            ax.barh(["Score"], [1 - fraud_prob], left=[fraud_prob],
                    color="#37474f", height=0.5, alpha=0.5)
            ax.axvline(x=opt_thresh, color="orange", linestyle="--", lw=2,
                       label=f"Threshold ({opt_thresh:.3f})")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.legend(loc="upper right", fontsize=9)
            ax.set_title("Fraud Probability vs Threshold", fontsize=11)
            fig.patch.set_alpha(0)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except ImportError:
            st.info("Install matplotlib for the probability gauge chart.")

# ===========================================================================
# PAGE: SHAP EXPLANATIONS
# ===========================================================================

elif page == "🔍 SHAP Explanations":
    st.title("🔍 SHAP Explanations")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) values quantify each base learner's "
        "contribution to the fraud probability for a specific transaction."
    )

    if bundle is None:
        st.error("❌ Model not loaded. Run `python scripts/create_demo_model.py` first.")
        st.stop()

    # Check shap is available
    try:
        import shap
        import matplotlib.pyplot as plt
        SHAP_AVAILABLE = True
    except ImportError:
        SHAP_AVAILABLE = False
        st.warning("⚠️ `shap` library not installed. Run `pip install shap` to enable SHAP charts.")

    col_inp1, col_inp2 = st.columns(2)
    with col_inp1:
        st.markdown("**Transaction to Explain**")
        shap_v1 = st.slider("V1 (most discriminative)", -20.0, 10.0, 0.0, 0.1)
        shap_v3 = st.slider("V3", -20.0, 10.0, 0.0, 0.1)
        shap_v4 = st.slider("V4", -5.0, 15.0, 0.0, 0.1)
        shap_amount = st.number_input("Amount (€)", 0.0, 10000.0, 50.0, 1.0)
    with col_inp2:
        shap_v2 = st.slider("V2", -15.0, 15.0, 0.0, 0.1)
        shap_v10 = st.slider("V10", -20.0, 10.0, 0.0, 0.1)
        shap_v14 = st.slider("V14", -20.0, 10.0, 0.0, 0.1)
        shap_hour = st.slider("Elapsed hours", 0.0, 48.0, 12.0, 0.1)

    import math
    shap_features = {
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "V1": shap_v1, "V2": shap_v2, "V3": shap_v3, "V4": shap_v4,
        "V10": shap_v10, "V14": shap_v14,
        "Amount_log": math.log1p(shap_amount),
        "Hour": shap_hour,
    }

    if st.button("🔍 Explain This Transaction", type="primary"):
        with st.spinner("Computing SHAP values…"):
            fraud_prob, base_probs = predict_transaction(bundle, shap_features)
            opt_thresh = bundle.get("optimal_threshold", 0.5)
            verdict = "FRAUD" if fraud_prob >= opt_thresh else "LEGITIMATE"

        st.markdown(f"**Result:** `{verdict}` — Fraud probability: **{fraud_prob:.3f}**")
        st.divider()

        if SHAP_AVAILABLE:
            try:
                meta_lr = bundle["meta_learner"]
                meta_x = base_probs.reshape(1, -1)
                feature_names = ["LR Probability", "RF Probability", "XGB Probability"]

                # Compute SHAP values
                explainer = shap.LinearExplainer(
                    meta_lr,
                    np.zeros((1, 3)),  # background (mean = 0 for standardised)
                    feature_perturbation="interventional",
                )
                shap_values = explainer.shap_values(meta_x)
                if isinstance(shap_values, list):
                    sv = np.array(shap_values[1])
                else:
                    sv = np.array(shap_values)
                sv_1d = sv.flatten()
                base_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)

                tab1, tab2 = st.tabs(["Waterfall Chart", "Bar Chart"])

                with tab1:
                    st.markdown("#### SHAP Waterfall Chart")
                    st.markdown(
                        "The waterfall chart shows how each base learner's prediction "
                        "pushes the final fraud probability up (red) or down (blue) from the baseline."
                    )
                    fig, ax = plt.subplots(figsize=(9, 4))
                    x_pos = np.arange(len(feature_names))
                    colours = ["#ef5350" if v > 0 else "#42a5f5" for v in sv_1d]
                    cumulative = base_value
                    starts = []
                    for v in sv_1d:
                        starts.append(cumulative if v > 0 else cumulative + v)
                        cumulative += v
                    bars = ax.barh(x_pos, np.abs(sv_1d), left=starts, color=colours,
                                   height=0.5, edgecolor="white", linewidth=0.5)
                    ax.axvline(x=base_value, color="gray", linestyle="--", alpha=0.7, label=f"Base: {base_value:.3f}")
                    ax.axvline(x=fraud_prob, color="orange", linestyle="-", lw=2, label=f"Output: {fraud_prob:.3f}")
                    ax.set_yticks(x_pos)
                    ax.set_yticklabels(feature_names, fontsize=11)
                    ax.set_xlabel("SHAP Value (impact on fraud probability)", fontsize=11)
                    ax.set_title("SHAP Waterfall — Meta-Learner Explanation", fontsize=13, fontweight="bold")
                    ax.legend(fontsize=10)
                    for bar, val in zip(bars, sv_1d):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_y() + bar.get_height() / 2,
                                f"{val:+.4f}", ha="center", va="center",
                                fontsize=10, color="white", fontweight="bold")
                    fig.patch.set_facecolor("#0e1117")
                    ax.set_facecolor("#1a1a2e")
                    ax.tick_params(colors="white")
                    ax.xaxis.label.set_color("white")
                    ax.title.set_color("white")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                with tab2:
                    st.markdown("#### SHAP Bar Chart — Feature Contributions")
                    fig2, ax2 = plt.subplots(figsize=(9, 4))
                    colours2 = ["#ef5350" if v > 0 else "#42a5f5" for v in sv_1d]
                    ax2.barh(feature_names, sv_1d, color=colours2, height=0.5, edgecolor="white")
                    ax2.axvline(x=0, color="white", lw=1, alpha=0.5)
                    ax2.set_xlabel("SHAP Value", fontsize=11)
                    ax2.set_title("SHAP Bar — Signed Feature Contributions", fontsize=13, fontweight="bold")
                    for i, val in enumerate(sv_1d):
                        ax2.text(val + (0.001 if val >= 0 else -0.001), i,
                                 f"{val:+.4f}", va="center",
                                 ha="left" if val >= 0 else "right",
                                 fontsize=10, color="white")
                    fig2.patch.set_facecolor("#0e1117")
                    ax2.set_facecolor("#1a1a2e")
                    ax2.tick_params(colors="white")
                    ax2.xaxis.label.set_color("white")
                    ax2.title.set_color("white")
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

                st.markdown("#### SHAP Value Summary")
                shap_df = pd.DataFrame({
                    "Base Learner": feature_names,
                    "Prediction": [f"{p:.4f}" for p in base_probs],
                    "SHAP Value": [f"{v:+.6f}" for v in sv_1d],
                    "Direction": ["↑ Fraud" if v > 0 else "↓ Legitimate" for v in sv_1d],
                })
                st.dataframe(shap_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.warning(f"SHAP computation failed: {e}")
                # Fallback: show base probs as bar chart
                st.markdown("#### Base Learner Predictions (fallback)")
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.barh(["LR", "RF", "XGB"], base_probs, color=["#ef5350" if p >= 0.5 else "#42a5f5" for p in base_probs])
                ax.set_xlim(0, 1)
                ax.axvline(x=0.5, color="white", linestyle="--", alpha=0.5)
                st.pyplot(fig)
                plt.close(fig)
        else:
            # SHAP not available — show simple bar chart of base probs
            try:
                import matplotlib.pyplot as plt
                st.markdown("#### Base Learner Probability Breakdown")
                fig, ax = plt.subplots(figsize=(8, 3))
                names = ["LR Prob", "RF Prob", "XGB Prob"]
                colours = ["#ef5350" if p >= 0.5 else "#42a5f5" for p in base_probs]
                bars = ax.barh(names, base_probs, color=colours, height=0.5)
                ax.axvline(x=0.5, color="orange", linestyle="--", lw=2, label="0.5 threshold")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Predicted Fraud Probability")
                ax.set_title("Base Learner Predictions")
                for bar, val in zip(bars, base_probs):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{val:.4f}", va="center", fontsize=11)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except ImportError:
                for name, prob in zip(["LR", "RF", "XGB"], base_probs):
                    st.metric(name, f"{prob:.4f}")

# ===========================================================================
# PAGE: METRICS
# ===========================================================================

elif page == "📊 Metrics":
    st.title("📊 Model Evaluation")
    st.markdown("Evaluation reports are loaded from the latest saved results/metrics.json.")

    metrics_data = load_metrics()
    if not metrics_data:
        st.info(
            "No saved evaluation report is available. Run the full experiment with the Kaggle dataset "
            "to generate a final holdout report. The dashboard will not substitute historical or "
            "synthetic scores for missing benchmark results."
        )
    else:
        if metrics_data.get("benchmark_status") == "synthetic_demo":
            st.warning(
                "SYNTHETIC DEMONSTRATION ONLY — these results come from generated data. "
                "They are not evidence of performance on the Kaggle benchmark or real transactions."
            )
            st.markdown("### Synthetic demo holdout results")
        else:
            st.markdown("### Chronological benchmark holdout results")
        if metrics_data.get("note"):
            st.caption(metrics_data["note"])
        m_opt = metrics_data.get("threshold_optimal", {})
        m_05 = metrics_data.get("threshold_0.5", {})
        threshold = metrics_data.get("optimal_threshold_value")
        selection = metrics_data.get("threshold_selection", {})
        protocol = metrics_data.get("split_protocol", {})

        if selection:
            st.caption(
                f"Operating threshold: {threshold:.5f} — selected using "
                f"{selection.get('selection_data', selection.get('method', 'validation data'))}. "
                f"Final test used for selection: {selection.get('test_set_used', 'not recorded')}."
            )
        if protocol:
            st.caption(
                f"Split protocol: {protocol.get('strategy', 'not recorded')} "
                f"(train {protocol.get('train_fraction', '—')}, "
                f"validation {protocol.get('validation_fraction', '—')}, "
                f"test {protocol.get('test_fraction', '—')})."
            )

        metrics = [
            ("AUPRC", "Average precision"),
            ("ROC_AUC", "ROC-AUC"),
            ("F1", "F1-score"),
            ("MCC", "Matthews correlation"),
            ("Precision", "Precision"),
            ("Recall", "Recall"),
        ]
        columns = st.columns(3)
        for index, (key, label) in enumerate(metrics):
            with columns[index % 3]:
                value = m_opt.get(key)
                st.metric(label, f"{float(value):.4f}" if value is not None else "—")

        with st.expander("Compare with the fixed 0.50 decision threshold"):
            comparison = []
            for key, label in metrics:
                comparison.append({
                    "Metric": label,
                    "At 0.50": m_05.get(key),
                    "At validation-selected threshold": m_opt.get(key),
                })
            st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

        fig_paths = {
            "ROC Curve": ["figures/roc_curve.png", "roc_curve.png"],
            "Precision–Recall Curve": ["figures/pr_curve.png", "pr_curve.png"],
            "Confusion Matrix": ["figures/confusion_matrix.png", "confusion_matrix.png"],
            "Calibration / Reliability Curve": ["figures/calibration_curve.png", "calibration_curve.png"],
        }
        for fig_name, candidates in fig_paths.items():
            image_path = next((candidate for candidate in candidates if Path(candidate).exists()), None)
            if image_path:
                st.markdown(f"**{fig_name}**")
                st.image(image_path, use_container_width=True)
            else:
                st.caption(f"{fig_name}: not generated for this run.")

        st.caption(
            "AUPRC and ROC-AUC are ranking metrics. F1, MCC, precision and recall depend on "
            "the saved operating threshold. A reliability curve is diagnostic; it does not by "
            "itself mean the probabilities have been calibrated."
        )

# PAGE: DATASET
# ===========================================================================

elif page == "📁 Dataset":
    st.title("📁 Dataset Explorer")
    st.markdown(
        "Inspect the dataset currently available to the dashboard. The public ULB benchmark "
        "contains PCA-anonymised V1–V28 features, Amount, Time and a binary Class label."
    )

    df = load_sample_data()
    if df is None:
        st.info(
            "No local CSV preview is available. Place the Kaggle creditcard.csv file in data/, "
            "or generate synthetic data for dashboard demonstration."
        )
    else:
        total_rows = len(df)
        fraud_count = int(df["Class"].sum()) if "Class" in df.columns else None
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Preview rows", f"{total_rows:,}")
        metrics_cols[1].metric("Fraud rows in preview", f"{fraud_count:,}" if fraud_count is not None else "—")
        metrics_cols[2].metric("Columns", f"{len(df.columns):,}")
        metrics_cols[3].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

        if "Class" not in df.columns:
            st.warning("The current CSV preview has no Class label; supervised evaluation is unavailable.")
        else:
            tab1, tab2, tab3 = st.tabs(["Sample Data", "Class Distribution", "Feature Distributions"])
            with tab1:
                st.dataframe(df.head(100), use_container_width=True)
                st.caption(f"Showing up to 100 of {total_rows:,} rows loaded by the dashboard.")
            with tab2:
                import matplotlib.pyplot as plt
                counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(["Legitimate", "Fraud"], counts.values)
                ax.set_ylabel("Rows in loaded preview")
                ax.set_title("Class Distribution — Loaded Preview")
                ax.grid(axis="y", alpha=0.25)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption("Preview proportions are not necessarily representative of the full dataset.")
            with tab3:
                candidates = [c for c in [f"V{i}" for i in range(1, 15)] + ["Amount", "Time"] if c in df.columns]
                selected = st.selectbox("Select feature", options=candidates)
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                for label, name in [(0, "Legitimate"), (1, "Fraud")]:
                    values = df.loc[df["Class"] == label, selected]
                    if len(values):
                        ax.hist(values, bins=50, alpha=0.55, label=name, density=True)
                ax.set_xlabel(selected)
                ax.set_ylabel("Density")
                ax.legend()
                ax.set_title(f"{selected} distribution by label")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# PAGE: MODEL COMPARISON
# ===========================================================================

elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    st.markdown(
        "Comparisons are populated from the current run's held-out test predictions. "
        "No manually entered or historical benchmark scores are displayed as current results."
    )

    metrics_data = load_metrics()
    base_metrics = metrics_data.get("base_model_test_metrics", {}) if metrics_data else {}
    ensemble_metrics = metrics_data.get("threshold_0.5", {}) if metrics_data else {}

    if base_metrics and ensemble_metrics:
        rows = []
        for model_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            if model_name in base_metrics:
                item = base_metrics[model_name]
                rows.append({
                    "Model": model_name,
                    "AUPRC": item.get("AUPRC"),
                    "ROC-AUC": item.get("ROC_AUC"),
                    "F1 @ 0.50": item.get("F1"),
                    "MCC @ 0.50": item.get("MCC"),
                    "Precision @ 0.50": item.get("Precision"),
                    "Recall @ 0.50": item.get("Recall"),
                })
        rows.append({
            "Model": "Stacking Ensemble",
            "AUPRC": ensemble_metrics.get("AUPRC"),
            "ROC-AUC": ensemble_metrics.get("ROC_AUC"),
            "F1 @ 0.50": ensemble_metrics.get("F1"),
            "MCC @ 0.50": ensemble_metrics.get("MCC"),
            "Precision @ 0.50": ensemble_metrics.get("Precision"),
            "Recall @ 0.50": ensemble_metrics.get("Recall"),
        })
        comparison = pd.DataFrame(rows)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        chart_columns = ["AUPRC", "F1 @ 0.50", "MCC @ 0.50", "ROC-AUC"]
        st.bar_chart(comparison.set_index("Model")[chart_columns])
    else:
        st.info(
            "Per-learner test metrics are not available in this report yet. Run the full research "
            "experiment to generate an apples-to-apples comparison on the same untouched test partition."
        )

    ablation_path = Path("results/ablation_results.csv")
    if ablation_path.exists():
        st.markdown("### Stacking ablation")
        ablation = pd.read_csv(ablation_path)
        st.dataframe(ablation, use_container_width=True, hide_index=True)
    else:
        st.caption("No ablation report has been generated for the current checkout.")

# PAGE: ABOUT
# ===========================================================================

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## Research prototype — credit-card transaction fraud detection

    This repository studies a stacked classification approach built from Logistic
    Regression, Random Forest and XGBoost, with a Logistic Regression meta-learner.
    It is an academic/research prototype and a demonstration interface, not a
    certified fraud-control system and not connected to a card network or bank.

    ### Methodological controls

    - Scaling and SMOTE are encapsulated in an imbalanced-learn pipeline so each
      cross-validation fold fits transformations and synthetic samples using its
      training rows only.
    - Base-model meta-features are generated out of fold.
    - The default evaluation protocol uses chronological train/validation/test
      partitions. The validation partition selects the operating threshold; the
      final test partition is not used for threshold selection.
    - Synthetic dashboard data is a UI demonstration, not benchmark evidence.

    ### Dataset and limitations

    The public ULB benchmark is a small, anonymised historical dataset spanning
    roughly two days. V1–V28 are PCA components, and Time is elapsed seconds since
    the first transaction. This is insufficient to establish performance on
    modern bank transactions, new institutions, evolving fraud patterns or a
    genuine live stream. External validation, drift monitoring, human review,
    operational controls and independent security assessment remain future work.

    ### Technical stack

    | Layer | Technology |
    |---|---|
    | Modelling | Python, scikit-learn, XGBoost |
    | Imbalance handling | imbalanced-learn / SMOTE |
    | Explainability | SHAP |
    | Dashboard | Streamlit |
    | Inference API | FastAPI |
    | Frontend | React, Vite, Tailwind |
    | CI | GitHub Actions |

    ### References

    - Chawla et al. (2002), SMOTE: Synthetic Minority Over-sampling Technique.
    - Wolpert (1992), Stacked Generalization.
    - Chen & Guestrin (2016), XGBoost: A Scalable Tree Boosting System.
    - Lundberg & Lee (2017), A Unified Approach to Interpreting Model Predictions.
    """)

    st.divider()
    st.markdown(
        "<div class='footer'>Built with Streamlit · Research prototype · 2026</div>",
        unsafe_allow_html=True,
    )
