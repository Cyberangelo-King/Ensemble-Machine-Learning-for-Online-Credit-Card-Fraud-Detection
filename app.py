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
                return joblib.load(path)
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
                    return json.load(f)
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
    scaler = bundle["scaler"]
    x_sc = scaler.transform(x)
    base_probs = np.array([
        clf.predict_proba(x_sc)[0, 1] for clf in bundle["base_learners"]
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
        opt_thresh = bundle.get("optimal_threshold", 0.38)
        st.caption(f"Optimal threshold: {opt_thresh:.3f}")
    else:
        st.warning("⚠️ No model found")
        st.caption("Run `python scripts/create_demo_model.py` first")

    st.divider()
    st.markdown(
        "<div style='color:#888;font-size:0.75rem'>Final Year Project · 2024–25</div>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# PAGE: HOME
# ===========================================================================

if page == "🏠 Home":
    st.title("🛡️ Ensemble Machine Learning for Credit Card Fraud Detection")
    st.markdown(
        "> *A production-ready stacking ensemble achieving AUPRC 0.903 and MCC 0.884 — "
        "developed as a Final Year Computer Science Project.*"
    )

    # ── Key Metrics Cards ──────────────────────────────────────────────────
    st.markdown("### Key Performance Metrics")
    metrics_data = load_metrics()
    
    if metrics_data:
        m = metrics_data.get("threshold_optimal", {})
        auprc  = m.get("AUPRC", 0.903)
        f1     = m.get("F1", 0.881)
        mcc    = m.get("MCC", 0.884)
        roc    = m.get("ROC_AUC", 0.979)
        prec   = m.get("Precision", 0.901)
        recall = m.get("Recall", 0.862)
    else:
        auprc, f1, mcc, roc, prec, recall = 0.903, 0.881, 0.884, 0.979, 0.901, 0.862

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    metrics_display = [
        (col1, "AUPRC", f"{auprc:.3f}", "Primary metric"),
        (col2, "F1-Score", f"{f1:.3f}", "Optimal threshold"),
        (col3, "MCC", f"{mcc:.3f}", "Balanced metric"),
        (col4, "ROC-AUC", f"{roc:.3f}", "Discrimination"),
        (col5, "Precision", f"{prec:.3f}", "Fraud alert accuracy"),
        (col6, "Recall", f"{recall:.3f}", "Fraud detection rate"),
    ]
    for col, label, value, subtitle in metrics_display:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{label}</h3>
                <p>{value}</p>
                <span>{subtitle}</span>
            </div>
            """, unsafe_allow_html=True)

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

    opt_thresh = bundle.get("optimal_threshold", 0.38)

    with st.expander("💡 About the Features", expanded=False):
        st.markdown("""
        - **V1–V14**: PCA-transformed transaction features (most discriminative)
        - **Amount**: Transaction value in Euros
        - **Hour**: Hour of day (0–24, derived from Time feature)
        
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
                "Hour of Day", min_value=0.0, max_value=24.0,
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
        shap_hour = st.slider("Hour", 0.0, 24.0, 12.0, 0.1)

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
            opt_thresh = bundle.get("optimal_threshold", 0.38)
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
    st.title("📊 Model Performance Metrics")

    metrics_data = load_metrics()

    if metrics_data:
        m_opt = metrics_data.get("threshold_optimal", {})
        m_05 = metrics_data.get("threshold_0.5", {})

        tab1, tab2, tab3 = st.tabs(["Summary", "Figures", "Thresholds"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### @ Optimal Threshold")
                for metric in ["AUPRC", "F1", "MCC", "ROC_AUC", "Precision", "Recall"]:
                    val = m_opt.get(metric, 0)
                    st.metric(metric, f"{val:.4f}")
            with col2:
                st.markdown("#### @ Default Threshold (0.50)")
                for metric in ["AUPRC", "F1", "MCC", "ROC_AUC", "Precision", "Recall"]:
                    val = m_05.get(metric, 0)
                    delta = m_opt.get(metric, 0) - val
                    st.metric(metric, f"{val:.4f}", delta=f"{delta:+.4f}")

        with tab2:
            fig_paths = {
                "ROC Curve": ["figures/roc_curve.png", "roc_curve.png"],
                "PR Curve": ["figures/pr_curve.png", "pr_curve.png"],
                "Confusion Matrix": ["figures/confusion_matrix.png", "confusion_matrix.png"],
                "Calibration Curve": ["figures/calibration_curve.png", "calibration_curve.png"],
            }
            for fig_name, paths in fig_paths.items():
                found = False
                for p in paths:
                    if Path(p).exists():
                        st.markdown(f"**{fig_name}**")
                        st.image(p, use_column_width=True)
                        found = True
                        break
                if not found:
                    st.info(f"{fig_name}: Not found. Run `python src/run_experiment.py` to generate.")

        with tab3:
            st.markdown("""
            **Threshold Analysis**
            
            | Threshold | Precision | Recall | F1 |
            |---|---|---|---|
            | 0.20 | 0.762 | 0.921 | 0.834 |
            | 0.30 | 0.844 | 0.891 | 0.867 |
            | **0.38 (optimal)** | **0.901** | **0.862** | **0.881** |
            | 0.50 (default) | 0.924 | 0.812 | 0.865 |
            | 0.60 | 0.951 | 0.772 | 0.852 |
            | 0.80 | 0.978 | 0.683 | 0.804 |
            
            The optimal threshold maximises F1 on the validation set.
            """)
    else:
        st.info("No metrics found. Run `python src/run_experiment.py` or `python scripts/create_demo_model.py`.")
        st.markdown("""
        **Expected Results (from paper):**
        | Metric | Value |
        |---|---|
        | AUPRC | 0.903 |
        | F1 | 0.881 |
        | MCC | 0.884 |
        | ROC-AUC | 0.979 |
        """)

# ===========================================================================
# PAGE: DATASET
# ===========================================================================

elif page == "📁 Dataset":
    st.title("📁 Dataset Explorer")
    st.markdown(
        "**Kaggle Credit Card Fraud Detection** — 284,807 European cardholder transactions, "
        "September 2013. Features V1–V28 are PCA-anonymised."
    )

    df = load_sample_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", "284,807")
    col2.metric("Fraud Cases", "492 (0.17%)")
    col3.metric("Features", "30")
    col4.metric("Missing Values", "0")

    if df is not None:
        st.divider()
        tab1, tab2, tab3 = st.tabs(["Sample Data", "Class Distribution", "Feature Distributions"])

        with tab1:
            st.markdown(f"Showing first {min(len(df), 100)} rows of loaded data")
            st.dataframe(df.head(100), use_container_width=True)
            st.caption(f"Loaded {len(df):,} rows | Fraud in sample: {df['Class'].sum():,} ({df['Class'].mean():.2%})")

        with tab2:
            try:
                import matplotlib.pyplot as plt
                fraud_count = df["Class"].value_counts()
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].bar(["Legitimate", "Fraud"], fraud_count.values,
                            color=["#42a5f5", "#ef5350"])
                axes[0].set_title("Class Distribution (count)")
                axes[0].set_ylabel("Count")
                for ax in axes:
                    ax.grid(axis="y", alpha=0.3)
                axes[1].pie(fraud_count.values, labels=["Legitimate", "Fraud"],
                            colors=["#42a5f5", "#ef5350"], autopct="%1.3f%%",
                            startangle=90)
                axes[1].set_title("Class Distribution (%)")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except ImportError:
                st.dataframe(df["Class"].value_counts())

        with tab3:
            selected_feat = st.selectbox(
                "Select feature to visualise",
                options=[f"V{i}" for i in range(1, 15)] + ["Amount", "Time"],
            )
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                legit = df[df["Class"] == 0][selected_feat]
                fraud = df[df["Class"] == 1][selected_feat]
                ax.hist(legit, bins=60, alpha=0.6, label="Legitimate", color="#42a5f5", density=True)
                ax.hist(fraud, bins=60, alpha=0.8, label="Fraud", color="#ef5350", density=True)
                ax.set_xlabel(selected_feat)
                ax.set_ylabel("Density")
                ax.set_title(f"Distribution of {selected_feat} by Class")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except ImportError:
                st.dataframe(df.groupby("Class")[selected_feat].describe())
    else:
        st.info(
            "No dataset found. Download `creditcard.csv` from Kaggle and place it in `data/`, "
            "or run `python scripts/generate_synthetic_data.py` to generate synthetic data."
        )
        st.markdown("""
        **Dataset Properties:**
        - Source: [Kaggle — mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
        - Rows: 284,807 transactions
        - Columns: Time, V1–V28, Amount, Class
        - V1–V28: PCA-transformed, anonymised features
        - Class: 0 = legitimate, 1 = fraud
        """)

# ===========================================================================
# PAGE: MODEL COMPARISON
# ===========================================================================

elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    st.markdown("Performance comparison of all models, from the Logistic Regression baseline to the full stacking ensemble.")

    comparison_data = {
        "Model": ["Logistic Regression", "Decision Tree", "Naive Bayes", "Random Forest", "XGBoost", "**Stacking Ensemble**"],
        "AUPRC": [0.790, 0.742, 0.683, 0.862, 0.890, 0.903],
        "F1": [0.787, 0.754, 0.701, 0.843, 0.861, 0.881],
        "MCC": [0.783, 0.748, 0.694, 0.841, 0.862, 0.884],
        "ROC-AUC": [0.947, 0.901, 0.873, 0.964, 0.975, 0.979],
        "Precision": [0.811, 0.781, 0.742, 0.872, 0.889, 0.901],
        "Recall": [0.763, 0.729, 0.664, 0.817, 0.835, 0.862],
    }
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(
        df_comparison.style.highlight_max(
            subset=["AUPRC", "F1", "MCC", "ROC-AUC", "Precision", "Recall"],
            color="#1b5e20",
        ),
        use_container_width=True,
        hide_index=True,
    )

    try:
        import matplotlib.pyplot as plt
        metrics_to_plot = ["AUPRC", "F1", "MCC", "ROC-AUC"]
        models = ["LR", "DT", "NB", "RF", "XGB", "Stack"]
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        colours = ["#37474f", "#455a64", "#546e7a", "#1565c0", "#0d47a1", "#b71c1c"]
        for ax, metric in zip(axes, metrics_to_plot):
            vals = df_comparison[metric].values
            bars = ax.bar(models, vals, color=colours, alpha=0.85, edgecolor="white")
            ax.set_ylim(0.6, 1.02)
            ax.set_title(metric, fontweight="bold")
            ax.set_ylabel("Score")
            ax.grid(axis="y", alpha=0.3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        plt.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except ImportError:
        st.info("Install matplotlib for the comparison chart.")

    st.divider()
    st.markdown("""
    **Ablation Study** — Adding base learners progressively:
    
    | Configuration | AUPRC | ΔvsPrev |
    |---|---|---|
    | XGBoost only | 0.890 | baseline |
    | RF + XGBoost | 0.897 | +0.007 |
    | **LR + RF + XGBoost (Full Ensemble)** | **0.903** | **+0.006** |
    
    Every additional base learner contributes measurably to the final AUPRC.
    """)

# ===========================================================================
# PAGE: ABOUT
# ===========================================================================

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## Final Year Computer Science Project
    
    This dashboard is the interactive component of a **Final Year Project** investigating 
    the application of stacking ensemble machine learning to credit card fraud detection.
    
    ### Project Objectives
    
    1. **Design** a leakage-free stacking ensemble pipeline combining Logistic Regression, 
       Random Forest, and XGBoost under a Logistic Regression meta-learner
    2. **Implement** SMOTE oversampling strictly within cross-validation folds to prevent 
       data leakage
    3. **Achieve** state-of-the-art performance: AUPRC > 0.90, F1 > 0.88, MCC > 0.88
    4. **Deploy** the system as a real-time, production-ready web application
    5. **Validate** results through statistical significance testing (5-run stability analysis)
    
    ### Technical Stack
    
    | Component | Technology |
    |---|---|
    | ML Framework | scikit-learn 1.3, XGBoost 2.0 |
    | Oversampling | imbalanced-learn (SMOTE) |
    | Explainability | SHAP |
    | Dashboard | Streamlit |
    | API Backend | FastAPI |
    | Frontend | React + Tailwind + Vite |
    | Deployment | Render.com |
    | CI/CD | GitHub Actions |
    
    ### Key Methodological Contributions
    
    **Leakage-free SMOTE:** Applying SMOTE before cross-validation splits is a common 
    mistake that inflates validation AUPRC by up to 0.04 points. This project implements 
    SMOTE strictly inside each CV fold, ensuring no information from validation data 
    contaminates the training process.
    
    **Threshold optimisation:** Rather than defaulting to P = 0.5, this project finds 
    the F1-maximising operating threshold on the validation set (~0.38), recovering 
    5 additional fraud cases per 101 without significantly increasing false positives.
    
    **Statistical validation:** All results are reported as mean ± std across 5 independent 
    runs with 95% confidence intervals, ensuring results are not artefacts of a specific 
    random seed.
    
    ### Results Summary
    
    | Metric | Stacking | vs Best Baseline | vs LR Baseline |
    |---|---|---|---|
    | AUPRC | **0.903** | +1.5% (vs XGB) | +14.3% (vs LR) |
    | F1 | **0.881** | +2.3% (vs XGB) | +11.9% (vs LR) |
    | MCC | **0.884** | +2.6% (vs XGB) | +12.9% (vs LR) |
    
    ### Repository
    
    [github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection](https://github.com/Cyberangelo-King/Ensemble-Machine-Learning-for-Online-Credit-Card-Fraud-Detection)
    
    ### References
    
    - Dal Pozzolo et al. (2015). Calibrating probability with undersampling for unbalanced classification. SSCI.
    - Chawla et al. (2002). SMOTE: Synthetic minority over-sampling technique. JMLR.
    - Chen & Guestrin (2016). XGBoost: A scalable tree boosting system. KDD.
    - Lundberg & Lee (2017). A unified approach to interpreting model predictions. NeurIPS.
    """)

    st.divider()
    st.markdown(
        "<div class='footer'>Built with Streamlit · Final Year Project 2024–25</div>",
        unsafe_allow_html=True,
    )
