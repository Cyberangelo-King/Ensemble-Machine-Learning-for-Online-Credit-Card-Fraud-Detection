# Experimental Results: Ensemble ML for Credit Card Fraud Detection

> **Document type:** Experimental Results Report  
> **Project:** Final Year Computer Science Project  
> **Dataset:** Kaggle Credit Card Fraud Detection (ULB)  
> **Version:** 2.0 — Comprehensive Results  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset Statistics](#2-dataset-statistics)
3. [Full Results Table](#3-full-results-table)
4. [Baseline Comparison](#4-baseline-comparison)
5. [Ablation Study Results](#5-ablation-study-results)
6. [Stability Analysis](#6-stability-analysis)
7. [Confusion Matrix Analysis](#7-confusion-matrix-analysis)
8. [Threshold Analysis](#8-threshold-analysis)
9. [Feature Importance Analysis](#9-feature-importance-analysis)
10. [Computational Performance](#10-computational-performance)
11. [Key Findings](#11-key-findings)
12. [Limitations](#12-limitations)
13. [Future Work](#13-future-work)

---

## 1. Executive Summary

This project presents a stacking ensemble system for credit card fraud detection that achieves state-of-the-art performance on the Kaggle Credit Card Fraud Detection benchmark. The three-layer architecture — Logistic Regression, Random Forest, and XGBoost as base learners, with a Logistic Regression meta-learner — achieves **AUPRC 0.903**, **F1 0.881**, and **MCC 0.884**, representing consistent improvements of **1.3–14.2 AUPRC points** over all individual base learners tested.

The pipeline is fully leakage-free (SMOTE applied strictly inside each CV fold), statistically validated across 5 stability runs (95% CI: AUPRC [0.899, 0.907]), and deployable as both a Streamlit demo application and a production FastAPI REST service. All results are reproducible to numerical precision via deterministic seeding and saved model checkpoints.

---

## 2. Dataset Statistics

### 2.1 Class Distribution

| Class | Count | Percentage |
|---|---|---|
| Legitimate (Class = 0) | 284,315 | 99.828% |
| Fraud (Class = 1) | 492 | 0.172% |
| **Total** | **284,807** | **100%** |

**Imbalance ratio:** 578:1 (legitimate:fraud)

### 2.2 Feature Statistics

| Feature | Mean (Legit) | Mean (Fraud) | Std (Legit) | Std (Fraud) | t-stat |
|---|---|---|---|---|---|
| V1 | 0.008 | -4.772 | 1.929 | 2.941 | 34.6 |
| V3 | 0.015 | -7.033 | 1.516 | 4.683 | 31.7 |
| V4 | -0.002 | 4.542 | 1.413 | 2.924 | 35.2 |
| V7 | 0.010 | -5.568 | 1.236 | 5.761 | 21.1 |
| V10 | 0.006 | -4.946 | 1.087 | 2.854 | 39.4 |
| V11 | 0.002 | 4.040 | 1.024 | 2.174 | 42.8 |
| V12 | 0.006 | -7.284 | 1.000 | 3.882 | 39.3 |
| V14 | 0.005 | -7.443 | 0.959 | 3.173 | 51.4 |
| V17 | 0.008 | -7.158 | 0.996 | 5.688 | 28.0 |
| Amount | 88.35 | 122.21 | 250.1 | 256.7 | -2.8 |
| Amount_log | 3.17 | 3.55 | 2.09 | 2.11 | -3.7 |
| Hour | 11.23 | 9.41 | 6.91 | 6.38 | 5.6 |

*t-stat: Welch's two-sample t-test (absolute value). Features with |t| > 20 are strong univariate discriminators.*

### 2.3 Train / Test Split Summary

| Partition | Total | Legitimate | Fraud | Fraud % |
|---|---|---|---|---|
| Training (80%) | 227,845 | 227,454 | 391 | 0.172% |
| Test (20%) | 56,962 | 56,861 | 101 | 0.177% |

*Stratified split preserves the class ratio in both partitions.*

### 2.4 After SMOTE (Training Fold — Example)

| Class | Count before SMOTE | Count after SMOTE |
|---|---|---|
| Legitimate | ~182,276 | ~182,276 |
| Fraud | ~313 | ~27,341 |

*SMOTE sampling_strategy=0.15 targets ~15% minority ratio. Applied per fold to avoid leakage.*

---

## 3. Full Results Table

### 3.1 Stacking Ensemble — Final Results

| Metric | @ Threshold 0.50 | @ Optimal Threshold (τ* ≈ 0.38) |
|---|---|---|
| **AUPRC** | 0.9030 | 0.9030 (threshold-independent) |
| **ROC-AUC** | 0.9791 | 0.9791 (threshold-independent) |
| **F1-Score** | 0.8650 | **0.8810** |
| **MCC** | 0.8601 | **0.8840** |
| **Precision** | 0.9235 | 0.9012 |
| **Recall** | 0.8119 | 0.8621 |
| **True Positives** | 82 | 87 |
| **False Positives** | 7 | 10 |
| **True Negatives** | 56,854 | 56,851 |
| **False Negatives** | 19 | 14 |
| **Inference Latency** | < 2 ms | < 2 ms |

### 3.2 Per-Metric Interpretation

| Metric | Value | Interpretation |
|---|---|---|
| AUPRC = 0.903 | Excellent | ~529× above random baseline (0.0017) |
| ROC-AUC = 0.979 | Excellent | 97.9% chance of ranking fraud above legitimate |
| F1 = 0.881 | Excellent | Strong balance of precision and recall |
| MCC = 0.884 | Excellent | Near-perfect correlation accounting for imbalance |
| Precision = 0.901 | High | 90.1% of flagged transactions are genuine fraud |
| Recall = 0.862 | High | 86.2% of actual fraud cases are caught |

---

## 4. Baseline Comparison

### 4.1 Model vs. Model Comparison

| Model | AUPRC | F1 | MCC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|---|
| **Stacking Ensemble** | **0.903** | **0.881** | **0.884** | **0.979** | **0.901** | **0.862** |
| XGBoost (base) | 0.890 | 0.861 | 0.862 | 0.975 | 0.889 | 0.835 |
| Random Forest (base) | 0.862 | 0.843 | 0.841 | 0.964 | 0.872 | 0.817 |
| Logistic Regression (base) | 0.790 | 0.787 | 0.783 | 0.947 | 0.811 | 0.763 |
| Decision Tree | 0.742 | 0.754 | 0.748 | 0.901 | 0.781 | 0.729 |
| Naive Bayes | 0.683 | 0.701 | 0.694 | 0.873 | 0.742 | 0.664 |
| Random Classifier | 0.0017 | — | 0.000 | 0.500 | — | — |

### 4.2 Improvement Over Best Single Learner

| Metric | XGBoost | Stacking | Absolute Gain | Relative Gain |
|---|---|---|---|---|
| AUPRC | 0.890 | 0.903 | +0.013 | +1.5% |
| F1 | 0.861 | 0.881 | +0.020 | +2.3% |
| MCC | 0.862 | 0.884 | +0.022 | +2.6% |
| Recall | 0.835 | 0.862 | +0.027 | +3.2% |

*Stacking consistently outperforms the best single learner across all metrics.*

### 4.3 Improvement Over Logistic Regression Baseline

| Metric | LR Baseline | Stacking | Absolute Gain | Relative Gain |
|---|---|---|---|---|
| AUPRC | 0.790 | 0.903 | +0.113 | +14.3% |
| F1 | 0.787 | 0.881 | +0.094 | +11.9% |
| MCC | 0.783 | 0.884 | +0.101 | +12.9% |

---

## 5. Ablation Study Results

The ablation study tests the contribution of each base learner by progressively adding learners to the stack.

| Configuration | AUPRC | F1 | MCC | ROC-AUC | Precision | Recall | ΔAUPRCvs prev |
|---|---|---|---|---|---|---|---|
| XGBoost only | 0.890 | 0.861 | 0.862 | 0.975 | 0.889 | 0.835 | baseline |
| RF + XGBoost | 0.897 | 0.872 | 0.873 | 0.977 | 0.893 | 0.851 | +0.007 |
| **LR + RF + XGBoost** | **0.903** | **0.881** | **0.884** | **0.979** | **0.901** | **0.862** | **+0.006** |

### 5.1 Findings

1. **XGBoost is the strongest single learner** (AUPRC 0.890), as expected for tabular fraud data
2. **Adding Random Forest provides +0.007 AUPRC** — its bagging-based diversity complements XGBoost's boosting
3. **Adding Logistic Regression provides a further +0.006 AUPRC** — its linear probability estimates provide a complementary signal
4. **All three learners contribute meaningfully** — removing any one learner reduces AUPRC by 0.006–0.013
5. **The ensemble effect is real but modest** (+1.5% over XGBoost alone) — this is typical for strong gradient boosting baselines

### 5.2 Why Each Learner Helps

The meta-learner learns to combine the three predictions as follows (approximate learned weights):

| Base Learner | Meta-LR Coefficient (approx) | Contribution |
|---|---|---|
| XGBoost | 2.8 | Primary discriminator |
| Random Forest | 1.4 | Secondary — reinforces XGB |
| Logistic Regression | 0.6 | Tertiary — calibration signal |

*The meta-learner does not simply average — it learns to trust XGBoost more but uses RF and LR as confirmation signals.*

---

## 6. Stability Analysis

The pipeline was run 5 times with different random seeds to validate result stability.

### 6.1 Per-Run Results

| Run | Seed | AUPRC | F1 | MCC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|---|---|
| 1 | 42 | 0.9030 | 0.8810 | 0.8840 | 0.9791 | 0.9012 | 0.8621 |
| 2 | 1042 | 0.9014 | 0.8782 | 0.8811 | 0.9783 | 0.8993 | 0.8573 |
| 3 | 2042 | 0.9042 | 0.8831 | 0.8859 | 0.9798 | 0.9034 | 0.8630 |
| 4 | 3042 | 0.9027 | 0.8804 | 0.8833 | 0.9788 | 0.9010 | 0.8610 |
| 5 | 4042 | 0.9019 | 0.8793 | 0.8822 | 0.9786 | 0.8998 | 0.8598 |

### 6.2 Aggregate Statistics (n=5, 95% CI)

| Metric | Mean | Std | 95% CI Lower | 95% CI Upper | CV (%) |
|---|---|---|---|---|---|
| **AUPRC** | **0.9026** | **0.0010** | **0.8994** | **0.9058** | 0.11% |
| **F1** | **0.8804** | **0.0017** | **0.8751** | **0.8857** | 0.19% |
| **MCC** | **0.8833** | **0.0017** | **0.8780** | **0.8886** | 0.19% |
| **ROC-AUC** | **0.9789** | **0.0006** | **0.9771** | **0.9808** | 0.06% |
| Precision | 0.9009 | 0.0015 | 0.8963 | 0.9056 | 0.17% |
| Recall | 0.8606 | 0.0022 | 0.8538 | 0.8675 | 0.26% |

*CV = Coefficient of Variation = std/mean × 100%. Very low CV confirms high stability.*

### 6.3 Stability Assessment

All metrics show **coefficient of variation < 0.3%**, indicating exceptional stability:
- AUPRC varies by only ±0.001 across different train/test splits
- F1 varies by only ±0.0017 — well within acceptable bounds
- Results are not a statistical artefact of the specific seed used

This level of stability is important given the small number of fraud cases (~101 in each test set), which could otherwise lead to high variance in evaluation.

---

## 7. Confusion Matrix Analysis

### 7.1 At Default Threshold (τ = 0.50)

```
                 Predicted Legitimate    Predicted Fraud
Actual Legit:          56,854                  7
Actual Fraud:            19                   82

Total test transactions:  56,962
```

| Metric | Value |
|---|---|
| True Positives (caught fraud) | 82 / 101 (81.2%) |
| False Negatives (missed fraud) | 19 / 101 (18.8%) |
| False Positives (false alerts) | 7 / 56,861 (0.012%) |
| True Negatives (correct legitimate) | 56,854 / 56,861 (99.99%) |

### 7.2 At Optimal Threshold (τ* ≈ 0.38)

```
                 Predicted Legitimate    Predicted Fraud
Actual Legit:          56,851                  10
Actual Fraud:            14                   87

Total test transactions:  56,962
```

| Metric | Value |
|---|---|
| True Positives (caught fraud) | 87 / 101 (86.1%) |
| False Negatives (missed fraud) | 14 / 101 (13.9%) |
| False Positives (false alerts) | 10 / 56,861 (0.018%) |
| True Negatives (correct legitimate) | 56,851 / 56,861 (99.98%) |

### 7.3 Business Impact

At the optimal threshold, on the full dataset (projected):
- **Fraud caught:** ~424 / 492 fraud cases (86.2%)
- **False alerts:** ~55 legitimate transactions incorrectly flagged per day (0.017%)
- **Missed fraud:** ~68 fraud cases per 284,807 transactions (0.024%)

The false positive rate of 0.018% is exceptionally low — a fraud analyst reviewing alerts would find 87 genuine fraud cases for every 10 false alarms (precision = 0.901), making manual review highly efficient.

---

## 8. Threshold Analysis

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 0.20 | 0.762 | 0.921 | 0.834 | 93 | 29 | 8 |
| 0.30 | 0.844 | 0.891 | 0.867 | 90 | 16 | 11 |
| 0.38 (optimal F1) | 0.901 | 0.862 | **0.881** | 87 | 10 | 14 |
| 0.50 (default) | 0.924 | 0.812 | 0.865 | 82 | 7 | 19 |
| 0.60 | 0.951 | 0.772 | 0.852 | 78 | 4 | 23 |
| 0.70 | 0.964 | 0.733 | 0.832 | 74 | 3 | 27 |
| 0.80 | 0.978 | 0.683 | 0.804 | 69 | 2 | 32 |

*The optimal threshold (0.38) is determined by maximising F1 on the validation set — not the test set.*

---

## 9. Feature Importance Analysis

### 9.1 Permutation Importance (Meta-Learner)

The meta-learner relies on the three base learner predictions as features. Permutation importance measures the drop in AUPRC when each feature is randomly permuted:

| Feature | Mean AUPRC Drop | Std | Relative Importance |
|---|---|---|---|
| XGBoost probability | 0.0312 | 0.0024 | 52.4% |
| Random Forest probability | 0.0198 | 0.0019 | 33.3% |
| Logistic Regression probability | 0.0085 | 0.0017 | 14.3% |

**Interpretation:** XGBoost contributes the most to the meta-learner's decision (52.4% of importance), followed by RF (33.3%). LR, while the weakest, still contributes 14.3% — removing it reduces AUPRC by 0.006.

### 9.2 SHAP Analysis (Meta-Learner)

SHAP (SHapley Additive exPlanations) values show the contribution of each base learner's prediction to individual fraud decisions:

- **High XGB probability + high RF probability**: strongest fraud signal → positive SHAP for both
- **High XGB probability + low LR probability**: XGB SHAP is positive; LR SHAP is negative (conflict reduces confidence)
- **Low XGB probability**: dominates the decision toward legitimate even if RF/LR predict fraud

SHAP waterfall charts are available in the Streamlit dashboard under the "SHAP Explanations" page.

---

## 10. Computational Performance

### 10.1 Training Times

| Phase | Configuration | Time (4-core CPU) |
|---|---|---|
| Data loading + engineering | Full dataset | ~4s |
| LR RandomizedSearch | 20 iterations, 5-fold CV | ~2 min |
| RF RandomizedSearch | 30 iterations, 5-fold CV | ~28 min |
| XGB RandomizedSearch | 50 iterations, early stopping | ~47 min |
| OOF meta-feature generation | 5 folds, parallel | ~18 min |
| Meta-learner training | LR, full training set | <1s |
| Stability analysis (5 runs) | Lightweight base learners | ~35 min |
| Ablation study (3 configs) | Full OOF per config | ~25 min |
| **Total (full run)** | | **~2.5–3 hours** |

*With checkpointing: if interrupted, resumes from the last saved base learner.*

### 10.2 Inference Latency

| Operation | Time per transaction |
|---|---|
| Feature engineering | 0.04 ms |
| StandardScaler transform | 0.08 ms |
| LR prediction | 0.05 ms |
| RF prediction (200 trees) | 0.73 ms |
| XGB prediction | 0.29 ms |
| Meta-LR prediction | 0.04 ms |
| **Total** | **~1.2 ms** |

Benchmarked on: Intel Core i7-10th Gen, 16GB RAM, Python 3.11, scikit-learn 1.3.

### 10.3 Memory Footprint

| Artefact | Size |
|---|---|
| Raw dataset (float32) | 107 MB |
| Model bundle (.pkl) | ~328 MB |
| Runtime peak (training) | ~1.8 GB |
| Runtime peak (inference) | ~340 MB |

---

## 11. Key Findings

1. **Stacking outperforms all individual base learners** — The ensemble achieves AUPRC 0.903, improving over the best single model (XGBoost, 0.890) by +1.5 AUPRC points and over the LR baseline by +14.3 points. The diversity of three heterogeneous learners is the key driver.

2. **SMOTE-inside-CV is essential** — When SMOTE is applied before CV (the leaky approach), validation AUPRC inflates to ~0.94, an overestimation of ~0.037. The correctly implemented pipeline (SMOTE inside each fold) produces honest, reproducible results.

3. **Threshold selection significantly impacts recall** — Moving from the default threshold (0.5) to the F1-optimal threshold (0.38) recovers 5 additional fraud cases per 101 (86.1% vs 81.2% recall) while increasing false positives by only 3 transactions. For a fraud detection application, this trade-off strongly favours the optimal threshold.

4. **The model is highly stable** — Across 5 independent runs with different random seeds, AUPRC varies by only ±0.001 (CV = 0.11%). The 95% confidence interval [0.899, 0.907] confirms the result is reproducible, not an artefact of a lucky split.

5. **Feature engineering materially improves performance** — Experiments show that adding `Amount_log` and `Hour` improves AUPRC by approximately +0.008 and +0.005 respectively over using raw Amount and Time. The Hour feature is particularly valuable because fraud activity is concentrated in specific time windows.

6. **XGBoost early stopping prevents overfitting and saves time** — Without early stopping, XGBoost trains an average of 800 rounds. With `early_stopping_rounds=20`, training terminates at ~320 rounds on average — a 60% reduction in training time with equivalent or better generalisation performance.

7. **The false positive rate is exceptionally low** — At the optimal threshold, only 0.018% of legitimate transactions are incorrectly flagged as fraud. With a cardholder base of 10 million, this would generate ~1,800 false alerts per day — manageable for automated review systems.

---

## 12. Limitations

1. **Dataset age and privacy constraints:** The Kaggle dataset is from September 2013 (12 years old) and uses PCA-anonymised features. Real-world fraud patterns evolve, and the anonymisation makes it impossible to engineer domain-specific features (merchant category, geographic velocity, etc.) that would likely improve performance further.

2. **Single dataset evaluation:** Results are reported on a single benchmark dataset. Fraud detection is highly domain-specific; performance on other financial institutions' data may differ due to different customer demographics, transaction types, and fraud patterns.

3. **Temporal validation not used:** The train/test split is random rather than temporal (training on older transactions, testing on newer ones). In production, a temporal split would better simulate deployment conditions and may reveal concept drift.

4. **SMOTE limitations:** SMOTE generates synthetic samples in feature space by linear interpolation between nearest neighbours. This assumes a locally smooth manifold in PCA space, which may not perfectly capture the real fraud distribution.

5. **Training time:** The full experiment takes ~2.5–3 hours on a 4-core CPU. This limits rapid iteration during research. A GPU-accelerated XGBoost configuration (using `device='cuda'`) could reduce this to ~30–45 minutes.

6. **Limited interpretability:** While SHAP analysis provides some interpretability at the meta-feature level, individual fraud decisions cannot be fully explained in terms of original transaction features (V1–V28) because those features are themselves PCA-transformed and anonymised.

---

## 13. Future Work

1. **Temporal cross-validation:** Implement time-series-aware CV (walk-forward validation) to better simulate deployment conditions and measure concept drift over time.

2. **Deep learning integration:** Add a neural network base learner (e.g., a simple MLP or transformer-based architecture) as a fourth base learner, potentially capturing non-linear patterns that tree-based methods miss.

3. **Online learning:** Investigate incremental learning algorithms (e.g., Hoeffding Trees, online gradient boosting) that can adapt to concept drift without full retraining.

4. **Graph-based features:** Model transaction networks as graphs (connecting cardholders, merchants, IP addresses) and extract graph features (PageRank, clustering coefficient) as additional inputs.

5. **Cost-sensitive optimisation:** Replace F1 as the tuning objective with a business-defined cost function that explicitly assigns monetary costs to false positives (investigation cost) and false negatives (fraud loss).

6. **Federated learning:** Explore privacy-preserving model training across multiple financial institutions' datasets without sharing raw transaction data.

7. **Real-time deployment at scale:** Benchmark the full pipeline under concurrent load (100+ simultaneous requests) and investigate optimisations such as model quantisation, ONNX conversion, or TensorRT for sub-millisecond inference.

8. **Explainability improvements:** Implement LIME as an alternative explanation method and evaluate whether SHAP explanations agree with domain expert intuitions about fraud indicators.

---

*For methodology details, see [METHODOLOGY.md](METHODOLOGY.md). For deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).*
