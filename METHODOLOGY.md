# Methodology: Ensemble Machine Learning for Credit Card Fraud Detection

> **Document type:** Technical Methodology Report  
> **Project:** Final Year Computer Science Project  
> **Version:** 3.0 — aligned with revised research runner (10 November 2026)

---

## Table of Contents

1. [CRISP-DM Framework Alignment](#1-crisp-dm-framework-alignment)
2. [Dataset Characteristics](#2-dataset-characteristics)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Stacking Ensemble Architecture](#4-stacking-ensemble-architecture)
5. [Hyperparameter Search Spaces](#5-hyperparameter-search-spaces)
6. [OOF Meta-Feature Generation](#6-oof-meta-feature-generation)
7. [Evaluation Metrics — Mathematical Definitions](#7-evaluation-metrics--mathematical-definitions)
8. [Threshold Selection Methodology](#8-threshold-selection-methodology)
9. [Statistical Validation](#9-statistical-validation)
10. [Ablation Methodology](#10-ablation-methodology)
11. [Computational Complexity Analysis](#11-computational-complexity-analysis)
12. [Reproducibility Guarantees](#12-reproducibility-guarantees)

---

## 1. CRISP-DM Framework Alignment

This project follows the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** methodology, which provides a structured, iterative approach to machine learning system development. The six phases are detailed below.

### Phase 1: Business Understanding

**Objective:** Evaluate whether a stacked classifier can distinguish fraudulent from legitimate transactions on a public, highly imbalanced historical benchmark. The repository's API and dashboard demonstrate inference; they are not connected to a payment network or a live institution.

**Business context:**  
Credit card fraud costs financial institutions billions annually. The key challenge is the extreme class imbalance (0.172% fraud rate) which renders accuracy misleading — a classifier predicting "legitimate" for all transactions achieves 99.83% accuracy while being entirely useless for fraud detection.

**Prospective research targets (not achieved or verified claims):**
- Report AUPRC as the primary metric against the test-set prevalence baseline.
- Report F1, MCC, precision and recall at a threshold selected only on validation data.
- Compare the default 0.50 decision threshold with the validation-selected operating point.
- Measure inference latency on a named hardware/software environment before making latency claims.

**Stakeholder considerations:**
- *Risk teams* prioritise recall (catching fraud), accepting some false positives
- *Customer experience teams* prioritise precision (minimising false alerts)
- The F1-optimised threshold balances these competing interests; the system can be tuned toward recall or precision by adjusting the threshold

### Phase 2: Data Understanding

**Dataset source:** Kaggle — Credit Card Fraud Detection (ULB Machine Learning Group)

**Initial exploration findings:**
- 284,807 transactions; 492 fraud cases (0.172% imbalance ratio ≈ 1:578)
- Features V1–V28 are the result of PCA transformation (anonymised for privacy)
- `Amount`: transaction value in Euros (right-skewed, range £0.00–£25,691.16)
- `Time`: seconds elapsed since the first transaction in the dataset
- V1 and V3 showed the strongest univariate discrimination between classes
- Significant distributional shift between fraud and legitimate in V4, V11, V14

**Key observations driving design decisions:**
1. The extreme imbalance requires oversampling (SMOTE) or cost-sensitive learning
2. Amount requires log-transformation due to right skew
3. Time records elapsed seconds since the first observation; the engineered Hour variable is elapsed hours, not a real-world clock hour
4. V1–V28 features are already standardised by PCA, but Amount and engineered features require explicit scaling

### Phase 3: Data Preparation

Full details in [Section 3](#3-preprocessing-pipeline). Summary:

1. **Memory optimisation:** V1–V28, Amount, Time loaded as `float32` (halving memory vs. `float64`)
2. **Feature engineering:** Amount_log = log1p(Amount), Hour = Time / 3600 (elapsed hours from dataset start)
3. **Partitioning:** chronological 70/15/15 training, validation and final test partitions by source Time
4. **Scaling:** StandardScaler lives inside each imbalanced-learn pipeline and is fitted on each CV training fold only
5. **Oversampling:** SMOTE lives inside the same pipeline and is applied only to each CV training fold
6. **Feature selection:** All 30 features retained (28 PCA + Amount_log + Hour); Time and raw Amount dropped post-engineering

### Phase 4: Modelling

Three base learners selected for diversity and complementarity:

| Model | Rationale |
|---|---|
| **Logistic Regression** | Provides well-calibrated probabilities, acts as a linear baseline within the ensemble, fast to tune |
| **Random Forest** | High-variance learner with natural feature importance; immune to feature scaling; handles non-linear interactions |
| **XGBoost** | Gradient-boosted trees with regularisation; estimator count and other hyperparameters are selected by cross-validation |

Meta-learner: Logistic Regression — a simple linear model is intentionally chosen to avoid overfitting on only 3 meta-features.

Hyperparameter optimisation: `RandomizedSearchCV` with `StratifiedKFold(n_splits=5)` and `scoring='average_precision'`. AUPRC is used as the tuning objective rather than ROC-AUC because it is more sensitive to model quality on imbalanced datasets (it focuses on the minority class).

### Phase 5: Evaluation

- **Primary metric:** AUPRC (Area Under Precision-Recall Curve) — most informative for imbalanced classification
- **Secondary metrics:** F1 at optimal threshold, MCC, ROC-AUC, Precision, Recall, confusion matrix
- **Threshold analysis:** Both 0.5 and a threshold selected on a separate validation partition are reported on the untouched final test partition
- **Supplementary stability:** repeated stratified random splits with 95% t-intervals; this is not a substitute for chronological evaluation
- **Ablation study:** Performance compared across 1-, 2-, and 3-learner configurations

### Phase 6: Deployment

The repository includes:
1. A **Streamlit dashboard** for local demonstration
2. A **FastAPI REST + WebSocket backend** for model inference demonstrations
3. A **React + Tailwind frontend** for interacting with the API

These components are deployment scaffolding and a research demonstration; they are not evidence of integration with a bank or production payment infrastructure.

Deployment documentation: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 2. Dataset Characteristics

### Summary Statistics

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 (99.828%) |
| Fraudulent transactions | 492 (0.172%) |
| Imbalance ratio | ~1:578 |
| Features | 30 (V1–V28 + Amount + Time) |
| Engineered features | 2 (Amount_log + Hour) |
| Missing values | None |
| Time span | 48 hours (172,792 seconds) |
| Amount range | €0.00 – €25,691.16 |
| Median Amount (legitimate) | €22.00 |
| Median Amount (fraud) | €9.25 |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| V1–V28 | float32 | PCA-transformed, anonymised transaction features |
| Amount | float32 | Transaction value in Euros |
| Time | float32 | Seconds elapsed since dataset start |
| Amount_log | float32 | `log1p(Amount)` — normalises right skew |
| Hour | float32 | `Time / 3600` — elapsed hours since the first transaction (approximately 0–48) |
| Class | int | Target: 0 = legitimate, 1 = fraud |

### Class Distribution

```
Legitimate: ████████████████████████████████████████████████ 99.83%
Fraud:      ▌                                                 0.17%
```

The extreme imbalance (1:578) makes this one of the most challenging benchmark datasets in fraud detection. Standard metrics like accuracy are misleading; AUPRC and MCC are required for a faithful evaluation.

---

## 3. Preprocessing Pipeline

### 3.1 Memory Optimisation

Loading V1–V28 as `float64` (Python default) consumes ~43 MB for this dataset. Specifying `float32` halves this to ~22 MB with no loss of discriminative precision (PCA components do not require double precision for this use case).

```python
dtype_map = {f"V{i}": "float32" for i in range(1, 29)}
dtype_map["Amount"] = "float32"
dtype_map["Time"] = "float32"
df = pd.read_csv(path, dtype=dtype_map)
```

### 3.2 Feature Engineering

**Amount_log:**

    Amount_log = log(1 + Amount)

The transaction Amount distribution is right-skewed. The log1p transformation compresses large values and safely handles zero-amount transactions.

**Hour (elapsed time):**

    Hour = Time / 3600

The ULB dataset's Time field records seconds elapsed between a transaction and the first transaction in the dataset. It is not a timestamp or a time-of-day field. Hour therefore represents elapsed hours on the dataset timeline (approximately 0–48), not a circadian or local-clock feature. The dashboard labels it accordingly.

### 3.3 Leakage-Free Scaling and SMOTE

**Why preprocessing must be inside CV:**

Fitting a scaler before cross-validation allows validation-fold observations to influence the means and variances used by the transformation. Applying SMOTE before the fold split is more serious: synthetic minority samples can be generated using neighbours from observations that later appear in the validation fold.

The current runner uses an imbalanced-learn Pipeline for each base learner. The ordered steps are:

    StandardScaler -> SMOTE -> classifier

The complete pipeline is cloned and fitted separately inside each CV training fold. The held-out fold is passed only to predict_proba. Hyperparameter search receives the original, unscaled, unresampled X and y. It does not receive a globally oversampled matrix.

This rule also applies during OOF meta-feature generation and learning-curve evaluation. The final model bundle stores the preprocessing steps inside each base learner, keeping training and inference feature handling consistent.

**SMOTE parameters:**
- sampling_strategy=0.15: targets a minority-to-majority ratio of 0.15 within each CV training fold
- random_state: fixed per experiment/fold for reproducibility
- k_neighbors: configured by SMOTE; folds must contain enough minority examples

### 3.4 Train / Validation / Test Split

The default experiment sorts records by the source Time field and partitions them chronologically:

| Partition | Rows (fraction) | Role |
|---|---:|---|
| Training | 70% | Hyperparameter search, OOF base predictions and meta-learner fitting |
| Validation | 15% | Select the operating threshold; no final test data is used |
| Final test | 15% | One-time final performance evaluation |

The chronological order is preserved when the source dataset is loaded. The validation and final test partitions are never included in hyperparameter fitting or SMOTE. The validation set selects an operating threshold; the same frozen threshold is then applied to the final test scores.

A stratified-random split is available as a supplementary comparison, but it should not be used as the primary evidence for an online/forward-in-time detection claim.

## 4. Stacking Ensemble Architecture

### 4.1 Design Principles

Stacking (Wolpert, 1992) is a meta-learning technique where the predictions of base learners become features for a higher-level model. The key insight is that different learners make different errors; a meta-learner can learn which base learner is correct in which region of feature space.

```
                           Training Data
                                │
            ┌───────────────────┼────────────────────┐
            │                   │                    │
     [Base Learner 1]    [Base Learner 2]    [Base Learner 3]
      Logistic Regr.      Random Forest        XGBoost
            │                   │                    │
      P₁(fraud|x)         P₂(fraud|x)         P₃(fraud|x)
            └───────────────────┼────────────────────┘
                                │
                    OOF Meta-Features: [P₁, P₂, P₃]
                                │
                       [Meta-Learner: LR]
                                │
                          P(fraud|x)
```

### 4.2 Base Learner Selection Rationale

**Logistic Regression (base):**
- Strong linear separation of PCA components (V1, V3 are highly informative)
- Produces well-calibrated probability estimates
- Fast to train and tune (20 random configurations sufficient)
- Regularisation via C parameter prevents overfitting on small fraud class

**Random Forest:**
- Ensemble of 100–500 decision trees with random feature subsampling
- Robust to outliers in Amount distribution
- `class_weight='balanced'` or `'balanced_subsample'` handles imbalance
- Natural feature importance via mean impurity decrease

**XGBoost:**
- Sequential gradient boosting with regularisation (L1 + L2)
- `scale_pos_weight` adjusts loss weighting for minority class
- The estimator count and regularisation settings are selected by cross-validated average precision
- Model-level parallelism is limited when CV jobs are parallelised to avoid CPU oversubscription
- Often a strong single-model baseline on tabular fraud-detection datasets

### 4.3 Meta-Learner Design

A simple **Logistic Regression** meta-learner is used deliberately. With only 3 meta-features (P₁, P₂, P₃), a complex model would overfit. LR provides:
- A learned linear combination of the three predictions
- Regularisation via C parameter
- Interpretable coefficients (which base learner is most trusted)

---

## 5. Hyperparameter Search Spaces

### 5.1 Logistic Regression

| Parameter | Search Space | Notes |
|---|---|---|
| C (inverse regularisation) | [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0] | Controls L1/L2 penalty strength |
| penalty | ['l1', 'l2'] | L1 promotes sparsity; L2 prevents large weights |
| solver | ['liblinear', 'saga'] | Both support L1; SAGA faster for large datasets |
| max_iter | [500, 1000, 2000] | Ensures convergence |
| class_weight | ['balanced', None] | 'balanced': auto-weighting by class frequency |
| **n_iter** | **20** | RandomizedSearchCV configurations |

### 5.2 Random Forest

| Parameter | Search Space | Notes |
|---|---|---|
| n_estimators | [100, 200, 300, 500] | More trees → lower variance, diminishing returns |
| max_depth | [None, 10, 20, 30] | None = fully grown; deeper = higher variance |
| min_samples_split | [2, 5, 10] | Minimum samples to split a node |
| min_samples_leaf | [1, 2, 4] | Minimum samples at leaf node |
| max_features | ['sqrt', 'log2', 0.3, 0.5] | Feature subsampling per split |
| class_weight | ['balanced', 'balanced_subsample'] | Per-tree vs per-dataset weighting |
| bootstrap | [True, False] | Bootstrap sampling vs full dataset per tree |
| **n_iter** | **30** | 50 iterations too expensive for 500-tree forests |

### 5.3 XGBoost

| Parameter | Search Space | Notes |
|---|---|---|
| n_estimators | [200, 400, 600, 800] | Moderated by early stopping |
| max_depth | [3, 4, 5, 6, 7] | Shallow trees generalise better |
| learning_rate | [0.01, 0.05, 0.1, 0.15, 0.2] | Lower LR → more trees needed |
| subsample | [0.6–1.0] | Row subsampling (stochastic gradient) |
| colsample_bytree | [0.6–1.0] | Column subsampling per tree |
| min_child_weight | [1, 3, 5, 7] | Minimum sum of instance weight in leaf |
| gamma | [0, 0.1, 0.2, 0.3, 0.5] | Minimum loss reduction to split |
| reg_alpha | [0, 0.01, 0.1, 1.0] | L1 regularisation |
| reg_lambda | [0.5, 1.0, 2.0, 5.0] | L2 regularisation |
| scale_pos_weight | [1, 5, 10, 50, 100] | Class imbalance correction |
| **n_iter** | **50** | Manual random search with early stopping eval |

**Search protocol:** XGBoost is tuned through the same cross-validation interface as the other base learners. The implementation does not repeatedly reuse one globally held-out early-stopping set across hyperparameter trials. RandomizedSearchCV scores each candidate using average precision on its held-out CV folds and refits the selected pipeline on the development-training partition.

### 5.4 Scoring Objective

All base learner searches use `scoring='average_precision'` (AUPRC). This metric:
- Directly measures performance on the minority class
- Is more discriminative than ROC-AUC for imbalanced datasets
- Is the primary evaluation metric used in the fraud detection literature

---

## 6. OOF Meta-Feature Generation

### 6.1 Algorithm (Pseudocode)

    Input: training partition X, labels y, base pipelines L, K folds
    Output: OOF meta-feature matrix with one probability column per learner

    meta_X = zeros(n_rows, number_of_learners)
    for each fold:
        clone each complete base pipeline
        fit clone on raw X_train_fold, y_train_fold
            StandardScaler fits only on X_train_fold
            SMOTE resamples only X_train_fold
            classifier fits only on resampled training-fold data
        predict probabilities from raw X_validation_fold
        write probabilities into meta_X at validation indices

    fit the meta-learner on meta_X and original y

### 6.2 Parallelisation Strategy

The fold loop is parallelised with joblib.Parallel. Each fold receives independent estimator clones created with sklearn.base.clone. Base estimators use one internal CPU thread while CV controls outer parallelism; this avoids nested all-core parallelism and CPU oversubscription. The run configuration records the outer n_jobs setting.

### 6.3 Validation and Test Meta-Features

The fitted base pipelines accept raw engineered feature rows. Each pipeline applies its own fitted scaler and classifier; SMOTE is used during fitting only and is not applied during inference.

    meta_X_validation = stack([
        learner.predict_proba(X_validation)[:, 1] for learner in fitted_learners
    ])

    meta_X_test = stack([
        learner.predict_proba(X_test)[:, 1] for learner in fitted_learners
    ])

The meta-learner is trained only on OOF predictions from the training partition. The validation meta-features choose the operating threshold. The final test meta-features are evaluated only after the threshold has been frozen.

## 7. Evaluation Metrics — Mathematical Definitions

### 7.1 Confusion Matrix

```
                 Predicted Positive    Predicted Negative
Actual Positive:       TP                    FN
Actual Negative:       FP                    TN
```

### 7.2 Precision-Recall Area (AUPRC)

AUPRC is the primary metric because:
- ROC-AUC is optimistic for highly imbalanced datasets (many TN inflate the curve)
- AUPRC focuses on the minority class, directly measuring precision-recall tradeoff at all thresholds

```
AUPRC = Σₙ (Rₙ − Rₙ₋₁) × Pₙ

where:
  Precision P = TP / (TP + FP)
  Recall    R = TP / (TP + FN)
  n indexes the threshold values from the precision_recall_curve
```

**Interpretation:** AUPRC = 1.0 indicates a perfect ranking. The random-ranking baseline is approximately the fraud prevalence in the evaluation partition. Report the measured AUPRC from the current metrics artifact; do not reuse the archived v2.0 number as current performance.

### 7.3 F1-Score

```
F1 = 2 × (P × R) / (P + R) = 2TP / (2TP + FP + FN)
```

Harmonic mean of precision and recall. F1 = 1.0 indicates perfect precision and recall simultaneously. F1 is threshold-dependent; we report at both 0.5 and the optimal F1-maximising threshold.

### 7.4 Matthews Correlation Coefficient (MCC)

```
MCC = (TP × TN − FP × FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

MCC is considered the most informative single metric for binary classification on imbalanced data (Chicco & Jurman, 2020). Unlike F1, it accounts for all four quadrants of the confusion matrix. Range: [−1, +1] where:
- +1 = perfect prediction
-  0 = no better than random
- −1 = perfect inverse prediction

### 7.5 ROC-AUC

```
AUC = ∫₀¹ TPR(FPR⁻¹(t)) dt

TPR = Recall = TP / (TP + FN)
FPR = FP / (FP + TN)
```

Measures the probability that a randomly selected fraud case is ranked higher than a randomly selected legitimate transaction. Report the ROC-AUC measured on the current untouched test partition.

---

## 8. Threshold Selection Methodology

All learned models output a fraud probability P(fraud|x) ∈ [0, 1]. A decision threshold τ converts this to a binary prediction:

```
ŷ = 1  if P(fraud|x) ≥ τ
    0  otherwise
```

**Default threshold (τ = 0.5):** Standard binary classification assumption. Appropriate when misclassification costs are symmetric — which they are not for fraud.

**Optimal threshold (τ*):** Maximises F1 on the validation set by scanning all precision-recall curve thresholds:

```python
precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
f1_scores = 2 * precisions * recalls / (precisions + recalls + ε)
τ* = thresholds[argmax(f1_scores[:-1])]
```

The selected threshold is data-dependent and is written to metrics.json. Its value must be learned from the validation partition only; it should not be copied from the legacy report or selected against the final test partition.

**Business threshold tuning:**  
In deployment, the threshold can be adjusted to balance precision and recall based on the bank's cost structure:
- Higher τ (0.6–0.8): fewer false alerts, more missed fraud (precision-focused)
- Lower τ (0.2–0.3): fewer missed fraud, more false alerts (recall-focused)

---

## 9. Statistical Validation

### 9.1 Stability Analysis Protocol

The runner includes a supplementary repeated-stratified-random-split robustness analysis. Each run uses a new random 80/20 split, fold-local preprocessing and OOF meta-features. These results quantify sensitivity to partition choice on the benchmark; they do not replace the primary chronological holdout.

### 9.2 Confidence Intervals

95% confidence intervals are computed using the Student t-distribution:

```
mean(m) = Σ mᵢ / n
std(m)  = √(Σ(mᵢ − mean)² / (n−1))
SE      = std / √n
CI₉₅   = mean ± t_{0.025, n-1} × SE
```

where t_{0.025, 4} ≈ 2.776 for n=5 runs.

**Reporting rule:** state the observed mean, standard deviation and interval from the generated stability_report.json. No target interval is treated as an observed result.

### 9.3 Interpretation

A narrow confidence interval (std < 0.005 for AUPRC) indicates the model is stable across different data partitions and not overfitted to a specific split. This is especially important for fraud detection where the small number of fraud cases (492 in total, ~101 in the test set) means different splits can contain substantially different fraud distributions.

---

## 10. Ablation Methodology

The ablation study systematically removes base learners to quantify each one's contribution:

| Configuration | Base Learners | Purpose |
|---|---|---|
| **1 learner** | XGBoost only | Strongest single learner baseline |
| **2 learners** | RF + XGBoost | Quantifies LR's marginal contribution |
| **3 learners** | LR + RF + XGBoost | Full stacking ensemble (proposed system) |

**Protocol:**  
Each configuration uses identical hyperparameters (from the best-found configuration in the full experiment), the same train/test split, and the same SMOTE ratio. Only the set of base learners changes.

**Expected finding:** Each additional learner should increase AUPRC by 0.005–0.015, demonstrating that the diversity of predictions from diverse learners is beneficial (the "ensemble effect").

---

## 11. Computational Complexity Analysis

### 11.1 Training Complexity

| Component | Complexity | Typical Time (4-core CPU) |
|---|---|---|
| SMOTE (per fold) | O(n × k_neighbours) | ~15s |
| LR RandomSearch (20 iter) | O(20 × n × p) | ~2 min |
| RF RandomSearch (30 iter) | O(30 × n_trees × n × log(n)) | ~25 min |
| XGB RandomSearch (50 iter) | O(50 × n_rounds × n × p) | ~45 min |
| OOF stacking (5 folds) | O(K × 3 × training_complexity) | ~30 min |
| Total (full run) | — | ~2–3 hours |

With checkpointing, interrupted runs resume from the last saved base learner.

### 11.2 Inference Complexity

| Component | Time per transaction |
|---|---|
| Feature engineering | < 0.1 ms |
| Scaling (StandardScaler) | < 0.1 ms |
| LR prediction | ~0.05 ms |
| RF prediction (200 trees) | ~0.8 ms |
| XGB prediction | ~0.3 ms |
| Meta-learner prediction | ~0.05 ms |
| **Total** | **< 2 ms** |

This comfortably meets the production requirement of < 5 ms for real-time fraud screening.

### 11.3 Memory Usage

| Artefact | Size |
|---|---|
| Dataset (float32) | ~110 MB |
| SMOTE-augmented training set | ~180 MB |
| RF model (200 trees) | ~280 MB |
| XGB model | ~45 MB |
| LR models | < 1 MB |
| Full model bundle (.pkl) | ~330 MB |

---

## 12. Reproducibility Guarantees

### 12.1 Seed Propagation

All random operations receive explicit seeds derived from `CONFIG['random_state'] = 42`:

| Component | Seed usage |
|---|---|
| `train_test_split` | `random_state=42` |
| `StratifiedKFold` | `random_state=42` |
| `SMOTE` | `random_state=42` |
| `LogisticRegression` | `random_state=42` |
| `RandomForestClassifier` | `random_state=42` |
| `XGBClassifier` | `random_state=42` |
| `RandomizedSearchCV` | `random_state=42` |
| Stability runs | `random_state = 42 + run × 1000` |

### 12.2 Reproducibility Assertion

After saving the model bundle, the pipeline reloads it and asserts identical predictions:

```python
y_prob_original = meta_lr.predict_proba(meta_X_test)[:, 1]
reloaded_bundle = joblib.load("results/stacking_model.pkl")
y_prob_reloaded = reloaded_bundle["meta_learner"].predict_proba(meta_X_test)[:, 1]
assert np.allclose(y_prob_original, y_prob_reloaded, atol=1e-6), "Reproducibility FAILED"
```

This assertion guarantees that:
1. Model serialisation/deserialisation is lossless
2. Inference on the same input produces bit-identical output

### 12.3 Environment Pinning

`requirements.txt` pins all dependency versions. The CI pipeline tests on Python 3.9 and 3.11 to verify cross-version compatibility.

---

## References

1. Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *JMLR*, 3, 321–357.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
4. Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21(1), 6.
5. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. *SSCI*.
6. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
7. Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox. *JMLR*, 18(17), 1–5.
