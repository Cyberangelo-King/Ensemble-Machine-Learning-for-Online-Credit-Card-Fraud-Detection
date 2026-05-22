# Methodology

## CRISP-DM Alignment
1. Business understanding: reduce fraud loss with real-time constraints.
2. Data understanding: highly imbalanced transaction records.
3. Data preparation: split, clean, and leakage-safe balancing.
4. Modeling: tuned stacking ensemble.
5. Evaluation: AUPRC, F1, MCC, confusion-focused review.
6. Deployment readiness: deterministic pipeline and latency profiling.

## Preprocessing and Class Imbalance
- Input features include PCA-like `V1`–`V28`, `Time`, `Amount`.
- `Class` is the target.
- Stratified split preserves minority prevalence.
- SMOTE is applied to **training partitions only** with sampling ratio `0.1` (1:10).

## Cross-Validation and Hyperparameter Search
- 5-fold `StratifiedKFold` with shuffle and fixed seed.
- `RandomizedSearchCV` with **50** configurations per base learner.
- Optimization metric: average precision (AUPRC proxy).

## Model Architecture Rationale
- Logistic Regression: stable linear boundary baseline.
- Random Forest: non-linear and interaction-aware ensemble baseline.
- XGBoost: high-capacity gradient boosting for complex fraud signatures.
- Stacking meta-learner (Logistic Regression) combines complementary error patterns.

## Evaluation Metric Selection
- **AUPRC**: robust under heavy class imbalance.
- **F1-score**: balances precision/recall at thresholded decisions.
- **MCC**: correlation-style metric robust to skewed class distribution.
- Accuracy is deprioritized due to majority-class bias in fraud settings.
