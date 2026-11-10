# Research and Model Integrity Audit

**Audit date:** 10 November 2026  
**Scope:** Research runner, stacking protocol, model artifacts, dashboard/API behavior, documentation and CI  
**Status:** Hardened research prototype — not production fraud infrastructure

## 1. Executive assessment

The repository now has a clearer separation between the scientific experiment and the demonstration application. The central modeling change is to fit scaling and SMOTE inside each training fold, rather than transforming or oversampling data before cross-validation. The revised research runner also separates base-model hyperparameter selection from the observations used to train the stacking meta-learner.

The experiment must still be run on the real ULB/Kaggle CSV before any performance claim can be made about the revised pipeline. Historical v2.0 scores are retained in `RESULTS.md` for traceability but are not verified results for this code.

## 2. Dataset contract

The expected input is the public ULB credit-card fraud dataset:

- Features: `V1`–`V28`, `Amount`, `Time`, and binary target `Class`.
- `V1`–`V28` are anonymised PCA components.
- `Time` is elapsed seconds from the first recorded transaction, not a wall-clock timestamp.
- Feature engineering produces `Amount_log = log1p(Amount)` and `Hour = Time / 3600`. Here, `Hour` means elapsed hours in the source sample, not local time-of-day.
- The well-known dataset contains 284,807 transactions and 492 fraud examples (about 0.172%). These are dataset-level facts, not the class counts in every chronological partition.

Source: [Kaggle — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The data loader checks required columns, missing values, binary labels, non-negative `Amount`/`Time`, then sorts by `Time`. The temporal split requires this verified ordering marker.

## 3. Model and preprocessing contract

The current stack contains:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Logistic Regression meta-learner

Every base estimator is an imbalanced-learn pipeline:

```
StandardScaler → SMOTE → classifier
```

The pipeline is cloned and fitted inside each cross-validation fold. The held-out fold is used only for prediction. SMOTE therefore cannot generate synthetic neighbours using observations from the fold it is meant to validate. The same fold-local pattern is used for OOF meta-features and learning curves.

The model bundle carries `feature_schema_version = ulb-pca-elapsed-hours-v1`. The dashboard and API reject artifacts that do not declare the current feature contract, avoiding accidental use of older models trained with a different `Time` transformation.

## 4. Stacking and evaluation protocol

The default research evaluation is chronological:

| Partition | Share | Purpose |
|---|---:|---|
| Training | 70% | Model development |
| Validation | 15% | Select the operating threshold |
| Final test | 15% | One-time final evaluation |

Inside the training partition:

- A separate tuning subset selects base-model hyperparameters using training-only cross-validation and average precision.
- A disjoint stacking subset receives out-of-fold predictions. The tuning subset may be included in each fold's training data, but its labels are not part of the OOF rows used to fit the meta-learner.
- The meta-learner is fit on the OOF stacking rows.
- The selected base pipelines are then refit on the complete training partition.
- The operating threshold is selected on validation data and frozen before final test evaluation.

The runner also supports a stratified-random split for supplementary comparisons. It must not be substituted for the default chronological evaluation when making forward-in-time claims.

## 5. Metrics and interpretation

For this imbalanced problem:

- **Average precision / AUPRC** is the primary ranking metric. Always show the positive-class prevalence as the no-skill reference for the evaluated partition.
- ROC-AUC is supplementary and can look strong even when precision is poor at a low fraud prevalence.
- Precision, recall, F1 and MCC depend on the chosen operating threshold.
- The threshold in the final report is learned on validation data only.
- Reliability diagrams are diagnostics; they do not mean that the model has been calibrated.
- No historical score, synthetic-data score or target value should be reported as current benchmark performance.

The repeated stratified-split report is a descriptive sensitivity analysis. Its random test partitions overlap, so its t-based intervals are not formal population confidence intervals and must not be used as a significance claim.

## 6. Artifact and reproducibility controls

- Research model bundle and metrics report use explicit schema/provenance fields.
- The artifact feature schema is checked at application startup.
- Hyperparameter checkpoints contain a SHA-256 fingerprint derived from the tuning data and relevant search configuration. Legacy or mismatched checkpoints are ignored.
- The serialized model is reloaded and its predictions are compared with pre-serialization outputs.
- Synthetic demo reports are marked `benchmark_status = synthetic_demo`; benchmark reports are marked `experimental_benchmark`.
- The Streamlit dashboard rejects metrics reports without a recognised provenance marker.
- CI builds a model using generated synthetic data, runs API smoke tests and runs unit tests. **This validates the software path, not the real Kaggle experiment.**

## 7. Current limitations — do not overclaim

1. The ULB dataset spans roughly two days. It cannot establish performance across months, institutions, regions or evolving fraud tactics.
2. The benchmark does not provide customer/cardholder identity fields suitable for customer-level or account-level generalisation tests.
3. No independent external dataset has been evaluated with the revised runner.
4. The WebSocket endpoint replays sample rows. It is not connected to a bank, issuer, acquirer, payment processor or live card network.
5. The Streamlit simulator accepts manually supplied features. It is a demonstration, not an operational decisioning workflow.
6. Probability calibration, issuer-specific cost modelling, human review policies, dispute outcomes and false-decline costs are not yet evaluated.
7. No verified inference latency, throughput, memory, availability or production SLA has been measured on a named deployment environment.
8. The prototype has not undergone independent security, privacy, resilience or regulatory review.

## 8. Recommended next research stages

### Stage A — establish a defensible benchmark
- Run the current experiment on the real CSV.
- Record the dataset checksum, partition counts, fraud prevalence per partition, package versions, run configuration and generated artifact hashes.
- Compare the stack with DummyClassifier, Logistic Regression, Random Forest and XGBoost on the exact same final chronological test partition.
- Report confusion matrices and precision/recall operating points alongside AUPRC.

### Stage B — test whether the ensemble is worth its complexity
- Compare no resampling, class weighting, random under-sampling, SMOTE and model-specific imbalance controls using the same split.
- Add ablations for each base learner and combinations.
- Compare the stack with a tuned single XGBoost model; retain the stack only if the improvement is repeatable and operationally worthwhile.
- Use multiple forward-chaining evaluation windows where data volume permits.

### Stage C — make probabilities and decisions operationally meaningful
- Fit and evaluate calibration methods without using the final test partition.
- Select thresholds from explicit expected-cost or capacity constraints, not F1 alone.
- Measure recall at a capped false-positive rate and precision at the daily alert-review capacity.
- Report uncertainty and confidence limitations honestly.

### Stage D — deployment research
- Benchmark warmed single-row and batch inference (p50/p95/p99), throughput, memory and cold start.
- Add monitored data contracts, schema validation, drift and data-quality alerts.
- Replace the sample WebSocket replay with a separately tested event-ingestion architecture before describing the system as online.
- Add durable audit logs, idempotency, model/version rollback, alert review and secure service-to-service authentication.
- Conduct independent security and privacy testing before handling real payment data.

---

**Bottom line:** The revised pipeline is materially safer to evaluate and more reproducible, but it is not yet proven to outperform the baselines on the current temporal holdout, and it is not a production-ready fraud-control system. The next evidence-producing step is a full run against the real dataset.
