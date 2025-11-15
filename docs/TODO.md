# Project TODOs – High-Dimensional Fault & Anomaly Detection

> Track progress across the pipeline. Use checkboxes for task status and nest subtasks when helpful.

## 1. Project Management & Environment
- [ ] Finalize scope with stakeholders (feature coverage, latency targets, alert outputs).
- [ ] Define success metrics (F1, detection delay, throughput) and acceptance thresholds.
- [ ] Set up Python env (3.10+) with TensorFlow/Keras, PyTorch (optional), scikit-learn, pandas, numpy, matplotlib, seaborn, plotly.
- [ ] Configure linting/formatting (black, isort, flake8) and pre-commit hooks.
- [ ] Establish experiment tracking (Weights & Biases, MLflow, or lightweight logging).
- [x] Create template config files in `configs/` for data prep, PCA, autoencoder, CNN.

## 2. Data Acquisition, Labeling & Management
- [ ] Inventory available smart grid datasets (>20 features) and document schema in `docs/`.
- [ ] Implement ingestion scripts (`src/smart_grid_fault_detection/data_prep/ingest.py`) with checksum validation.
  - [x] Add manifest template + ingestion CLI scaffold (parquet + CSV support).
- [ ] Integrate video benchmark converters (e.g., PED2) into the data-prep set.
  - [x] Added `ped2_converter.py` to emit tabular features from raw frames.
- [x] Build cleaning pipeline: timestamp alignment, missing data imputation, outlier clipping.
  - Added `regularize_time_index`, `impute_columns`, and `clip_zscore_outliers` helpers.
- [ ] Design synthetic fault generator (spikes, dropouts, cyber anomalies) with parameter controls.
  - [x] Introduced `faults.py` (PED2-informed augmentation) plus synthetic CSV generation script.
- [ ] Define labeling strategy (Normal vs Fault) and export balanced splits.
  - [x] Synthetic dataset + augmentation pipeline emit `fault_flag` / `fault_type`.
- [ ] Version datasets (DVC or lightweight manifest) to ensure reproducibility.

## 3. Exploratory Data Analysis & Baselines
- [x] Create `notebooks/eda/01_data_overview.ipynb` for descriptive stats, correlation heatmaps, and processed-data comparisons.
- [ ] Visualize temporal patterns (daily/weekly seasonality, load-generation coupling).
- [x] Implement sliding-window z-score detector baseline in `src/smart_grid_fault_detection/detectors/zscore.py`.
- [x] Benchmark baseline accuracy/latency and log results in `reports/`.
  - [x] Sweep z-score windows/thresholds and store metrics in `reports/zscore_sweep`.
  - [x] Record runtime stats via `zscore_benchmark.py` (`reports/zscore_benchmark`).

## 4. Dimensionality Reduction Experiments
- [x] Implement PCA utilities (`smart_grid_fault_detection/models/pca_pipeline.py`) with variance stats + projections saved to disk.
- [x] Add t-SNE/UMAP visualization scripts for latent structure exploration.
- [x] Prototype LSTM autoencoder architecture search (encoder depth, bottleneck size, dropout).
- [x] Compare reconstruction error distributions between normal and fault windows (feature/time heatmaps).
- [x] Evaluate compression ratios vs detection quality trade-offs.

## 5. Modeling & Detection
### 5.1 PCA + Classical Detectors
- [x] Train PCA-reduced feature set feeding into Isolation Forest / One-Class SVM (`pca_detectors.py`).
- [x] Optimize thresholds and number of retained components per detector (`pca_sweep.py` summary).

### 5.2 LSTM Autoencoder Anomaly Detection
- [x] Implement training loop with teacher forcing, early stopping, learning-rate scheduling.
- [x] Calibrate anomaly score thresholds using validation ROC/PR curves.
- [ ] Add capability for multivariate reconstruction error heatmaps.

### 5.3 CNN Fault-Type Classifier
- [x] Architect CNN (1D) for classifying spikes vs dropouts vs cyber anomalies.
- [x] Augment dataset with labeled subclasses and evaluate confusion matrix. *(stratified split + predictions/CM generated)*
- [ ] Explore hybrid models (latent features from autoencoder feeding CNN head).

- [ ] Standardize train/val/test splits with stratification over fault types.
- [x] Build evaluation harness comparing detectors (`detectors/eval_compare.py` summary CSV).
- [ ] Run rare-event stress tests with controlled anomaly injection.
- [ ] Document interpretability insights (feature contributions, latent direction analysis).
- [ ] Compare full-dimensional vs reduced-dimensional performance in tabular + narrative form.

## 7. Visualization & Reporting
- [ ] Implement plotting utilities for anomaly timelines, latent space projections, feature importances.
- [ ] Generate automated report notebook (`notebooks/eda/99_summary.ipynb`) that compiles metrics + figures.
- [ ] Store final figures in `reports/figures/` with captions and metadata.

## 8. Integration & Deployment
- [ ] Define API contract for ALDC integration (inputs, outputs, alert schema).
- [ ] Package models into a lightweight service (FastAPI/Flask) or edge-runtime script.
- [ ] Add monitoring hooks (Kafka/MQTT subscribers, logging, heartbeat checks).
- [ ] Draft runbook for operators (startup/shutdown, troubleshooting, retraining cadence).

## 9. Documentation, QA & Future Work
- [ ] Maintain `README.md` and architecture diagrams in `docs/`.
- [ ] Write unit/integration tests covering data loaders, preprocessors, and models.
- [ ] Set up CI pipeline (GitHub Actions) for linting/tests.
- [ ] Capture backlog ideas (transfer learning, graph neural nets for topology-aware detection).
