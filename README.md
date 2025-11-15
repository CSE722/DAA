# High-Dimensional Fault and Anomaly Detection in Power Systems

## Overview
This repository hosts an end-to-end research/development effort to design and benchmark dimensionality reduction pipelines (PCA, autoencoders, t-SNE/UMAP) that sharpen anomaly/fault detection in smart grid telemetry (>20 features spanning demand, generation, and environmental signals). The end goal is a deployable module for an Area Load Dispatch Center (ALDC) that compresses redundant features, accelerates real-time inference, and flags abnormal load/voltage/frequency behavior along with fault-type classification output.

## Repository Layout
```
├── configs/                 # YAML/JSON configs for experiments, models, and data splits
├── data/
│   ├── raw/                 # Original smart-grid data dumps (read-only)
│   ├── external/            # Third-party/public datasets or benchmarks
│   ├── interim/             # Intermediate artifacts (cleaned, labeled windows)
│   └── processed/           # Final feature matrices ready for training/testing
├── docs/                    # Reports, design notes, meeting minutes
├── notebooks/
│   ├── eda/                 # Exploratory analysis, visualization notebooks
│   └── prototyping/         # Scratchpads for trying model ideas quickly
├── reports/
│   └── figures/             # Generated plots for papers/dashboards
├── src/
│   ├── data_prep/           # Loading, cleaning, labeling, synthetic augmentation
│   ├── features/            # Feature engineering & dimensionality-reduction utilities
│   ├── models/
│   │   ├── autoencoder/     # Sequence autoencoder architectures & training code
│   │   ├── pca/             # PCA/SVD helpers and wrappers
│   │   └── cnn/             # CNN classifier for fault-type labeling
│   ├── detectors/           # Sliding-window z-score, thresholds, ensemble logic
│   ├── visualization/       # Plotting/anomaly timeline utilities
│   └── utils/               # Shared helpers (config, logging, metrics)
├── tests/
│   ├── unit/                # Fast unit tests for utilities and model components
│   └── integration/         # End-to-end dataset/model validation suites
├── requirements.txt         # Base Python dependencies (to be populated)
└── README.md
```

## Getting Started (uv workflow)
1. Ensure [uv](https://github.com/astral-sh/uv) is installed (`brew install uv` on macOS/Linux).
2. Align Python with the project version: `uv python install 3.10 && uv python pin 3.10`.
3. Install project and dev dependencies into a local `.venv`: `uv sync --dev`.
4. Verify the scaffold and config wiring without running experiments:
   ```bash
   uv run smart-grid-fault-detection --config configs/base.yaml --dry-run --create-paths
   ```
5. Place raw grid datasets inside `data/raw/` (keep write-protection on originals) before running training scripts.
6. Manage experiment variations by copying `configs/base.yaml` and pointing the CLI to the new file via `--config`.

## Data Preparation Workflow
1. Copy `configs/data_manifest.example.yaml` → `configs/data_manifest.yaml` and edit each source block (paths, features, timezone, dtypes).
2. Stage the referenced files under `data/raw/` (CSV or Parquet supported).
3. Generate aligned interim tables plus a schema report:
   ```bash
   uv run python -m smart_grid_fault_detection.data_prep.ingest \
       --manifest configs/data_manifest.yaml \
       --save-csv
   ```
4. Inspect `data/interim/*.parquet` outputs and `docs/data_schema.md` before launching modeling experiments.
5. Sample benchmark data (e.g., the UCSD PED2 surveillance set) is mirrored under `data/raw/ped2`—see `docs/data_sources.md` for provenance before use. Convert it into tabular features with:
   ```bash
   uv run python -m smart_grid_fault_detection.data_prep.ped2_converter \
       --root data/raw/ped2 \
       --output data/interim \
       --dry-run  # drop this flag to materialize parquet files
   ```
6. A synthetic smart-grid telemetry file (`data/raw/smart_grid_signals.csv`) is wired into `configs/data_manifest.yaml`; rerun the ingestion CLI after regenerating or swapping in real SCADA/PMU feeds.

## Cleaning & Fault Augmentation
- Normalize time indices and impute gaps:
  ```python
  from smart_grid_fault_detection.data_prep import regularize_time_index

  cleaned, report = regularize_time_index(df, timestamp_col="timestamp", freq="15min")
  ```
- Clip extreme z-score outliers or fill residual NaNs via `clip_zscore_outliers` / `impute_columns`.
- Inject additional synthetic spikes/dropouts/cyber drifts that borrow severity cues from the PED2 latent motion stats:
  ```python
  from smart_grid_fault_detection.data_prep import FaultAugmentConfig, augment_faults

  aug_cfg = FaultAugmentConfig(ped2_stats=Path("data/interim/ped2_training_features.parquet"))
  df_aug = augment_faults(cleaned, config=aug_cfg)
  ```
- Run the full cleaning + augmentation loop on the latest interim artifact (produces `data/processed/smart_grid_clean.parquet` and `..._augmented.parquet`):
  ```bash
  uv run python -m smart_grid_fault_detection.data_prep.process_pipeline \
      --input data/interim/smart_grid_master.parquet
  ```

## Exploratory Analysis
- Launch JupyterLab inside the uv environment and open `notebooks/eda/01_data_overview.ipynb`:
  ```bash
  uv run jupyter lab
  ```
- The notebook loads interim, cleaned, and augmented tables (plus optional PED2 features) to produce summary stats, correlation heatmaps, and synthetic-fault timelines. Update it as new datasets arrive.

## Dimensionality Reduction (PCA)
- Fit PCA on the cleaned feature matrix and emit projections/components/variance tables:
  ```bash
  uv run python -m smart_grid_fault_detection.models.pca_pipeline \
      --input data/processed/smart_grid_clean.parquet \
      --output data/processed/pca \
      --variance-threshold 0.9
  ```
- Outputs: `components.parquet`, `pca_projection.parquet`, `variance.csv`, and `pca_model.joblib` under the chosen output directory.

## Statistical Baseline (Sliding Z-Score)
- Run the baseline detector over cleaned features (defaults: load/voltage/frequency, 32-sample window):
  ```bash
  uv run python -m smart_grid_fault_detection.detectors.zscore \
      --input data/processed/smart_grid_clean.parquet \
      --window 32 \
      --threshold 3.0 \
      --output-dir reports/zscore
  ```
- Produces `zscore_metrics.json` plus per-timestamp predictions/scores in `zscore_predictions.csv`.
- Benchmark multiple hyper-parameter combos and capture a summary table:
  ```bash
  uv run python -m smart_grid_fault_detection.detectors.zscore_sweep \
      --input data/processed/smart_grid_clean.parquet \
      --windows 16 24 32 48 \
      --thresholds 2.5 3.0 3.5 4.0 \
      --output-dir reports/zscore_sweep
  ```
  The sweep writes one subfolder per setting plus `zscore_sweep_summary.csv` sorted by F1.
- Capture latency numbers for a chosen configuration (default: best sweep result `window=48`, `threshold=2.5`):
  ```bash
  uv run python -m smart_grid_fault_detection.detectors.zscore_benchmark \
      --input data/processed/smart_grid_clean.parquet \
      --window 48 \
      --threshold 2.5 \
      --repeats 5 \
      --output-dir reports/zscore_benchmark
  ```
  Produces `zscore_benchmark.json` containing avg/std runtime plus accuracy metrics.

## PCA + Classical Detectors
- Consume `pca_projection.parquet` and evaluate classical anomaly detectors (Isolation Forest or One-Class SVM):
  ```bash
  uv run python -m smart_grid_fault_detection.detectors.pca_detectors \
      --projection data/processed/pca/pca_projection.parquet \
      --labels data/processed/smart_grid_clean.parquet \
      --detector isolation_forest \
      --output-dir reports/pca_detectors
  ```
- Metrics (`*_metrics.json`) and timestamp-level predictions (`*_predictions.csv`) are stored in `reports/pca_detectors/` for downstream reporting/comparison.

## LSTM Autoencoder (Neural Detector)
- Train a sequence-to-sequence autoencoder on sliding windows of the cleaned dataset (defaults: 32-sample windows, stacked encoder/decoder). Metal acceleration is provided through `tensorflow-macos`/`tensorflow-metal`.
  ```bash
  uv run python -m smart_grid_fault_detection.models.autoencoder \
      --input data/processed/smart_grid_clean.parquet \
      --sequence-length 48 \
      --latent-dim 32 \
      --epochs 40 \
      --output-dir models/autoencoder_baseline
  ```
- Artifacts: `autoencoder.keras`, `scaler.joblib`, `history.json`, and `reconstruction_errors.csv` (includes a 95th percentile threshold for anomaly scoring). Verify GPU visibility before training:
  ```bash
  uv run python - <<'PY'
  import tensorflow as tf
  print(tf.config.list_physical_devices("GPU"))
  PY
  ```

## CNN Fault-Type Classifier (Upcoming)
- `src/models/cnn/` is reserved for a convolutional classifier that will ingest spatial-temporal tensors (PED2-derived grids or reshaped SCADA windows) to label spikes vs dropouts vs cyber anomalies. Integrate shared preprocessing (scaling, windowing) from the autoencoder pipeline and log metrics next to the PCA/z-score baselines once implemented.

## Manifold Visualization (t-SNE / UMAP)
- Generate 2-D embeddings (defaults to PCA projection input, colored by `fault_type`):
  ```bash
  uv run python -m smart_grid_fault_detection.visualization.manifold \
      --input data/processed/pca/pca_projection.parquet \
      --label-col fault_type \
      --method tsne \
      --output-dir reports/manifold \
      --figure-dir reports/figures
  ```
- Outputs an embedding CSV plus a scatter plot (`reports/figures/{method}_embedding.png`). For UMAP support install the dependency once via `uv sync --dev` (already declared) or `uv pip install umap-learn` if working outside the synced environment.

## Pipeline Snapshot
1. **Data Prep** – ingest raw feeds, align timestamps, impute gaps, synthesize labeled fault cases.
2. **Dimensionality Reduction** – evaluate PCA variance retention, t-SNE/UMAP visualization, and LSTM autoencoder latent compression for different window sizes.
3. **Detection Models** – compare sliding-window statistical detectors, PCA+classic classifiers, LSTM autoencoders, and CNN-based fault typing.
4. **Evaluation** – benchmark accuracy, F1, detection delay, throughput, and interpretability gaps between full vs reduced spaces.
5. **Reporting & Integration** – visualize anomalies, export alerts, and document hooks for ALDC integration.

Refer to `docs/TODO.md` for the current work breakdown and milestone tracking.
