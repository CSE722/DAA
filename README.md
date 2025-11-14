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

## Pipeline Snapshot
1. **Data Prep** – ingest raw feeds, align timestamps, impute gaps, synthesize labeled fault cases.
2. **Dimensionality Reduction** – evaluate PCA variance retention, t-SNE/UMAP visualization, and LSTM autoencoder latent compression for different window sizes.
3. **Detection Models** – compare sliding-window statistical detectors, PCA+classic classifiers, LSTM autoencoders, and CNN-based fault typing.
4. **Evaluation** – benchmark accuracy, F1, detection delay, throughput, and interpretability gaps between full vs reduced spaces.
5. **Reporting & Integration** – visualize anomalies, export alerts, and document hooks for ALDC integration.

Refer to `docs/TODO.md` for the current work breakdown and milestone tracking.
