# Data Sources

## PED2 Benchmark
- **Origin**: [`github.com/zshafique25/anomaly-detection`](https://github.com/zshafique25/anomaly-detection/tree/master/ped2)
- **Contents**: `training/` and `testing/` directories with UCSD Ped2 surveillance videos commonly used for anomaly detection benchmarking (frame-level `.tif` images plus ground-truth masks).
- **Local Path**: `data/raw/ped2`
- **Conversion Utility**: `uv run python -m smart_grid_fault_detection.data_prep.ped2_converter --root data/raw/ped2 --output data/interim`
  - Produces `data/interim/ped2_training_features.parquet` and `data/interim/ped2_testing_features.parquet`.
  - Each row corresponds to a frame with >20 spatial summary features (grid-wise means, gradients, frame differences).
- **Usage Notes**:
  - The converter skips `_gt` directories (ground-truth masks) automatically.
  - Treat the generated parquet files like any other tabular source—reference them from a manifest if you want to mix them with grid telemetry.
  - Respect the original dataset license/distribution terms when sharing downstream artifacts.

## Synthetic Smart-Grid Telemetry
- **Origin**: Procedurally generated via `uv run python data/raw/...` script (see Git history for generator snippet).
- **Contents**: Two weeks of 15-minute samples with 30+ electrical and environmental features plus `fault_type`/`fault_flag`.
- **Local Path**: `data/raw/smart_grid_signals.csv`
- **Usage Notes**:
  - Already wired into `configs/data_manifest.yaml` for ingestion experiments.
  - Fault windows (spike, dropout, cyber drift) were injected with randomized durations/magnitudes to mimic rare events.
  - Extend/regen by editing the generator script snippet in the README or converting it into a reusable utility under `src/data_prep`.
