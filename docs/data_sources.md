# Data Sources

## PED2 Benchmark
- **Origin**: [`github.com/zshafique25/anomaly-detection`](https://github.com/zshafique25/anomaly-detection/tree/master/ped2)
- **Contents**: `training/` and `testing/` directories with UCSD Ped2 surveillance videos commonly used for anomaly detection benchmarking.
- **Local Path**: `data/raw/ped2`
- **Usage Notes**:
  - Files remain in their original directory layout to avoid path mismatches with reference notebooks.
  - Integrate via a manifest entry if you plan to reinterpret the frames as time series (e.g., aggregate per-frame statistics before feeding them to the smart-grid pipeline).
  - Respect the original dataset license/distribution terms when sharing downstream artifacts.
