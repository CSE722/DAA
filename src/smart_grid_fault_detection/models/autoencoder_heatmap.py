from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

@dataclass
class HeatmapConfig:
    model_dir: Path = Path("models/autoencoder_baseline")
    dataset_path: Path = Path("data/processed/smart_grid_clean.parquet")
    timestamp_col: str = "timestamp"
    label_col: str = "fault_type"
    output_dir: Path = Path("reports/autoencoder")


def _load_metadata(model_dir: Path) -> dict:
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    return json.loads(meta_path.read_text())


def _create_sequences(values: np.ndarray, seq_len: int) -> np.ndarray:
    sequences = []
    for start in range(0, len(values) - seq_len + 1):
        end = start + seq_len
        sequences.append(values[start:end])
    return np.stack(sequences)


def build_heatmaps(cfg: HeatmapConfig) -> dict[str, Path]:
    metadata = _load_metadata(cfg.model_dir)
    seq_len = int(metadata["sequence_length"])

    df = pd.read_parquet(cfg.dataset_path)
    scaler_bundle = joblib.load(cfg.model_dir / "scaler.joblib")
    scaler = scaler_bundle["scaler"]
    features = scaler_bundle["features"]
    values = df[features].to_numpy(dtype=np.float32)
    values_scaled = scaler.transform(values)

    sequences = _create_sequences(values_scaled, seq_len)

    model = tf.keras.models.load_model(cfg.model_dir / "autoencoder.keras")
    recon = model.predict(sequences, batch_size=128, verbose=0)
    abs_errors = np.abs(sequences - recon)

    avg_feature_error = abs_errors.mean(axis=(0, 1))  # feature-level
    feature_df = pd.DataFrame({"feature": features, "avg_abs_error": avg_feature_error})

    heatmap_matrix = abs_errors.mean(axis=0)  # (seq_len, n_features)
    heatmap_df = pd.DataFrame(heatmap_matrix, columns=features)
    heatmap_df.insert(0, "time_index", range(seq_len))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cfg.output_dir / "autoencoder_feature_errors.csv"
    feature_df.to_csv(feature_path, index=False)

    heatmap_path = cfg.output_dir / "autoencoder_time_feature_heatmap.csv"
    heatmap_df.to_csv(heatmap_path, index=False)

    return {"feature_errors": feature_path, "heatmap": heatmap_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute aggregate reconstruction-error heatmaps for the autoencoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=HeatmapConfig.model_dir)
    parser.add_argument("--dataset", type=Path, default=HeatmapConfig.dataset_path)
    parser.add_argument("--output-dir", type=Path, default=HeatmapConfig.output_dir)
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = HeatmapConfig(model_dir=args.model_dir, dataset_path=args.dataset, output_dir=args.output_dir)
    paths = build_heatmaps(cfg)
    print(f"Saved heatmap CSVs: {paths}")


if __name__ == "__main__":  # pragma: no cover
    main()
