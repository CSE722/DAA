from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

console = Console()


@dataclass
class AEDetectorConfig:
    errors_path: Path = Path("models/autoencoder_baseline/reconstruction_errors.csv")
    metadata_path: Path = Path("models/autoencoder_baseline/metadata.json")
    dataset_path: Path = Path("data/processed/smart_grid_clean.parquet")
    label_col: str = "fault_flag"
    threshold: float = 0.657140194  # 80th percentile from evaluation sweep
    output_dir: Path = Path("reports/autoencoder")


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return json.loads(path.read_text())


def _align(labels: np.ndarray, seq_len: int, n_windows: int) -> np.ndarray:
    start = seq_len - 1
    end = start + n_windows
    if end > len(labels):
        raise ValueError("Dataset shorter than expected for autoencoder windows")
    return labels[start:end]


def run_autoencoder_detector(cfg: AEDetectorConfig) -> dict[str, Path]:
    if not cfg.errors_path.exists():
        raise FileNotFoundError(f"Reconstruction errors missing: {cfg.errors_path}")
    errors_df = pd.read_csv(cfg.errors_path, parse_dates=["timestamp"])
    metadata = _load_metadata(cfg.metadata_path)
    seq_len = int(metadata["sequence_length"])

    df = pd.read_parquet(cfg.dataset_path)
    labels = df[cfg.label_col].to_numpy(dtype=int)
    y_true = _align(labels, seq_len, len(errors_df))

    errors = errors_df["reconstruction_error"].to_numpy()
    preds = (errors >= cfg.threshold).astype(int)

    metrics = {
        "threshold": cfg.threshold,
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    preds_df = errors_df.copy()
    preds_df[cfg.label_col] = y_true
    preds_df["prediction"] = preds
    preds_df["threshold"] = cfg.threshold
    preds_path = cfg.output_dir / "autoencoder_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    metrics_path = cfg.output_dir / "autoencoder_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    console.print(
        "[cyan]Autoencoder detector metrics:[/] "
        + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k not in {"threshold"})
    )

    return {"metrics": metrics_path, "predictions": preds_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply an LSTM autoencoder detector using reconstruction errors and a chosen threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--errors", type=Path, default=AEDetectorConfig.errors_path)
    parser.add_argument("--metadata", type=Path, default=AEDetectorConfig.metadata_path)
    parser.add_argument("--dataset", type=Path, default=AEDetectorConfig.dataset_path)
    parser.add_argument("--label-col", type=str, default=AEDetectorConfig.label_col)
    parser.add_argument("--threshold", type=float, default=AEDetectorConfig.threshold)
    parser.add_argument("--output-dir", type=Path, default=AEDetectorConfig.output_dir)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = AEDetectorConfig(
        errors_path=args.errors,
        metadata_path=args.metadata,
        dataset_path=args.dataset,
        label_col=args.label_col,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )
    run_autoencoder_detector(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
