from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

console = Console()


@dataclass
class ZScoreConfig:
    input_path: Path = Path("data/processed/smart_grid_clean.parquet")
    timestamp_col: str = "timestamp"
    label_col: str = "fault_flag"
    feature_cols: List[str] | None = None
    window: int = 24
    threshold: float = 3.5
    min_periods: int = 12
    output_dir: Path = Path("reports/zscore")

    def __post_init__(self):
        if self.feature_cols is None:
            self.feature_cols = [
                "load_mw",
                "voltage_kv",
                "frequency_hz",
            ]


def _load_dataset(cfg: ZScoreConfig) -> pd.DataFrame:
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cfg.input_path}")
    if cfg.input_path.suffix == ".parquet":
        df = pd.read_parquet(cfg.input_path)
    else:
        df = pd.read_csv(cfg.input_path, parse_dates=[cfg.timestamp_col])
    for col in cfg.feature_cols + [cfg.label_col]:
        if col not in df:
            raise KeyError(f"Column '{col}' missing from dataset")
    df = df.sort_values(cfg.timestamp_col).reset_index(drop=True)
    return df


def _rolling_zscores(df: pd.DataFrame, cfg: ZScoreConfig) -> pd.DataFrame:
    zscores = {}
    for col in cfg.feature_cols:
        rolling = df[col].rolling(window=cfg.window, min_periods=cfg.min_periods)
        mean = rolling.mean()
        std = rolling.std().replace(0, np.nan)
        z = (df[col] - mean) / std
        zscores[col] = z.abs()
    return pd.DataFrame(zscores)


def _score_to_labels(scores: pd.Series, threshold: float) -> np.ndarray:
    return (scores > threshold).astype(int).to_numpy()


def run_zscore(cfg: ZScoreConfig, materialize: bool = True) -> dict[str, object]:
    df = _load_dataset(cfg)
    z_df = _rolling_zscores(df, cfg)
    score = z_df.max(axis=1).fillna(0)
    predictions = _score_to_labels(score, cfg.threshold)
    y_true = df[cfg.label_col].to_numpy(dtype=int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    console.print("[cyan]z-score metrics:[/cyan] " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    artifacts: dict[str, object] = {"metrics": metrics}
    if materialize:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        preds_df = df[[cfg.timestamp_col, cfg.label_col]].copy()
        preds_df["zscore"] = score
        preds_df["prediction"] = predictions
        preds_path = cfg.output_dir / "zscore_predictions.csv"
        preds_df.to_csv(preds_path, index=False)

        metrics_path = cfg.output_dir / "zscore_metrics.json"
        metrics_path.write_text(
            json.dumps({**metrics, "window": cfg.window, "threshold": cfg.threshold}, indent=2)
        )
        artifacts["metrics_path"] = metrics_path
        artifacts["predictions_path"] = preds_path

    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sliding-window z-score baseline detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=ZScoreConfig.input_path)
    parser.add_argument("--timestamp-col", type=str, default=ZScoreConfig.timestamp_col)
    parser.add_argument("--label-col", type=str, default=ZScoreConfig.label_col)
    parser.add_argument("--features", type=str, nargs="*", help="Columns to monitor for z-score anomalies.")
    parser.add_argument("--window", type=int, default=ZScoreConfig.window)
    parser.add_argument("--min-periods", type=int, default=ZScoreConfig.min_periods)
    parser.add_argument("--threshold", type=float, default=ZScoreConfig.threshold)
    parser.add_argument("--output-dir", type=Path, default=ZScoreConfig.output_dir)
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = ZScoreConfig(
        input_path=args.input,
        timestamp_col=args.timestamp_col,
        label_col=args.label_col,
        feature_cols=args.features or None,
        window=args.window,
        min_periods=args.min_periods,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )
    run_zscore(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
