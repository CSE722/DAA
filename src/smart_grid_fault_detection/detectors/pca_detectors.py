from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from rich.console import Console
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import OneClassSVM
import re


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    if not is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, utc=True)
    else:
        if getattr(series.dt, "tz", None) is None:
            series = series.dt.tz_localize("UTC")
        else:
            series = series.dt.tz_convert("UTC")
    return series

console = Console()

DetectorType = Literal["isolation_forest", "one_class_svm"]


@dataclass
class DetectorConfig:
    projection_path: Path = Path("data/processed/pca/pca_projection.parquet")
    label_source_path: Path = Path("data/processed/smart_grid_clean.parquet")
    timestamp_col: str = "timestamp"
    label_col: str = "fault_flag"
    detector: DetectorType = "isolation_forest"
    train_fraction: float = 0.7
    train_on_normal_only: bool = True
    contamination: float = 0.02
    nu: float = 0.05
    random_state: int = 13
    output_dir: Path = Path("reports/pca_detectors")
    component_count: int | None = None


def _load_frames(cfg: DetectorConfig) -> pd.DataFrame:
    if not cfg.projection_path.exists():
        raise FileNotFoundError(f"PCA projection file not found: {cfg.projection_path}")
    if not cfg.label_source_path.exists():
        raise FileNotFoundError(f"Label source not found: {cfg.label_source_path}")

    proj = pd.read_parquet(cfg.projection_path)
    labels = pd.read_parquet(cfg.label_source_path)

    proj[cfg.timestamp_col] = _normalize_timestamp(proj[cfg.timestamp_col])
    labels[cfg.timestamp_col] = _normalize_timestamp(labels[cfg.timestamp_col])

    merged = proj.merge(
        labels[[cfg.timestamp_col, cfg.label_col, "fault_type"]],
        on=cfg.timestamp_col,
        how="inner",
        suffixes=("", "_label"),
    )

    if merged.empty:
        raise ValueError("No overlapping timestamps between projection and label dataset.")
    merged = merged.sort_values(cfg.timestamp_col).reset_index(drop=True)
    return merged


def _sorted_pc_columns(columns: List[str]) -> List[str]:
    def key(col: str) -> int:
        match = re.search(r"pc(\d+)", col)
        return int(match.group(1)) if match else 0

    return sorted(columns, key=key)


def _split_data(df: pd.DataFrame, cfg: DetectorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = _sorted_pc_columns([c for c in df.columns if c.startswith("pc")])
    if cfg.component_count:
        feature_cols = feature_cols[: cfg.component_count]
    if not feature_cols:
        raise ValueError("Projection dataframe missing pc* columns.")

    n_train = int(len(df) * cfg.train_fraction)
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    if cfg.train_on_normal_only:
        train_df = train_df[train_df[cfg.label_col] == 0]
        if train_df.empty:
            raise ValueError("No normal samples available for training after filtering.")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df[cfg.label_col].to_numpy(dtype=int)

    return X_train, X_test, y_test, df[feature_cols].to_numpy(dtype=np.float32)


def _build_detector(cfg: DetectorConfig):
    if cfg.detector == "isolation_forest":
        return IsolationForest(
            contamination=cfg.contamination,
            random_state=cfg.random_state,
            n_estimators=300,
        )
    if cfg.detector == "one_class_svm":
        return OneClassSVM(kernel="rbf", nu=cfg.nu, gamma="scale")
    raise ValueError(f"Unsupported detector type {cfg.detector}")


def _predict_scores(model, X: np.ndarray, detector: DetectorType) -> np.ndarray:
    if detector in {"isolation_forest", "one_class_svm"}:
        scores = model.decision_function(X)
        return scores
    raise ValueError("Unsupported detector for scoring")


def _scores_to_labels(model, X: np.ndarray, detector: DetectorType) -> np.ndarray:
    raw = model.predict(X)
    return np.where(raw == -1, 1, 0)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_pca_detector(cfg: DetectorConfig) -> Dict[str, Path | Dict[str, float]]:
    df = _load_frames(cfg)
    X_train, X_test, y_test, X_all = _split_data(df, cfg)

    model = _build_detector(cfg)
    model.fit(X_train)

    y_pred = _scores_to_labels(model, X_test, cfg.detector)
    metrics = _compute_metrics(y_test, y_pred)
    console.print(
        f"[cyan]{cfg.detector} metrics:[/] "
        + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    )

    scores_all = _predict_scores(model, X_all, cfg.detector)
    predictions_df = df[[cfg.timestamp_col, cfg.label_col, "fault_type"]].copy()
    predictions_df["score"] = scores_all
    predictions_df["prediction"] = _scores_to_labels(model, X_all, cfg.detector)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    preds_path = cfg.output_dir / f"{cfg.detector}_predictions.csv"
    predictions_df.to_csv(preds_path, index=False)

    metrics_entry = {**metrics, "train_fraction": cfg.train_fraction}
    metrics_path = cfg.output_dir / f"{cfg.detector}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_entry, indent=2))

    return {"metrics": metrics_path, "predictions": preds_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PCA-based classical detectors on the smart-grid dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--projection", type=Path, default=DetectorConfig.projection_path)
    parser.add_argument("--labels", type=Path, default=DetectorConfig.label_source_path)
    parser.add_argument("--timestamp-col", type=str, default=DetectorConfig.timestamp_col)
    parser.add_argument("--label-col", type=str, default=DetectorConfig.label_col)
    parser.add_argument("--detector", choices=["isolation_forest", "one_class_svm"], default="isolation_forest")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--no-normal-filter", action="store_true", help="Train on all samples, not just normal ones.")
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--components", type=int, default=None, help="Number of principal components to keep.")
    parser.add_argument("--output-dir", type=Path, default=DetectorConfig.output_dir)
    parser.add_argument("--random-state", type=int, default=13)
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = DetectorConfig(
        projection_path=args.projection,
        label_source_path=args.labels,
        timestamp_col=args.timestamp_col,
        label_col=args.label_col,
        detector=args.detector,
        train_fraction=args.train_fraction,
        train_on_normal_only=not args.no_normal_filter,
        contamination=args.contamination,
        nu=args.nu,
        component_count=args.components,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )

    run_pca_detector(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
