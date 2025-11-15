from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from rich.console import Console

from .zscore import ZScoreConfig, run_zscore
from .autoencoder_detector import AEDetectorConfig, run_autoencoder_detector
from .pca_detectors import DetectorConfig, run_pca_detector

console = Console()


@dataclass
class StressConfig:
    base_dataset: Path = Path("data/processed/smart_grid_clean.parquet")
    timestamp_col: str = "timestamp"
    feature_col: str = "load_mw"
    output_dir: Path = Path("reports/stress_tests")
    spike_scale: float = 1.5
    dropout_scale: float = 0.3
    drift_rate: float = 0.02
    window: int = 48


def _inject_spike(df: pd.DataFrame, cfg: StressConfig) -> pd.DataFrame:
    df = df.copy()
    idx = int(len(df) * 0.25)
    df.loc[idx : idx + cfg.window, cfg.feature_col] *= cfg.spike_scale
    df.loc[idx : idx + cfg.window, "fault_flag"] = 1
    df.loc[idx : idx + cfg.window, "fault_type"] = "stress_spike"
    return df


def _inject_dropout(df: pd.DataFrame, cfg: StressConfig) -> pd.DataFrame:
    df = df.copy()
    idx = int(len(df) * 0.5)
    df.loc[idx : idx + cfg.window, cfg.feature_col] *= cfg.dropout_scale
    df.loc[idx : idx + cfg.window, "fault_flag"] = 1
    df.loc[idx : idx + cfg.window, "fault_type"] = "stress_dropout"
    return df


def _inject_drift(df: pd.DataFrame, cfg: StressConfig) -> pd.DataFrame:
    df = df.copy()
    idx = int(len(df) * 0.75)
    duration = cfg.window * 2
    drift = np.linspace(0, cfg.drift_rate, duration)
    df.loc[idx : idx + duration - 1, cfg.feature_col] *= (1 + drift)
    df.loc[idx : idx + duration - 1, "fault_flag"] = 1
    df.loc[idx : idx + duration - 1, "fault_type"] = "stress_drift"
    return df


INJECTORS: Dict[str, Callable[[pd.DataFrame, StressConfig], pd.DataFrame]] = {
    "spike": _inject_spike,
    "dropout": _inject_dropout,
    "drift": _inject_drift,
}


def run_stress(cfg: StressConfig, injectors: List[str]) -> Path:
    if not cfg.base_dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {cfg.base_dataset}")
    df = pd.read_parquet(cfg.base_dataset)

    results = []
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for name in injectors:
        injected = INJECTORS[name](df, cfg)
        scenario_dir = cfg.output_dir / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        injected_path = scenario_dir / f"{name}_dataset.parquet"
        injected.to_parquet(injected_path, index=False)

        # z-score
        z_cfg = ZScoreConfig(input_path=injected_path, output_dir=scenario_dir / "zscore")
        run_zscore(z_cfg)
        z_metrics = json.loads((scenario_dir / "zscore" / "zscore_metrics.json").read_text())

        # PCA detector (reuse best config)
        pca_cfg = DetectorConfig(
            projection_path=Path("data/processed/pca/pca_projection.parquet"),
            label_source_path=injected_path,
            detector="isolation_forest",
            contamination=0.03,
            train_fraction=0.8,
            component_count=16,
            output_dir=scenario_dir / "pca_iforest",
        )
        run_pca_detector(pca_cfg)
        pca_metrics = json.loads((scenario_dir / "pca_iforest" / "isolation_forest_metrics.json").read_text())

        # Autoencoder detector
        ae_cfg = AEDetectorConfig(
            errors_path=Path("models/autoencoder_baseline/reconstruction_errors.csv"),
            metadata_path=Path("models/autoencoder_baseline/metadata.json"),
            dataset_path=injected_path,
            output_dir=scenario_dir / "autoencoder",
        )
        run_autoencoder_detector(ae_cfg)
        ae_metrics = json.loads((scenario_dir / "autoencoder" / "autoencoder_metrics.json").read_text())

        results.append(
            {
                "scenario": name,
                "zscore_f1": z_metrics["f1"],
                "pca_f1": pca_metrics["f1"],
                "ae_f1": ae_metrics["f1"],
            }
        )

    summary = pd.DataFrame(results)
    summary_path = cfg.output_dir / "stress_test_summary.csv"
    summary.to_csv(summary_path, index=False)
    console.print(f"[green]Stress test summary saved to[/green] {summary_path}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inject synthetic anomalies (spike/dropout/drift) and re-evaluate detectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base", type=Path, default=StressConfig.base_dataset)
    parser.add_argument("--feature", type=str, default=StressConfig.feature_col)
    parser.add_argument("--output", type=Path, default=StressConfig.output_dir)
    parser.add_argument("--inject", type=str, nargs="+", default=["spike", "dropout", "drift"])
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = StressConfig(base_dataset=args.base, feature_col=args.feature, output_dir=args.output)
    run_stress(cfg, args.inject)


if __name__ == "__main__":  # pragma: no cover
    main()
