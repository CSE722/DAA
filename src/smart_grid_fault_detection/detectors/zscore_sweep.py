from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .zscore import ZScoreConfig, run_zscore


@dataclass
class SweepConfig:
    input_path: Path
    output_dir: Path
    windows: Iterable[int]
    thresholds: Iterable[float]
    timestamp_col: str = "timestamp"
    label_col: str = "fault_flag"
    feature_cols: List[str] | None = None


def sweep(cfg: SweepConfig) -> Path:
    results = []
    for window in cfg.windows:
        for threshold in cfg.thresholds:
            z_cfg = ZScoreConfig(
                input_path=cfg.input_path,
                timestamp_col=cfg.timestamp_col,
                label_col=cfg.label_col,
                feature_cols=cfg.feature_cols,
                window=window,
                threshold=threshold,
                output_dir=cfg.output_dir / f"w{window}_t{threshold:.1f}".replace(".", "p"),
            )
            outputs = run_zscore(z_cfg)
            metrics = dict(outputs["metrics"])
            metrics.update({"window": window, "threshold": threshold})
            results.append(metrics)

    summary = pd.DataFrame(results)
    summary.sort_values(by="f1", ascending=False, inplace=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.output_dir / "zscore_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep z-score windows/thresholds.")
    parser.add_argument("--input", type=Path, default=ZScoreConfig.input_path)
    parser.add_argument("--windows", type=int, nargs="+", default=[16, 24, 32, 48])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[2.5, 3.0, 3.5, 4.0])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/zscore_sweep"))
    parser.add_argument("--features", type=str, nargs="*", help="Optional feature list")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = SweepConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        windows=args.windows,
        thresholds=args.thresholds,
        feature_cols=args.features or None,
    )
    summary_path = sweep(cfg)
    print(f"Saved sweep summary to {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
