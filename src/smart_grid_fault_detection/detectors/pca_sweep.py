from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .pca_detectors import DetectorConfig, run_pca_detector


@dataclass
class PCASweepConfig:
    projection_path: Path
    label_path: Path
    components: Iterable[int]
    detectors: Iterable[str]
    contaminations: Iterable[float]
    nus: Iterable[float]
    train_fractions: Iterable[float]
    output_dir: Path


def sweep(cfg: PCASweepConfig) -> Path:
    rows = []
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for comp in cfg.components:
        for det in cfg.detectors:
            param_values = cfg.contaminations if det == "isolation_forest" else cfg.nus
            for param in param_values:
                for frac in cfg.train_fractions:
                    run_dir = cfg.output_dir / f"{det}_comp{comp}_param{param}_frac{frac}".replace(".", "p")
                    run_dir.mkdir(parents=True, exist_ok=True)
                    det_cfg = DetectorConfig(
                        projection_path=cfg.projection_path,
                        label_source_path=cfg.label_path,
                        detector=det,
                        contamination=param if det == "isolation_forest" else 0.02,
                        nu=param if det == "one_class_svm" else 0.05,
                        train_fraction=frac,
                        output_dir=run_dir,
                        component_count=comp,
                    )
                    outputs = run_pca_detector(det_cfg)
                    metrics_path = outputs["metrics"]
                    with open(metrics_path, "r", encoding="utf-8") as fp:
                        metrics = json.load(fp)
                    metrics.update(
                        {
                            "detector": det,
                            "components": comp,
                            "param": param,
                            "train_fraction": frac,
                            "metrics_path": str(metrics_path),
                            "predictions_path": str(outputs["predictions"]),
                        }
                    )
                    rows.append(metrics)

    summary = pd.DataFrame(rows)
    summary.sort_values(by="f1", ascending=False, inplace=True)
    summary_path = cfg.output_dir / "pca_detector_sweep.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep PCA detector hyperparameters (components, contamination/nu, train fraction).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--projection", type=Path, default=Path("data/processed/pca/pca_projection.parquet"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/smart_grid_clean.parquet"))
    parser.add_argument("--components", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--detectors", type=str, nargs="+", default=["isolation_forest", "one_class_svm"])
    parser.add_argument("--contaminations", type=float, nargs="+", default=[0.01, 0.02, 0.03])
    parser.add_argument("--nus", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--train-fractions", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/pca_detectors/sweep"))
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = PCASweepConfig(
        projection_path=args.projection,
        label_path=args.labels,
        components=args.components,
        detectors=args.detectors,
        contaminations=args.contaminations,
        nus=args.nus,
        train_fractions=args.train_fractions,
        output_dir=args.output_dir,
    )
    summary_path = sweep(cfg)
    print(f"Saved PCA sweep summary to {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
