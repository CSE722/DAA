from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import List

from rich.console import Console

from .zscore import ZScoreConfig, run_zscore

console = Console()


@dataclass
class BenchmarkConfig:
    input_path: Path = Path("data/processed/smart_grid_clean.parquet")
    output_dir: Path = Path("reports/zscore")
    window: int = 48
    threshold: float = 2.5
    repeats: int = 5
    timestamp_col: str = "timestamp"
    label_col: str = "fault_flag"
    feature_cols: List[str] | None = None


def benchmark(cfg: BenchmarkConfig) -> Path:
    durations = []
    metrics = None
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(cfg.repeats):
        start = perf_counter()
        outputs = run_zscore(
            ZScoreConfig(
                input_path=cfg.input_path,
                timestamp_col=cfg.timestamp_col,
                label_col=cfg.label_col,
                feature_cols=cfg.feature_cols,
                window=cfg.window,
                threshold=cfg.threshold,
                output_dir=cfg.output_dir,
            ),
            materialize=(i == 0),
        )
        durations.append((perf_counter() - start) * 1000)
        metrics = outputs["metrics"]

    summary = {
        "window": cfg.window,
        "threshold": cfg.threshold,
        "repeats": cfg.repeats,
        "durations_ms": durations,
        "avg_runtime_ms": mean(durations),
        "std_runtime_ms": pstdev(durations) if cfg.repeats > 1 else 0.0,
        **metrics,
    }

    benchmark_path = cfg.output_dir / "zscore_benchmark.json"
    benchmark_path.write_text(json.dumps(summary, indent=2))
    console.print(f"[green]Saved z-score benchmark:[/] {benchmark_path}")
    return benchmark_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a single z-score configuration over multiple runs.")
    parser.add_argument("--input", type=Path, default=BenchmarkConfig.input_path)
    parser.add_argument("--window", type=int, default=BenchmarkConfig.window)
    parser.add_argument("--threshold", type=float, default=BenchmarkConfig.threshold)
    parser.add_argument("--repeats", type=int, default=BenchmarkConfig.repeats)
    parser.add_argument("--output-dir", type=Path, default=BenchmarkConfig.output_dir)
    parser.add_argument("--features", type=str, nargs="*", help="Optional feature subset")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = BenchmarkConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        window=args.window,
        threshold=args.threshold,
        repeats=args.repeats,
        feature_cols=args.features or None,
    )
    benchmark(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
