from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
from rich.console import Console

from .cleaning import GapFillReport, clip_zscore_outliers, impute_columns, regularize_time_index
from .faults import FaultAugmentConfig, augment_faults

console = Console()


@dataclass
class ProcessArgs:
    input_path: Path = Path("data/interim/smart_grid_master.parquet")
    clean_output: Path = Path("data/processed/smart_grid_clean.parquet")
    augmented_output: Path | None = Path("data/processed/smart_grid_augmented.parquet")
    freq: str = "15min"
    timestamp_col: str = "timestamp"
    zscore_columns: List[str] = field(
        default_factory=lambda: [
            "load_mw",
            "voltage_kv",
            "frequency_hz",
            "line_temp_c",
            "transformer_load_pct",
            "reactive_power_mvar",
        ]
    )
    ped2_stats: Path | None = None


def _load_frame(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)
    if input_path.suffix == ".csv":
        return pd.read_csv(input_path, parse_dates=["timestamp"])
    raise ValueError(f"Unsupported input format {input_path.suffix}")


def run_pipeline(args: ProcessArgs) -> dict[str, GapFillReport | Path]:
    df = _load_frame(args.input_path)
    console.print(f"[cyan]Loaded[/cyan] {len(df):,} rows from {args.input_path}")

    cleaned, report = regularize_time_index(df, args.timestamp_col, args.freq)
    console.print(
        "[green]Regularized timeline:[/] "
        f"expected={report.expected_points:,}, filled={report.filled_points:,}, remaining_na={report.remaining_na}"
    )

    cleaned = clip_zscore_outliers(cleaned, args.zscore_columns)
    cleaned = impute_columns(cleaned)

    args.clean_output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(args.clean_output, index=False)
    console.print(f"[green]Saved cleaned data:[/] {args.clean_output}")

    outputs: dict[str, GapFillReport | Path] = {"report": report, "clean": args.clean_output}

    if args.augmented_output:
        aug_cfg = FaultAugmentConfig(ped2_stats=args.ped2_stats)
        augmented = augment_faults(cleaned, config=aug_cfg)
        args.augmented_output.parent.mkdir(parents=True, exist_ok=True)
        augmented.to_parquet(args.augmented_output, index=False)
        console.print(f"[green]Saved augmented data:[/] {args.augmented_output}")
        outputs["augmented"] = args.augmented_output

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean and augment smart-grid datasets starting from interim parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=Path("data/interim/smart_grid_master.parquet"))
    parser.add_argument("--clean-output", type=Path, default=Path("data/processed/smart_grid_clean.parquet"))
    parser.add_argument("--augmented-output", type=Path, default=Path("data/processed/smart_grid_augmented.parquet"))
    parser.add_argument("--freq", type=str, default="15min")
    parser.add_argument("--timestamp-col", type=str, default="timestamp")
    parser.add_argument(
        "--zscore-columns",
        type=str,
        nargs="*",
        default=None,
        help="Columns to winsorize via z-score clipping. Defaults to common electrical features.",
    )
    parser.add_argument(
        "--ped2-stats",
        type=Path,
        default=Path("data/interim/ped2_training_features.parquet"),
        help="Optional path to PED2 feature parquet for severity scaling.",
    )
    parser.add_argument("--no-augment", action="store_true", help="Skip synthetic fault augmentation.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    zscore_columns = args.zscore_columns or ProcessArgs().zscore_columns
    process_args = ProcessArgs(
        input_path=args.input,
        clean_output=args.clean_output,
        augmented_output=None if args.no_augment else args.augmented_output,
        freq=args.freq,
        timestamp_col=args.timestamp_col,
        zscore_columns=zscore_columns,
        ped2_stats=args.ped2_stats if not args.no_augment else None,
    )

    run_pipeline(process_args)


if __name__ == "__main__":  # pragma: no cover
    main()
