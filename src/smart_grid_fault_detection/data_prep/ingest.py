from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from rich.console import Console
from rich.progress import track

from .manifest import DatasetManifest, SourceConfig, load_manifest

console = Console()


def _read_source(cfg: SourceConfig) -> pd.DataFrame:
    """Load a single dataset based on its declared format."""

    if not cfg.path.exists():
        raise FileNotFoundError(f"Source '{cfg.name}' missing at {cfg.path}")

    read_kwargs = {
        "parse_dates": [cfg.timestamp_col],
        "na_values": cfg.null_markers,
    }
    if cfg.dtypes:
        read_kwargs["dtype"] = cfg.dtypes
    if cfg.format == "csv":
        df = pd.read_csv(cfg.path, **read_kwargs)
    elif cfg.format == "parquet":
        df = pd.read_parquet(cfg.path)
    else:  # pragma: no cover - protected by Literal typing
        raise ValueError(f"Unsupported format '{cfg.format}'")

    if cfg.features:
        keep_cols = list(dict.fromkeys(cfg.features + [cfg.timestamp_col]))
        missing = set(keep_cols).difference(df.columns)
        if missing:
            raise KeyError(f"Source '{cfg.name}' missing columns: {missing}")
        df = df[keep_cols]

    df = df.sort_values(cfg.timestamp_col)
    if cfg.timezone:
        ts = df[cfg.timestamp_col]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(cfg.timezone, nonexistent="shift_forward", ambiguous="NaT")
        ts = ts.dt.tz_convert("UTC")
        df[cfg.timestamp_col] = ts
    return df


def _merge_sources(frames: Iterable[pd.DataFrame], timestamp_col: str, how: str) -> pd.DataFrame:
    frames = list(frames)
    if not frames:
        raise ValueError("No frames supplied for merging.")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=timestamp_col, how=how)
    return merged


def _save_artifacts(df: pd.DataFrame, manifest: DatasetManifest, save_csv: bool) -> None:
    output = manifest.output
    output.interim_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output.interim_table, index=False)
    console.print(f"[green]Saved parquet:[/] {output.interim_table}")

    if save_csv:
        csv_path = output.interim_table.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        console.print(f"[green]Saved csv:[/] {csv_path}")

    report_path = manifest.output.schema_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fp:
        fp.write("# Data Schema\n\n")
        for col, dtype in df.dtypes.items():
            fp.write(f"- **{col}**: `{dtype}`\n")
    console.print(f"[green]Wrote schema report:[/] {report_path}")


def ingest_sources(manifest_path: Path, save_csv: bool = False) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)

    frames: List[pd.DataFrame] = []
    for source in track(manifest.sources, description="Reading sources"):
        df = _read_source(source)
        frames.append(df)

    merged = _merge_sources(frames, manifest.timestamp_col, manifest.merge_strategy)
    if manifest.sort_timestamps:
        merged = merged.sort_values(manifest.timestamp_col)

    _save_artifacts(merged, manifest, save_csv)
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest raw smart-grid datasets as defined in a manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/data_manifest.example.yaml"),
        help="Path to a YAML file describing raw data sources.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Emit an additional CSV alongside the parquet artifact.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    ingest_sources(args.manifest, save_csv=args.save_csv)


if __name__ == "__main__":  # pragma: no cover
    main()
