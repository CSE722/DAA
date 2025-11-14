from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from rich.console import Console
from rich.progress import track

console = Console()


@dataclass(frozen=True)
class Ped2ConversionConfig:
    root: Path = Path("data/raw/ped2")
    output_dir: Path = Path("data/interim")
    resize: Tuple[int, int] = (160, 120)  # width, height
    grid: Tuple[int, int] = (4, 5)  # rows, cols => 20 spatial bins
    fps: float = 30.0
    normalize: bool = True


SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg")


def _load_frame(path: Path, size: Tuple[int, int], normalize: bool) -> np.ndarray:
    image = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    if normalize:
        arr /= 255.0
    return arr


def _grid_feature_names(rows: int, cols: int) -> Iterable[str]:
    for r in range(rows):
        for c in range(cols):
            yield f"cell_mean_r{r}_c{c}"


def _extract_features(arr: np.ndarray, prev_arr: np.ndarray | None, grid: Tuple[int, int]) -> Dict[str, float]:
    h, w = arr.shape
    grid_rows, grid_cols = grid
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    usable_h = cell_h * grid_rows
    usable_w = cell_w * grid_cols
    cropped = arr[:usable_h, :usable_w]

    features: Dict[str, float] = {
        "pixel_mean": float(arr.mean()),
        "pixel_std": float(arr.std()),
        "pixel_min": float(arr.min()),
        "pixel_max": float(arr.max()),
        "pixel_energy": float(np.mean(arr ** 2)),
    }

    grid_means = cropped.reshape(grid_rows, cell_h, grid_cols, cell_w).mean(axis=(1, 3))
    for name, value in zip(_grid_feature_names(grid_rows, grid_cols), grid_means.flatten(order="C")):
        features[name] = float(value)

    if prev_arr is None:
        features["frame_absdiff_mean"] = 0.0
        features["frame_absdiff_std"] = 0.0
    else:
        diff = np.abs(arr - prev_arr)
        features["frame_absdiff_mean"] = float(diff.mean())
        features["frame_absdiff_std"] = float(diff.std())

    gx, gy = np.gradient(arr)
    features["grad_mean"] = float((np.abs(gx) + np.abs(gy)).mean())
    features["grad_std"] = float((np.abs(gx) + np.abs(gy)).std())

    return features


def _iter_sequences(root: Path, split: str) -> Iterable[Path]:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"PED2 split directory not found: {split_dir}")
    for seq_dir in sorted(split_dir.iterdir()):
        if seq_dir.is_dir():
            yield seq_dir


def process_sequence(seq_dir: Path, split: str, cfg: Ped2ConversionConfig) -> pd.DataFrame:
    frame_paths = sorted(
        [p for p in seq_dir.glob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )
    if not frame_paths:
        console.print(f"[yellow]Skipping {seq_dir}: no frame files detected.[/yellow]")
        return pd.DataFrame()

    rows: List[Dict[str, float]] = []
    prev_arr: np.ndarray | None = None
    for idx, frame_path in enumerate(frame_paths):
        arr = _load_frame(frame_path, cfg.resize, cfg.normalize)
        features = _extract_features(arr, prev_arr, cfg.grid)
        prev_arr = arr

        features.update(
            {
                "split": split,
                "sequence": seq_dir.name,
                "frame_index": idx,
                "timestamp_s": idx / cfg.fps,
                "frame_path": str(frame_path.relative_to(cfg.root)),
            }
        )
        rows.append(features)
    return pd.DataFrame(rows)


def convert_ped2(cfg: Ped2ConversionConfig, dry_run: bool = False) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    for split in ("training", "testing"):
        all_frames: List[pd.DataFrame] = []
        sequence_dirs = list(_iter_sequences(cfg.root, split))
        for seq_dir in track(sequence_dirs, description=f"Processing {split}"):
            df_seq = process_sequence(seq_dir, split, cfg)
            if df_seq.empty:
                continue
            all_frames.append(df_seq)
        if not all_frames:
            console.print(f"[yellow]No sequences found for split {split}. Skipping.[/yellow]")
            continue
        split_df = pd.concat(all_frames, ignore_index=True)
        output_path = cfg.output_dir / f"ped2_{split}_features.parquet"
        outputs[split] = output_path
        if dry_run:
            console.print(
                f"[cyan]Dry-run:[/] would write {len(split_df):,} rows to {output_path}"
            )
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            split_df.to_parquet(output_path, index=False)
            console.print(
                f"[green]Saved {split} features:[/] {output_path} ({len(split_df):,} rows)"
            )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert UCSD PED2 frame directories into tabular features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Ped2ConversionConfig.root)
    parser.add_argument("--output", type=Path, default=Ped2ConversionConfig.output_dir)
    parser.add_argument("--width", type=int, default=Ped2ConversionConfig.resize[0])
    parser.add_argument("--height", type=int, default=Ped2ConversionConfig.resize[1])
    parser.add_argument("--grid-rows", type=int, default=Ped2ConversionConfig.grid[0])
    parser.add_argument("--grid-cols", type=int, default=Ped2ConversionConfig.grid[1])
    parser.add_argument("--fps", type=float, default=Ped2ConversionConfig.fps)
    parser.add_argument("--no-normalize", action="store_true", help="Keep raw 0-255 values.")
    parser.add_argument("--dry-run", action="store_true", help="Skip writing parquet files.")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = Ped2ConversionConfig(
        root=args.root,
        output_dir=args.output,
        resize=(args.width, args.height),
        grid=(args.grid_rows, args.grid_cols),
        fps=args.fps,
        normalize=not args.no_normalize,
    )

    convert_ped2(cfg, dry_run=args.dry_run)


if __name__ == "__main__":  # pragma: no cover
    main()
