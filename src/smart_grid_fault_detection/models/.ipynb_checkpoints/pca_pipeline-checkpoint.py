from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import dump
from rich.console import Console
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

console = Console()


@dataclass
class PCAConfig:
    input_path: Path = Path("data/processed/smart_grid_clean.parquet")
    output_dir: Path = Path("data/processed/pca")
    timestamp_col: str = "timestamp"
    feature_list: Optional[List[str]] = None
    drop_columns: List[str] = field(default_factory=lambda: ["fault_flag"])
    scaler: str = "standard"  # "standard", "minmax", "none"
    n_components: Optional[int] = None
    variance_threshold: Optional[float] = 0.95
    random_state: int = 13


def _load_frame(path: Path, timestamp_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, parse_dates=[timestamp_col])
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    if timestamp_col not in df:
        raise KeyError(f"Timestamp column '{timestamp_col}' missing from dataset")
    return df


def _choose_features(df: pd.DataFrame, cfg: PCAConfig) -> List[str]:
    candidates = cfg.feature_list if cfg.feature_list else df.select_dtypes(include=["number"]).columns.tolist()
    drop = set(cfg.drop_columns + [cfg.timestamp_col])
    return [c for c in candidates if c not in drop]


def _build_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "none" or name is None:
        return None
    raise ValueError(f"Unsupported scaler '{name}'")


def run_pca_pipeline(cfg: PCAConfig) -> dict[str, Path | dict[str, float]]:
    df = _load_frame(cfg.input_path, cfg.timestamp_col)
    feature_cols = _choose_features(df, cfg)
    if not feature_cols:
        raise ValueError("No numeric features available for PCA.")

    console.print(
        f"[cyan]PCA source:[/] {cfg.input_path} | rows={len(df):,} | features={len(feature_cols)}"
    )

    X = df[feature_cols].to_numpy(dtype=np.float32)

    scaler = _build_scaler(cfg.scaler)
    if scaler is not None:
        X = scaler.fit_transform(X)

    if cfg.n_components is not None:
        pca = PCA(n_components=cfg.n_components, random_state=cfg.random_state)
    elif cfg.variance_threshold is not None:
        pca = PCA(n_components=cfg.variance_threshold, svd_solver="full", random_state=cfg.random_state)
    else:
        pca = PCA(random_state=cfg.random_state)

    transformed = pca.fit_transform(X)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    components_df = pd.DataFrame(
        pca.components_, columns=feature_cols, index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    components_path = cfg.output_dir / "components.parquet"
    components_df.to_parquet(components_path)

    transformed_df = pd.DataFrame(transformed, columns=[f"pc{i+1}" for i in range(pca.n_components_)])
    transformed_df.insert(0, cfg.timestamp_col, df[cfg.timestamp_col].values)
    transformed_df[cfg.timestamp_col] = pd.to_datetime(transformed_df[cfg.timestamp_col])
    transformed_path = cfg.output_dir / "pca_projection.parquet"
    transformed_df.to_parquet(transformed_path, index=False)

    variance = pca.explained_variance_ratio_
    cumulative = variance.cumsum()
    variance_df = pd.DataFrame(
        {
            "component": [f"pc{i+1}" for i in range(len(variance))],
            "variance_ratio": variance,
            "cumulative": cumulative,
        }
    )
    variance_path = cfg.output_dir / "variance.csv"
    variance_df.to_csv(variance_path, index=False)

    model_path = cfg.output_dir / "pca_model.joblib"
    dump({"scaler": scaler, "pca": pca, "features": feature_cols}, model_path)

    metadata = {
        "rows": len(df),
        "feature_count": len(feature_cols),
        "timestamp_col": cfg.timestamp_col,
        "scaler": cfg.scaler,
        "n_components": int(pca.n_components_),
        "explained_variance": float(cumulative[-1]),
    }
    metadata_path = cfg.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    console.print(f"[green]Saved PCA components:[/] {components_path}")
    console.print(f"[green]Saved transformed data:[/] {transformed_path}")
    console.print(f"[green]Saved variance stats:[/] {variance_path}")
    console.print(f"[green]Serialized PCA model:[/] {model_path}")

    return {
        "components": components_path,
        "transformed": transformed_path,
        "variance": variance_path,
        "model": model_path,
        "metadata": metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit PCA on processed smart-grid features and export projections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=PCAConfig.input_path)
    parser.add_argument("--output", type=Path, default=PCAConfig.output_dir)
    parser.add_argument("--timestamp-col", type=str, default=PCAConfig.timestamp_col)
    parser.add_argument("--features", type=str, nargs="*", help="Optional feature subset")
    parser.add_argument("--drop-columns", type=str, nargs="*", default=None)
    parser.add_argument("--scaler", choices=["standard", "minmax", "none"], default="standard")
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.95,
        help="If set and n-components is None, retain enough components to reach this cumulative variance.",
    )
    parser.add_argument("--random-state", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = PCAConfig(
        input_path=args.input,
        output_dir=args.output,
        timestamp_col=args.timestamp_col,
        feature_list=args.features,
        drop_columns=args.drop_columns or PCAConfig().drop_columns,
        scaler=args.scaler,
        n_components=args.n_components,
        variance_threshold=args.variance_threshold if args.n_components is None else None,
        random_state=args.random_state,
    )

    run_pca_pipeline(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
