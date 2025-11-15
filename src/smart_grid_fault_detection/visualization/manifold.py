from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.console import Console
from sklearn.manifold import TSNE

console = Console()


@dataclass
class ManifoldConfig:
    input_path: Path = Path("data/processed/pca/pca_projection.parquet")
    timestamp_col: str = "timestamp"
    label_col: str = "fault_type"
    feature_cols: List[str] | None = None
    method: str = "tsne"  # "tsne" or "umap"
    perplexity: float = 30.0
    n_neighbors: int = 15
    min_dist: float = 0.1
    random_state: int = 13
    output_dir: Path = Path("reports/manifold")
    figure_dir: Path = Path("reports/figures")


def _load_dataset(cfg: ManifoldConfig) -> pd.DataFrame:
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {cfg.input_path}")
    if cfg.input_path.suffix == ".parquet":
        df = pd.read_parquet(cfg.input_path)
    else:
        df = pd.read_csv(cfg.input_path, parse_dates=[cfg.timestamp_col])
    df = df.sort_values(cfg.timestamp_col).reset_index(drop=True)
    return df


def _select_features(df: pd.DataFrame, cfg: ManifoldConfig) -> List[str]:
    if cfg.feature_cols:
        missing = [c for c in cfg.feature_cols if c not in df]
        if missing:
            raise KeyError(f"Feature columns missing: {missing}")
        return cfg.feature_cols
    pc_cols = [c for c in df.columns if c.startswith("pc")]
    if pc_cols:
        return pc_cols
    return df.select_dtypes(include=["number"]).columns.drop(cfg.label_col, errors="ignore").tolist()


def _embed_tsne(X, cfg: ManifoldConfig):
    tsne = TSNE(
        n_components=2,
        perplexity=cfg.perplexity,
        random_state=cfg.random_state,
        init="pca",
    )
    return tsne.fit_transform(X)


def _embed_umap(X, cfg: ManifoldConfig):
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "UMAP requires the optional dependency 'umap-learn'. Install via 'uv pip install umap-learn'."
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.min_dist,
        random_state=cfg.random_state,
    )
    return reducer.fit_transform(X)


def run_manifold_embedding(cfg: ManifoldConfig) -> dict[str, Path]:
    df = _load_dataset(cfg)
    features = _select_features(df, cfg)
    X = df[features].to_numpy()

    console.print(
        f"[cyan]Running {cfg.method.upper()}[/cyan] on {len(df):,} samples with {len(features)} features"
    )

    if cfg.method == "tsne":
        embedding = _embed_tsne(X, cfg)
    elif cfg.method == "umap":
        embedding = _embed_umap(X, cfg)
    else:
        raise ValueError(f"Unsupported method {cfg.method}")

    embed_df = pd.DataFrame(embedding, columns=["dim1", "dim2"])
    embed_df[cfg.timestamp_col] = df[cfg.timestamp_col].values
    if cfg.label_col in df:
        embed_df[cfg.label_col] = df[cfg.label_col].values

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)

    csv_path = cfg.output_dir / f"{cfg.method}_embedding.csv"
    embed_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 6))
    if cfg.label_col in embed_df:
        sns.scatterplot(
            data=embed_df,
            x="dim1",
            y="dim2",
            hue=cfg.label_col,
            palette="tab10",
            s=25,
            alpha=0.8,
        )
    else:
        sns.scatterplot(data=embed_df, x="dim1", y="dim2", s=25, alpha=0.8)
    plt.title(f"{cfg.method.upper()} embedding")
    plt.tight_layout()
    fig_path = cfg.figure_dir / f"{cfg.method}_embedding.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    console.print(f"[green]Saved embedding CSV:[/] {csv_path}")
    console.print(f"[green]Saved embedding figure:[/] {fig_path}")

    return {"csv": csv_path, "figure": fig_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate t-SNE/UMAP embeddings for visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=ManifoldConfig.input_path)
    parser.add_argument("--timestamp-col", type=str, default=ManifoldConfig.timestamp_col)
    parser.add_argument("--label-col", type=str, default=ManifoldConfig.label_col)
    parser.add_argument("--features", type=str, nargs="*", help="Optional feature subset to embed.")
    parser.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=ManifoldConfig.perplexity)
    parser.add_argument("--n-neighbors", type=int, default=ManifoldConfig.n_neighbors)
    parser.add_argument("--min-dist", type=float, default=ManifoldConfig.min_dist)
    parser.add_argument("--random-state", type=int, default=ManifoldConfig.random_state)
    parser.add_argument("--output-dir", type=Path, default=ManifoldConfig.output_dir)
    parser.add_argument("--figure-dir", type=Path, default=ManifoldConfig.figure_dir)
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = ManifoldConfig(
        input_path=args.input,
        timestamp_col=args.timestamp_col,
        label_col=args.label_col,
        feature_cols=args.features or None,
        method=args.method,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
    )

    run_manifold_embedding(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
