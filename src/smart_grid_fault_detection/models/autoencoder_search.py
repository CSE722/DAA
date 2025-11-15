from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List

import pandas as pd

from .autoencoder import AutoencoderConfig, run_autoencoder
from .autoencoder_eval import EvalConfig, evaluate


@dataclass
class SearchConfig:
    output_root: Path
    sequence_lengths: List[int]
    latent_dims: List[int]
    hidden_options: List[List[int]]
    dropouts: List[float]
    epochs: int = 20
    base_input: Path = Path("data/processed/smart_grid_clean.parquet")


def parse_hidden_option(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x]


def search(cfg: SearchConfig) -> Path:
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seq_len, latent, hidden, dropout in product(
        cfg.sequence_lengths, cfg.latent_dims, cfg.hidden_options, cfg.dropouts
    ):
        run_dir = cfg.output_root / f"seq{seq_len}_lat{latent}_hid{'-'.join(map(str, hidden))}_drop{dropout}".replace(".", "p")
        run_dir.mkdir(parents=True, exist_ok=True)
        auto_cfg = AutoencoderConfig(
            input_path=cfg.base_input,
            sequence_length=seq_len,
            latent_dim=latent,
            hidden_units=hidden,
            dropout=dropout,
            epochs=cfg.epochs,
            output_dir=run_dir,
        )
        outputs = run_autoencoder(auto_cfg)
        eval_cfg = EvalConfig(
            errors_path=outputs["errors"],
            metadata_path=outputs["metadata"],
            dataset_path=cfg.base_input,
            output_dir=run_dir / "eval",
        )
        eval_paths = evaluate(eval_cfg)
        summary_df = pd.read_csv(eval_paths["summary"])
        best_row = summary_df.sort_values("f1", ascending=False).iloc[0]
        run_metadata = json.loads(Path(outputs["metadata"]).read_text())
        feature_count = run_metadata.get("feature_count", 0) or 1
        compression_ratio = latent / (feature_count * seq_len)
        rows.append(
            {
                "sequence_length": seq_len,
                "latent_dim": latent,
                "hidden_units": "-".join(map(str, hidden)),
                "dropout": dropout,
                "best_percentile": best_row["percentile"],
                "best_threshold": best_row["threshold"],
                "precision": best_row["precision"],
                "recall": best_row["recall"],
                "f1": best_row["f1"],
                "compression_ratio": compression_ratio,
                "model_dir": str(run_dir),
            }
        )
    summary = pd.DataFrame(rows)
    summary.sort_values("f1", ascending=False, inplace=True)
    summary_path = cfg.output_root / "autoencoder_search_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search LSTM autoencoder architectures over sequence length, latent dim, hidden stacks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/autoencoder_search"))
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[32, 48])
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[24, 32])
    parser.add_argument("--hidden-options", type=str, nargs="+", default=["128,64", "96,48"])
    parser.add_argument("--dropouts", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--input", type=Path, default=Path("data/processed/smart_grid_clean.parquet"))
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    hidden_options = [parse_hidden_option(opt) for opt in args.hidden_options]
    cfg = SearchConfig(
        output_root=args.output_dir,
        sequence_lengths=args.sequence_lengths,
        latent_dims=args.latent_dims,
        hidden_options=hidden_options,
        dropouts=args.dropouts,
        epochs=args.epochs,
        base_input=args.input,
    )
    summary_path = search(cfg)
    print(f"Saved autoencoder search summary to {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
