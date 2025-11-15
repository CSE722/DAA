from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.metrics import (auc, average_precision_score, f1_score,
                             precision_score, recall_score, roc_curve)

console = Console()


@dataclass
class EvalConfig:
    errors_path: Path = Path("models/autoencoder_baseline/reconstruction_errors.csv")
    metadata_path: Path = Path("models/autoencoder_baseline/metadata.json")
    dataset_path: Path = Path("data/processed/smart_grid_clean.parquet")
    label_col: str = "fault_flag"
    percentiles: Iterable[float] = (80, 85, 90, 95, 97.5, 99)
    output_dir: Path = Path("reports/autoencoder")


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return json.loads(path.read_text())


def _align_labels(cfg: EvalConfig, seq_len: int, n_windows: int) -> np.ndarray:
    if not cfg.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cfg.dataset_path}")
    df = pd.read_parquet(cfg.dataset_path)
    labels = df[cfg.label_col].to_numpy(dtype=int)
    start = seq_len - 1
    end = start + n_windows
    if end > len(labels):
        raise ValueError("Dataset shorter than expected for autoencoder windows")
    return labels[start:end]


def evaluate(cfg: EvalConfig) -> dict[str, Path]:
    if not cfg.errors_path.exists():
        raise FileNotFoundError(f"Reconstruction errors not found: {cfg.errors_path}")

    errors_df = pd.read_csv(cfg.errors_path)
    recon = errors_df["reconstruction_error"].to_numpy()
    metadata = _load_metadata(cfg.metadata_path)
    seq_len = int(metadata["sequence_length"])
    labels = _align_labels(cfg, seq_len, len(errors_df))

    roc_fpr, roc_tpr, roc_thresh = roc_curve(labels, recon)
    roc_auc = auc(roc_fpr, roc_tpr)
    pr_auc = average_precision_score(labels, recon)

    summary_rows = []
    for perc in cfg.percentiles:
        thresh = float(np.percentile(recon, perc))
        preds = (recon >= thresh).astype(int)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        summary_rows.append(
            {
                "percentile": perc,
                "threshold": thresh,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.output_dir / "autoencoder_threshold_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    roc_df = pd.DataFrame({"fpr": roc_fpr, "tpr": roc_tpr, "threshold": np.r_[roc_thresh, np.nan][: len(roc_fpr)]})
    roc_path = cfg.output_dir / "autoencoder_roc.csv"
    roc_df.to_csv(roc_path, index=False)

    pr_path = cfg.output_dir / "autoencoder_pr_auc.json"
    pr_path.write_text(json.dumps({"roc_auc": roc_auc, "pr_auc": pr_auc}, indent=2))

    console.print(
        "[green]Autoencoder evaluation saved[/green] "
        f"(ROC AUC={roc_auc:.3f}, PR AUC={pr_auc:.3f})"
    )

    return {"summary": summary_path, "roc": roc_path, "pr": pr_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LSTM autoencoder reconstruction errors across thresholds and curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--errors", type=Path, default=EvalConfig.errors_path)
    parser.add_argument("--metadata", type=Path, default=EvalConfig.metadata_path)
    parser.add_argument("--dataset", type=Path, default=EvalConfig.dataset_path)
    parser.add_argument("--label-col", type=str, default=EvalConfig.label_col)
    parser.add_argument("--percentiles", type=float, nargs="+", default=list(EvalConfig.percentiles))
    parser.add_argument("--output-dir", type=Path, default=EvalConfig.output_dir)
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = EvalConfig(
        errors_path=args.errors,
        metadata_path=args.metadata,
        dataset_path=args.dataset,
        label_col=args.label_col,
        percentiles=args.percentiles,
        output_dir=args.output_dir,
    )

    evaluate(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
