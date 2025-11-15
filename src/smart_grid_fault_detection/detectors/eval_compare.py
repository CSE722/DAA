from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from rich.console import Console
from sklearn.metrics import accuracy_score, classification_report

console = Console()


@dataclass
class DetectorSpec:
    name: str
    path: Path
    label_col: str = "fault_flag"
    pred_col: str = "prediction"


def parse_spec(text: str) -> DetectorSpec:
    parts = dict(part.split("=", 1) for part in text.split(","))
    return DetectorSpec(
        name=parts["name"],
        path=Path(parts["path"]),
        label_col=parts.get("label", "fault_flag"),
        pred_col=parts.get("pred", "prediction"),
    )


def summarize(specs: List[DetectorSpec], output: Path) -> Path:
    rows = []
    for spec in specs:
        if not spec.path.exists():
            raise FileNotFoundError(f"Predictions file missing for {spec.name}: {spec.path}")
        df = pd.read_csv(spec.path)
        if spec.label_col not in df.columns or spec.pred_col not in df.columns:
            raise KeyError(f"Columns {spec.label_col}/{spec.pred_col} not found in {spec.path}")
        y_true = df[spec.label_col]
        y_pred = df[spec.pred_col]
        labels = sorted(pd.unique(y_true))
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        rows.append(
            {
                "detector": spec.name,
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
                "support": report["macro avg"]["support"],
                "predictions_path": str(spec.path),
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = output / "detector_comparison.csv"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare detector predictions files and compute macro metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--detector",
        type=str,
        action="append",
        required=True,
        help="Detector spec entries: name=... , path=... , label=<col>, pred=<col>",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/detector_compare"))
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    specs = [parse_spec(text) for text in args.detector]
    summary_path = summarize(specs, args.output)
    console.print(f"[green]Saved detector comparison to[/green] {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
