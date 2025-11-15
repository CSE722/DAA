from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml
from rich.console import Console
from rich.table import Table

console = Console()
DEFAULT_CONFIG_PATH = Path("configs/base.yaml")


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file '{config_path}' not found. Create it or point --config elsewhere."
        )
    with config_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return data


def ensure_directories(paths_config: Dict[str, Any]) -> Iterable[Path]:
    created = []
    for _, path_str in paths_config.items():
        target = Path(path_str)
        target.mkdir(parents=True, exist_ok=True)
        created.append(target)
    return created


def summarize_config(config: Dict[str, Any]) -> None:
    project = config.get("project", {})
    data_cfg = config.get("data", {})
    dim_cfg = config.get("dimensionality_reduction", {})
    det_cfg = config.get("detectors", {})

    table = Table(title="Pipeline Summary", show_header=True, header_style="bold cyan")
    table.add_column("Section", style="bold")
    table.add_column("Key", justify="right")
    table.add_column("Value", overflow="fold")

    table.add_row("Project", "name", project.get("name", "n/a"))
    table.add_row("Project", "seed", str(project.get("seed", "n/a")))
    table.add_row(
        "Data",
        "sequence_length",
        str(data_cfg.get("sequence_length", "n/a")),
    )
    table.add_row(
        "Data",
        "train/val/test",
        "/".join(str(x) for x in data_cfg.get("train_val_test_split", [])) or "n/a",
    )
    pca_cfg = dim_cfg.get("pca", {})
    ae_cfg = dim_cfg.get("autoencoder", {})
    table.add_row(
        "Dimensionality",
        "pca_components",
        str(pca_cfg.get("n_components", "n/a")),
    )
    table.add_row(
        "Dimensionality",
        "latent_dim",
        str(ae_cfg.get("latent_dim", "n/a")),
    )
    table.add_row(
        "Detectors",
        "baseline_method",
        det_cfg.get("baseline", {}).get("method", "n/a"),
    )
    table.add_row(
        "Detectors",
        "autoencoder_threshold",
        str(det_cfg.get("autoencoder", {}).get("reconstruction_threshold", "n/a")),
    )

    console.print(table)


def run_pipeline(config: Dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        console.print("[yellow]Dry-run mode: no training/inference performed.[/yellow]")
        return

    console.print(
        "[green]Pipeline bootstrap complete.[/green] "
        "Add data loaders/models in src/ to turn this into an executable experiment."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap smart-grid anomaly detection experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and create directories without running experiments.",
    )
    parser.add_argument(
        "--create-paths",
        action="store_true",
        help="Ensure directories from the config exist before exiting.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if args.create_paths:
        ensure_directories(config.get("paths", {}))
        console.print("[cyan]Verified output/input directory tree.[/cyan]")

    summarize_config(config)
    run_pipeline(config, dry_run=args.dry_run)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
