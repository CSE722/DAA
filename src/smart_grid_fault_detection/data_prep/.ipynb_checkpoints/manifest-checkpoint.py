from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class SourceConfig(BaseModel):
    """Configuration describing a single raw dataset."""

    name: str
    path: Path
    format: Literal["csv", "parquet"] = "csv"
    timestamp_col: str = "timestamp"
    timezone: str = "UTC"
    frequency: Optional[str] = None
    features: Optional[List[str]] = None
    dtypes: Optional[Dict[str, str]] = None
    null_markers: List[str] = Field(default_factory=lambda: ["", "NA", "NaN"])

    @field_validator("path", mode="before")
    @classmethod
    def _expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser()


class OutputConfig(BaseModel):
    """Paths for cleaned/intermediate artifacts."""

    interim_table: Path = Path("data/interim/master.parquet")
    schema_report: Path = Path("reports/data/schema.md")

    @field_validator("interim_table", "schema_report", mode="before")
    @classmethod
    def _expand_output_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser()


class DatasetManifest(BaseModel):
    """Top-level ingestion manifest with common options."""

    sources: List[SourceConfig]
    output: OutputConfig = OutputConfig()
    merge_strategy: Literal["outer", "inner", "left"] = "outer"
    sort_timestamps: bool = True

    @property
    def timestamp_col(self) -> str:
        # assume aligned timestamp column names
        if not self.sources:
            raise ValueError("Manifest must define at least one source")
        return self.sources[0].timestamp_col


def load_manifest(path: Path) -> DatasetManifest:
    """Load and validate a dataset manifest from YAML."""

    if not path.exists():
        raise FileNotFoundError(f"Manifest file '{path}' not found")

    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}

    return DatasetManifest(**raw)
