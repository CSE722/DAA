from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


@dataclass
class GapFillReport:
    """Summary of reindexing/imputation operations."""

    expected_points: int
    missing_before: int
    filled_points: int
    remaining_na: int


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if not is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, utc=True)
    else:
        if getattr(series.dt, "tz", None) is None:
            series = series.dt.tz_localize("UTC")
        else:
            series = series.dt.tz_convert("UTC")
    return series


def regularize_time_index(
    df: pd.DataFrame,
    timestamp_col: str,
    freq: str,
    interpolation: str = "linear",
    limit: Optional[int] = None,
) -> tuple[pd.DataFrame, GapFillReport]:
    """Reindex a dataframe to a fixed frequency and interpolate missing rows."""

    timestamp = _ensure_datetime(df[timestamp_col])
    ordered = df.copy()
    ordered[timestamp_col] = timestamp
    ordered = ordered.sort_values(timestamp_col)
    ordered = ordered.set_index(timestamp_col)

    full_index = pd.date_range(ordered.index.min(), ordered.index.max(), freq=freq, tz="UTC")
    missing_before = len(full_index.difference(ordered.index))
    reindexed = ordered.reindex(full_index)
    numeric_cols = reindexed.select_dtypes(include=["number"]).columns
    na_before = int(reindexed[numeric_cols].isna().sum().sum())
    reindexed[numeric_cols] = reindexed[numeric_cols].interpolate(
        method=interpolation, limit=limit
    ).ffill().bfill()
    na_after = int(reindexed.isna().sum().sum())

    report = GapFillReport(
        expected_points=len(full_index),
        missing_before=missing_before,
        filled_points=max(na_before - na_after, 0),
        remaining_na=na_after,
    )

    reindexed = reindexed.reset_index().rename(columns={"index": timestamp_col})
    return reindexed, report


def clip_zscore_outliers(
    df: pd.DataFrame,
    columns: Iterable[str],
    threshold: float = 3.5,
) -> pd.DataFrame:
    """Winsorize values exceeding a z-score threshold."""

    cleaned = df.copy()
    for col in columns:
        if col not in cleaned:
            continue
        series = cleaned[col]
        mean = series.mean()
        std = series.std() or 1.0
        zscores = (series - mean) / std
        high_mask = zscores > threshold
        low_mask = zscores < -threshold
        cleaned.loc[high_mask, col] = mean + threshold * std
        cleaned.loc[low_mask, col] = mean - threshold * std
    return cleaned


def impute_columns(
    df: pd.DataFrame,
    strategy: str = "ffill",
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Apply a simple imputation strategy over selected columns."""

    columns = list(columns) if columns else df.columns
    filled = df.copy()
    for col in columns:
        if col not in filled:
            continue
        if strategy == "ffill":
            filled[col] = filled[col].ffill()
        elif strategy == "bfill":
            filled[col] = filled[col].bfill()
        elif strategy == "zero":
            filled[col] = filled[col].fillna(0)
        else:
            filled[col] = filled[col].fillna(filled[col].median())
    return filled
