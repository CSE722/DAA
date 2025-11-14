from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class FaultWindowSpec:
    probability: float
    duration_range: tuple[int, int]
    magnitude_range: tuple[float, float]


@dataclass
class FaultAugmentConfig:
    seed: int = 13
    spike: FaultWindowSpec = FaultWindowSpec(0.02, (4, 12), (1.2, 1.5))
    dropout: FaultWindowSpec = FaultWindowSpec(0.015, (3, 10), (0.05, 0.4))
    cyber: FaultWindowSpec = FaultWindowSpec(0.01, (12, 40), (0.02, 0.12))
    ped2_stats: Optional[Path] = Path("data/interim/ped2_training_features.parquet")


@dataclass
class Ped2Stats:
    grad_mean_p95: float
    diff_mean_p95: float


DEFAULT_PEDS_STATS = Ped2Stats(grad_mean_p95=0.0, diff_mean_p95=0.0)


def load_ped2_stats(path: Path | None) -> Ped2Stats:
    if path is None or not path.exists():
        return DEFAULT_PEDS_STATS
    df = pd.read_parquet(path)
    return Ped2Stats(
        grad_mean_p95=float(df.get("grad_mean", pd.Series([0])).quantile(0.95)),
        diff_mean_p95=float(df.get("frame_absdiff_mean", pd.Series([0])).quantile(0.95)),
    )


def _sample_windows(n_rows: int, spec: FaultWindowSpec, rng: np.random.Generator) -> Iterable[slice]:
    for idx in range(n_rows):
        if rng.random() < spec.probability:
            duration = rng.integers(spec.duration_range[0], spec.duration_range[1] + 1)
            yield slice(idx, min(idx + duration, n_rows))


def augment_faults(
    df: pd.DataFrame,
    load_col: str = "load_mw",
    voltage_col: str = "voltage_kv",
    frequency_col: str = "frequency_hz",
    config: FaultAugmentConfig | None = None,
) -> pd.DataFrame:
    cfg = config or FaultAugmentConfig()
    enhanced = df.copy()
    rng = np.random.default_rng(cfg.seed)
    stats = load_ped2_stats(cfg.ped2_stats)
    severity = 1.0 + stats.grad_mean_p95

    # Spikes
    for window in _sample_windows(len(enhanced), cfg.spike, rng):
        scale = rng.uniform(*cfg.spike.magnitude_range) * severity
        idx = enhanced.index[window]
        enhanced.loc[idx, load_col] *= scale
        enhanced.loc[idx, voltage_col] -= 10 * severity
        enhanced.loc[idx, "fault_flag"] = 1
        enhanced.loc[idx, "fault_type"] = "spike"

    # Dropouts
    for window in _sample_windows(len(enhanced), cfg.dropout, rng):
        scale = rng.uniform(*cfg.dropout.magnitude_range)
        idx = enhanced.index[window]
        enhanced.loc[idx, load_col] *= scale
        enhanced.loc[idx, voltage_col] -= 20
        enhanced.loc[idx, frequency_col] -= 0.05
        enhanced.loc[idx, "fault_flag"] = 1
        enhanced.loc[idx, "fault_type"] = "dropout"

    # Cyber drifts
    for window in _sample_windows(len(enhanced), cfg.cyber, rng):
        duration = window.stop - window.start
        drift = np.linspace(0, rng.uniform(*cfg.cyber.magnitude_range), duration)
        drift *= (1 + stats.diff_mean_p95)
        idx = enhanced.iloc[window].index
        enhanced.loc[idx, frequency_col] += drift
        enhanced.loc[idx, load_col] *= 1 + 0.03 * drift
        enhanced.loc[idx, "fault_flag"] = 1
        enhanced.loc[idx, "fault_type"] = "cyber"

    return enhanced
