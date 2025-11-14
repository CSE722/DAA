"""Smart-grid fault detection package."""

from __future__ import annotations

from importlib import metadata


def _get_version() -> str:
    try:
        return metadata.version("smart-grid-fault-detection")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__version__ = _get_version()

__all__ = ["__version__"]
