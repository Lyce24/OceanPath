"""Canonical JSON conversion for NumPy-backed reports."""

from __future__ import annotations

from typing import Any

import numpy as np


def json_default(value: Any) -> Any:
    """Convert common NumPy values or fall back to their string form."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)
