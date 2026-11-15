"""Dataset transformation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def convert_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: convert_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert_numpy(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
