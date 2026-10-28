"""Shape-preserving JSON form of numpy arrays.

``ndarray.tolist()`` loses the shape of an empty array (a ``(0, 8)`` matrix comes
back as ``(0,)``), which breaks a model with no numeric parameters. Persisted
arrays therefore carry their shape.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def array_to_json(array: np.ndarray) -> Dict[str, Any]:
    """Return *array* as ``{"shape": [...], "data": [...]}``."""
    return {"shape": list(array.shape), "data": array.ravel().tolist()}


def array_from_json(raw: Dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`array_to_json`."""
    return np.asarray(raw["data"], dtype=float).reshape(raw["shape"])


__all__ = ["array_from_json", "array_to_json"]
