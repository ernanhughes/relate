"""Small shared metric primitives (NumPy only, no sklearn).

Direction convention lives with the scorers in
:mod:`relate.evaluation.baselines`: a scorer returns a *preference* score
where higher always means more preferred, so the evaluator itself never
branches on cosine vs distance vs relation geometry.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from relate.model import RelateError


def _vector(name: str, value: npt.ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise RelateError(f"{name} must be a one-dimensional vector")
    if not np.isfinite(array).all():
        raise RelateError(f"{name} must contain only finite values")
    return array


def cosine_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Cosine similarity in [-1, 1]; higher means more similar."""
    va, vb = _vector("a", a), _vector("b", b)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        raise RelateError("cosine similarity is undefined for zero-norm vectors")
    return float(np.clip((va @ vb) / (na * nb), -1.0, 1.0))


def euclidean_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Euclidean distance; lower means more similar."""
    va, vb = _vector("a", a), _vector("b", b)
    return float(np.linalg.norm(va - vb))


def chebyshev_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Chebyshev (L-infinity) distance; lower means more similar."""
    va, vb = _vector("a", a), _vector("b", b)
    return float(np.max(np.abs(va - vb)))


def mean(values: list[float]) -> float:
    if not values:
        raise RelateError("mean of no values is undefined")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def median(values: list[float]) -> float:
    if not values:
        raise RelateError("median of no values is undefined")
    return float(np.median(np.asarray(values, dtype=np.float64)))
