"""Linear / ridge / Procrustes bridge fits (NumPy only)."""

from __future__ import annotations

import numpy as np

from relate.bridges.base import Bridge
from relate.model import RelateError


def _matrices(source, target) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise RelateError("source and target anchors must be matrices")
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise RelateError("anchors need the same nonzero row count (>=2)")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RelateError("anchors must contain only finite values")
    return x, y


def fit_linear(source, target, *, source_hash="", target_hash="") -> Bridge:
    x, y = _matrices(source, target)
    mapping, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return Bridge(
        source_space_hash=source_hash,
        target_space_hash=target_hash,
        direction=f"{source_hash}->{target_hash}",
        method="linear",
        mapping=np.asarray(mapping, dtype=np.float64),
        anchor_coverage=f"n={x.shape[0]}",
    )


def fit_ridge(source, target, *, alpha=1.0, source_hash="", target_hash="") -> Bridge:
    x, y = _matrices(source, target)
    if not np.isfinite(alpha) or alpha < 0:
        raise RelateError("alpha must be finite and non-negative")
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += alpha
    try:
        mapping = np.linalg.solve(gram, x.T @ y)
    except np.linalg.LinAlgError:
        mapping = np.linalg.pinv(gram) @ (x.T @ y)
    return Bridge(
        source_space_hash=source_hash,
        target_space_hash=target_hash,
        direction=f"{source_hash}->{target_hash}",
        method="ridge",
        mapping=np.asarray(mapping, dtype=np.float64),
        anchor_coverage=f"n={x.shape[0]},alpha={alpha}",
    )


def fit_procrustes(source, target, *, source_hash="", target_hash="") -> Bridge:
    """Orthogonal Procrustes: M = U V^T from SVD(X^T Y)."""
    x, y = _matrices(source, target)
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    mapping = u @ vt
    return Bridge(
        source_space_hash=source_hash,
        target_space_hash=target_hash,
        direction=f"{source_hash}->{target_hash}",
        method="procrustes",
        mapping=np.asarray(mapping, dtype=np.float64),
        anchor_coverage=f"n={x.shape[0]}",
    )


def fit_bridge(source, target, *, method="procrustes", **kwargs) -> Bridge:
    if method == "procrustes":
        return fit_procrustes(source, target, **kwargs)
    if method == "ridge":
        return fit_ridge(source, target, **kwargs)
    if method == "linear":
        return fit_linear(source, target, **kwargs)
    raise RelateError(f"unknown bridge method: {method}")
