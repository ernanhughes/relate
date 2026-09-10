"""Orthogonal Procrustes producer: Y ~= XW with W'W = I.

The geometry-preserving linear control: its rigid-motion assumption is
interpretable, which is why it earns first-class status rather than being
merely one optimizer among many.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from relate.bridges.base import (
    Bridge,
    BridgeSpec,
    assemble_bridge,
    resolve_anchors,
)
from relate.evaluation.cross_space import CorrespondenceSet
from relate.model import RelateError


def fit_procrustes(
    *,
    source_vectors: npt.ArrayLike,
    target_vectors: npt.ArrayLike,
    correspondence: CorrespondenceSet,
    spec: BridgeSpec,
    coverage: dict | None = None,
) -> Bridge:
    """Orthogonal Procrustes: M = U V' from SVD(X'Y) (NumPy only)."""
    if spec.method != "procrustes":
        raise RelateError("procrustes producer needs a procrustes spec")
    source, target = resolve_anchors(source_vectors, target_vectors, correspondence)
    left, _, right_transposed = np.linalg.svd(source.T @ target, full_matrices=False)
    mapping = np.asarray(left @ right_transposed, dtype=np.float64)
    bias = np.zeros(mapping.shape[1])
    return assemble_bridge(
        spec=spec,
        mapping=mapping,
        bias=bias,
        correspondence=correspondence,
        anchor_set_hash=correspondence.content_hash,
        n_anchors=source.shape[0],
        coverage=coverage,
    )
