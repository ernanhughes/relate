"""Ridge bridge producer: regularized supervised linear map."""

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


def fit_ridge(
    *,
    source_vectors: npt.ArrayLike,
    target_vectors: npt.ArrayLike,
    correspondence: CorrespondenceSet,
    spec: BridgeSpec,
    coverage: dict | None = None,
) -> Bridge:
    """Ridge map on correspondence-resolved anchors (NumPy only)."""
    if spec.method != "ridge":
        raise RelateError("ridge producer needs a ridge spec")
    try:
        alpha = float(spec.params.get("alpha", 1.0))
    except (TypeError, ValueError) as error:
        raise RelateError("ridge alpha must be numeric") from error
    if not np.isfinite(alpha) or alpha < 0.0:
        raise RelateError("ridge alpha must be finite and non-negative")
    source, target = resolve_anchors(source_vectors, target_vectors, correspondence)
    gram = source.T @ source
    gram.flat[:: gram.shape[0] + 1] += alpha
    try:
        mapping = np.linalg.solve(gram, source.T @ target)
    except np.linalg.LinAlgError:
        mapping = np.linalg.pinv(gram) @ (source.T @ target)
    mapping = np.asarray(mapping, dtype=np.float64)
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
