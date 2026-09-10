"""Control producers: same artifact contract, explicit constructions.

Every control emits a ``Bridge`` through the identical affine path, so a
benchmark never has "real methods" and "special baselines" -- only
producers, candidates, and the measurement spine.

Constructions are named precisely because "null" hides too much:

- ``identity``: the no-op map, same dimensions only. Decent coarse
  geometry under it is evidence, never proof of compatibility.
- ``constant_target_centroid``: every input maps to the target anchors'
  centroid. The floor any fitted map must beat.
- ``random_map``: a seeded Gaussian matrix. Seeded, hashed, reproducible.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from relate.bridges.base import (
    Bridge,
    BridgeSpec,
    assemble_bridge,
)
from relate.model import RelateError


def identity_bridge(
    dimensions: int,
    *,
    source_space_hash: str,
    target_space_hash: str,
) -> Bridge:
    """No-op producer. Same dimensions only; no anchors, no fitting."""
    if (
        not isinstance(dimensions, int)
        or isinstance(dimensions, bool)
        or dimensions <= 0
    ):
        raise RelateError("dimensions must be a positive integer")
    spec = BridgeSpec(
        source_space_hash=source_space_hash,
        target_space_hash=target_space_hash,
        method="identity",
        params={"dimensions": dimensions},
    )
    return assemble_bridge(
        spec=spec,
        mapping=np.eye(dimensions),
        bias=np.zeros(dimensions),
        correspondence=None,
        anchor_set_hash="none",
        n_anchors=0,
    )


def constant_centroid_bridge(
    *,
    target_anchors: npt.ArrayLike,
    source_dimensions: int,
    source_space_hash: str,
    target_space_hash: str,
    anchor_set_hash: str = "none",
    coverage: dict | None = None,
) -> Bridge:
    """Every input maps to the target anchors' centroid (the floor)."""
    anchors = np.asarray(target_anchors, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[0] < 1:
        raise RelateError("target anchors must be a non-empty matrix")
    if not np.isfinite(anchors).all():
        raise RelateError("target anchors must contain only finite values")
    if (
        not isinstance(source_dimensions, int)
        or isinstance(source_dimensions, bool)
        or source_dimensions <= 0
    ):
        raise RelateError("source_dimensions must be a positive integer")
    centroid = anchors.mean(axis=0)
    spec = BridgeSpec(
        source_space_hash=source_space_hash,
        target_space_hash=target_space_hash,
        method="constant_target_centroid",
        params={},
    )
    return assemble_bridge(
        spec=spec,
        mapping=np.zeros((source_dimensions, anchors.shape[1])),
        bias=np.asarray(centroid, dtype=np.float64),
        correspondence=None,
        anchor_set_hash=anchor_set_hash,
        n_anchors=anchors.shape[0],
        coverage=coverage,
    )


def random_map_bridge(
    *,
    source_dimensions: int,
    target_dimensions: int,
    seed: int,
    source_space_hash: str,
    target_space_hash: str,
) -> Bridge:
    """Seeded Gaussian map: the chance-floor producer with provenance."""
    for name, value in (
        ("source_dimensions", source_dimensions),
        ("target_dimensions", target_dimensions),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RelateError(f"{name} must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise RelateError("seed must be an integer")
    mapping = (
        np.random.default_rng(seed).normal(size=(source_dimensions, target_dimensions))
        / np.sqrt(source_dimensions)
    )
    spec = BridgeSpec(
        source_space_hash=source_space_hash,
        target_space_hash=target_space_hash,
        method="random_map",
        params={"seed": seed},
    )
    return assemble_bridge(
        spec=spec,
        mapping=np.asarray(mapping, dtype=np.float64),
        bias=np.zeros(target_dimensions),
        correspondence=None,
        anchor_set_hash="none",
        n_anchors=0,
        seed=seed,
    )
