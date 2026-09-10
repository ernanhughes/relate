"""Fitted-method dispatcher: one entry point, no judgment inside."""

from __future__ import annotations

import numpy.typing as npt

from relate.bridges.base import Bridge, BridgeSpec
from relate.bridges.linear import fit_linear
from relate.bridges.procrustes import fit_procrustes
from relate.bridges.ridge import fit_ridge
from relate.evaluation.cross_space import CorrespondenceSet
from relate.model import RelateError


def fit_bridge(
    *,
    source_vectors: npt.ArrayLike,
    target_vectors: npt.ArrayLike,
    correspondence: CorrespondenceSet,
    spec: BridgeSpec,
    coverage: dict | None = None,
) -> Bridge:
    """Fit procrustes/linear/ridge on correspondence-resolved anchors.

    Controls (identity, constant centroid, random map) need no anchors
    and are constructed directly in :mod:`relate.bridges.controls`;
    every producer returns the identical ``Bridge`` artifact shape.
    No MLP/nonlinear producer exists: expressiveness never earned it.
    """
    if spec.method == "procrustes":
        return fit_procrustes(
            source_vectors=source_vectors,
            target_vectors=target_vectors,
            correspondence=correspondence,
            spec=spec,
            coverage=coverage,
        )
    if spec.method == "linear":
        return fit_linear(
            source_vectors=source_vectors,
            target_vectors=target_vectors,
            correspondence=correspondence,
            spec=spec,
            coverage=coverage,
        )
    if spec.method == "ridge":
        return fit_ridge(
            source_vectors=source_vectors,
            target_vectors=target_vectors,
            correspondence=correspondence,
            spec=spec,
            coverage=coverage,
        )
    raise RelateError(
        f"unknown fitted bridge method: {spec.method} "
        "(controls are constructed directly, not fitted)"
    )
