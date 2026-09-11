"""Generic transformation contract: identity, spec, artifact, derivation.

A transformation takes an identified representation, produces a new
derived representation, and incurs an obligation to measure preservation
against an explicit authority. Bridges satisfy this contract structurally
(a read-only ``transformation_id`` over ``bridge_id``); no state is
duplicated and no bridge API changes.

Two producer kinds stay distinguishable by construction:

- ``VectorTransformation`` acts directly on vectors (bridge, PCA,
  projection): ``values -> values``.
- ``ContentTransformation`` acts upstream (rewrite, compression plan):
  ``content -> content' -> embed -> values``. It deliberately exposes
  no vector ``transform``; its outputs are evaluated after embedding.

This module computes no metric: identity and derivation only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from relate.model import RelateError
from relate.spaces.identity import SpaceIdentity, derive_space


@dataclass(frozen=True, slots=True)
class TransformationSpec:
    """What will be built: kind, source identity, and parameters."""

    kind: str
    source_space_hash: str
    parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise RelateError("transformation kind must be non-empty")
        if not self.source_space_hash:
            raise RelateError("spec needs an exact source space hash")


@dataclass(frozen=True, slots=True)
class TransformationProvenance:
    """How the artifact came to be: code, seed, and free-form detail."""

    source_space_hash: str = ""
    kind: str = ""
    code_identity: str = ""
    seed: int | None = None
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransformationArtifact:
    """A persisted transformation: spec plus deterministic identity."""

    transformation_id: str
    spec: TransformationSpec
    parameter_hash: str
    provenance: TransformationProvenance | None = None

    def __post_init__(self) -> None:
        if not self.transformation_id:
            raise RelateError("transformation_id must be non-empty")
        if not self.parameter_hash:
            raise RelateError("parameter_hash must be non-empty")


def make_transformation_id(spec: TransformationSpec, parameter_hash: str) -> str:
    """Deterministic artifact identity from spec and parameters."""
    canonical = json.dumps(
        {
            "kind": spec.kind,
            "source": spec.source_space_hash,
            "parameters": spec.parameters,
            "parameter_hash": parameter_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def hash_parameters(payload: object) -> str:
    """Deterministic hash for JSON-canonicalizable parameters."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@runtime_checkable
class VectorTransformation(Protocol):
    """Structural contract for vector-to-vector producers."""

    transformation_id: str
    source_space_hash: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Produce candidate vectors. Shape-checked, never judged."""
        ...


@runtime_checkable
class ContentTransformation(Protocol):
    """Structural contract for content-upstream producers.

    No vector ``transform`` exists here on purpose: a rewrite acts on
    content, and only the embedded outputs enter measurement.
    """

    transformation_id: str
    source_space_hash: str
    content_kind: str


def derived_transformation_space(
    parent: SpaceIdentity,
    transformation_id: str,
    parameters: dict | None,
    *,
    dimensions: int | None = None,
    kind: str = "transformation",
) -> SpaceIdentity:
    """New derived identity for transformed outputs.

    Generic transformations carry no target reference (bridges keep
    their own ``bridge_output_space`` with one). The parent, the
    transformation identity, and its parameters always travel along;
    the result never inherits the parent hash.
    """
    if not transformation_id:
        raise RelateError("transformation_id must be non-empty")
    derived = derive_space(
        parent,
        kind,
        {"transformation_id": transformation_id, "parameters": parameters or {}},
        dimensions=dimensions,
    )
    if derived.space_hash == parent.space_hash:
        raise RelateError("derived transformation space collides with its parent")
    return derived


def as_vector_transformation(values: npt.ArrayLike) -> np.ndarray:
    """Shared input validation for vector producers (finite 1-D/2-D)."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim not in (1, 2):
        raise RelateError("transformation needs a vector or matrix")
    if not np.isfinite(matrix).all():
        raise RelateError("transformation needs only finite values")
    return matrix
