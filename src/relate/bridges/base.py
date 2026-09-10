"""Bridge contract: directional transformation plus provenance, never judgment.

A bridge learns ``source -> target`` parameters from anchor rows resolved
through an explicit correspondence, then produces candidate vectors via
one affine contract: ``transform(X) = X @ mapping + bias``. It records
what was fitted, on which anchors, under which code -- and nothing about
whether the result is good. Preservation, usability, and calibration
transfer are downstream measurements, never bridge fields.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from relate.evaluation.cross_space import CorrespondenceSet, aligned_matrices
from relate.model import RelateError
from relate.spaces.identity import SpaceIdentity, derive_space

VALID_STATUSES = ("ACTIVE", "EXPERIMENTAL", "DEPRECATED")
BRIDGE_METHODS = (
    "procrustes",
    "linear",
    "ridge",
    "identity",
    "constant_target_centroid",
    "random_map",
)


def code_identity() -> str:
    """Implementation identity bound into every fitted artifact."""
    try:
        version = importlib.metadata.version("relate-search")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return f"relate-search {version}"


@dataclass(frozen=True, slots=True)
class BridgeSpec:
    """What will be fitted: direction, method, and hyperparameters."""

    source_space_hash: str
    target_space_hash: str
    method: str
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_space_hash or not self.target_space_hash:
            raise RelateError("spec needs exact source and target space hashes")
        if self.method not in BRIDGE_METHODS:
            raise RelateError(f"unknown bridge method: {self.method}")


@dataclass(frozen=True, slots=True)
class BridgeFitProvenance:
    """How the artifact was fitted: anchors, shapes, code, coverage."""

    source_space_hash: str
    target_space_hash: str
    anchor_correspondence_hash: str
    n_anchors: int
    source_dimensions: int
    target_dimensions: int
    method: str
    hyperparameters: dict = field(default_factory=dict)
    seed: int | None = None
    code_identity: str = ""
    coverage: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Bridge:
    """A fitted directional artifact: parameters plus provenance."""

    bridge_id: str
    source_space_hash: str
    target_space_hash: str
    direction: str
    method: str
    mapping: np.ndarray = field(repr=False)
    bias: np.ndarray = field(repr=False)
    anchor_set_hash: str = ""
    fit_provenance: BridgeFitProvenance | None = None
    status: str = "EXPERIMENTAL"

    def __post_init__(self) -> None:
        mapping = np.asarray(self.mapping, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if mapping.ndim != 2:
            raise RelateError("bridge mapping must be a two-dimensional matrix")
        if bias.ndim != 1 or bias.shape[0] != mapping.shape[1]:
            raise RelateError("bridge bias must match mapping output width")
        if not np.isfinite(mapping).all() or not np.isfinite(bias).all():
            raise RelateError("bridge parameters must contain only finite values")
        if self.method not in BRIDGE_METHODS:
            raise RelateError(f"unknown bridge method: {self.method}")
        if self.status not in VALID_STATUSES:
            raise RelateError(f"unknown bridge status: {self.status}")
        expected = f"{self.source_space_hash}->{self.target_space_hash}"
        if self.direction != expected:
            raise RelateError("bridge direction must be source_hash->target_hash")
        if not self.bridge_id:
            raise RelateError("bridge_id must be non-empty")

    @property
    def parameter_hash(self) -> str:
        """Deterministic identity of the fitted parameters."""
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.mapping).tobytes())
        digest.update(np.ascontiguousarray(self.bias).tobytes())
        digest.update(json.dumps(list(self.mapping.shape), separators=(",", ":")).encode())
        return digest.hexdigest()[:16]

    def transform(self, vectors: npt.ArrayLike) -> np.ndarray:
        """Produce candidate vectors: input contract enforced, nothing judged.

        A bridge can return perfectly shaped garbage; detecting that is
        the measurement spine's job, not this method's.
        """
        matrix = np.asarray(vectors, dtype=np.float64)
        was_vector = matrix.ndim == 1
        if matrix.ndim not in (1, 2):
            raise RelateError("transform needs a vector or matrix")
        if was_vector:
            matrix = matrix[None, :]
        if matrix.shape[1] != self.mapping.shape[0]:
            raise RelateError(
                "input dimensions do not match the fitted bridge contract"
            )
        if not np.isfinite(matrix).all():
            raise RelateError("transform needs only finite values")
        out = matrix @ self.mapping + self.bias
        if not np.isfinite(out).all():
            raise RelateError("bridge produced non-finite candidates")
        if out.shape[1] != self.mapping.shape[1]:
            raise RelateError("bridge output width violates its contract")
        return out[0] if was_vector else out


def make_bridge_id(
    spec: BridgeSpec, parameter_hash: str, anchor_set_hash: str
) -> str:
    """Deterministic artifact identity from spec, parameters, and anchors."""
    canonical = json.dumps(
        {
            "source": spec.source_space_hash,
            "target": spec.target_space_hash,
            "method": spec.method,
            "params": spec.params,
            "parameters": parameter_hash,
            "anchors": anchor_set_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def resolve_anchors(
    source_vectors: npt.ArrayLike,
    target_vectors: npt.ArrayLike,
    correspondence: CorrespondenceSet,
) -> tuple[np.ndarray, np.ndarray]:
    """Anchor matrices in correspondence order. No implicit row alignment."""
    if not isinstance(correspondence, CorrespondenceSet):
        raise RelateError("bridge fitting needs an explicit CorrespondenceSet")
    source, target, _ = aligned_matrices(
        source_vectors, target_vectors, correspondence
    )
    if source.shape[0] < 2:
        raise RelateError("bridge fitting needs at least two anchors")
    return source, target


def assemble_bridge(
    *,
    spec: BridgeSpec,
    mapping: np.ndarray,
    bias: np.ndarray,
    correspondence: CorrespondenceSet | None,
    anchor_set_hash: str,
    n_anchors: int,
    seed: int | None = None,
    coverage: dict | None = None,
    status: str = "EXPERIMENTAL",
) -> Bridge:
    """Shared constructor: every producer emits the identical artifact shape."""
    mapping_array = np.asarray(mapping, dtype=np.float64)
    bias_array = np.asarray(bias, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mapping_array).tobytes())
    digest.update(np.ascontiguousarray(bias_array).tobytes())
    digest.update(
        json.dumps(list(mapping_array.shape), separators=(",", ":")).encode()
    )
    parameter_hash = digest.hexdigest()[:16]
    provenance = BridgeFitProvenance(
        source_space_hash=spec.source_space_hash,
        target_space_hash=spec.target_space_hash,
        anchor_correspondence_hash=(
            correspondence.content_hash if correspondence is not None else anchor_set_hash
        ),
        n_anchors=int(n_anchors),
        source_dimensions=int(mapping_array.shape[0]),
        target_dimensions=int(mapping_array.shape[1]),
        method=spec.method,
        hyperparameters=dict(spec.params),
        seed=seed,
        code_identity=code_identity(),
        coverage=dict(coverage or {}),
    )
    return Bridge(
        bridge_id=make_bridge_id(spec, parameter_hash, anchor_set_hash),
        source_space_hash=spec.source_space_hash,
        target_space_hash=spec.target_space_hash,
        direction=f"{spec.source_space_hash}->{spec.target_space_hash}",
        method=spec.method,
        mapping=mapping_array,
        bias=bias_array,
        anchor_set_hash=anchor_set_hash,
        fit_provenance=provenance,
        status=status,
    )


def bridge_output_space(
    source_space: SpaceIdentity,
    bridge: Bridge,
    target_reference: SpaceIdentity,
) -> SpaceIdentity:
    """Derived identity for transformed vectors.

    Candidate vectors live in the target's coordinate system but are NOT
    native target vectors: stamping them with the target hash would
    collapse coordinate compatibility into representation identity. The
    derived space retains the source parent, the bridge, and the target
    reference -- all three, always.
    """
    if bridge.source_space_hash != source_space.space_hash:
        raise RelateError("bridge source does not match the parent space")
    if bridge.target_space_hash != target_reference.space_hash:
        raise RelateError("bridge target does not match the reference space")
    return derive_space(
        source_space,
        "bridge",
        {
            "bridge_id": bridge.bridge_id,
            "target_reference_space_hash": target_reference.space_hash,
        },
        dimensions=int(bridge.mapping.shape[1]),
    )
