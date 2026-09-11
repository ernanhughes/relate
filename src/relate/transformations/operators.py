"""Semantic operator hypotheses: represent and test vector maps, never text.

An operator hypothesis claims a semantic edit corresponds to a reusable
vector-space map: source embedding -> candidate target embedding. The
producers here learn such maps from paired vectors (embedding generation
stays external; no text, model, or provider call exists in this module).

Complexity is declared, never inferred::

    identity_map       0
    constant_delta     1
    linear             2
    affine             3

``identity_map`` means operator identity -- applying no vector map
sufficed under the evaluated space, task, and tolerance -- never
semantic identity of the two texts. ``NONE_PASS`` is a successful
scientific outcome: no tested rung earned preservation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt

from relate.model import RelateError, code_identity
from relate.transformations.contract import (
    TransformationArtifact,
    TransformationProvenance,
    TransformationSpec,
    as_vector_transformation,
    hash_parameters,
    make_transformation_id,
)

OPERATOR_COMPLEXITY = {
    "identity_map": 0,
    "constant_delta": 1,
    "linear": 2,
    "affine": 3,
}


class OperatorSelectionOutcome(str, Enum):
    SELECTED = "selected"
    NONE_PASS = "none_pass"


@dataclass(frozen=True, slots=True)
class ContentTransformationCase:
    """A content pair with immutable identity. No text travels here."""

    case_id: str
    source_content_hash: str
    target_content_hash: str
    relation: str
    source_id: str | None = None
    target_id: str | None = None
    group: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("case_id", "source_content_hash", "target_content_hash",
                     "relation"):
            if not getattr(self, name):
                raise RelateError(f"ContentTransformationCase.{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class OperatorSelection:
    relation: str
    outcome: OperatorSelectionOutcome
    selected_operator_id: str | None
    evaluated_operator_ids: tuple = field(default_factory=tuple)
    policy_hash: str = ""


def hash_case_set(cases: list[ContentTransformationCase]) -> str:
    """Deterministic identity for a case list (fit vs eval splits)."""
    canonical = json.dumps(
        sorted(
            (case.case_id, case.source_content_hash, case.target_content_hash,
             case.relation)
            for case in cases
        ),
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def hash_content(text: str) -> str:
    """Content identity for source/target texts (benchmark-side use)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _artifact(
    *,
    kind: str,
    relation: str,
    source_space_hash: str,
    parameters: dict,
    parameter_hash: str,
    train_case_set_hash: str,
    n_train_pairs: int,
) -> TransformationArtifact:
    if not source_space_hash:
        raise RelateError("operator needs an exact source space hash")
    if not relation:
        raise RelateError("operator needs the trained relation")
    spec = TransformationSpec(
        kind=kind,
        source_space_hash=source_space_hash,
        parameters={"relation": relation, **dict(parameters)},
    )
    return TransformationArtifact(
        transformation_id=make_transformation_id(spec, parameter_hash),
        spec=spec,
        parameter_hash=parameter_hash,
        provenance=TransformationProvenance(
            source_space_hash=source_space_hash,
            kind=kind,
            code_identity=code_identity(),
            seed=None,
            detail={
                "relation": relation,
                "train_case_set_hash": train_case_set_hash,
                "n_train_pairs": int(n_train_pairs),
            },
        ),
    )


def _parameter_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        digest.update(contiguous.tobytes())
        digest.update(
            json.dumps(list(contiguous.shape), separators=(",", ":")).encode()
        )
    return digest.hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class IdentityMap:
    """Operator identity: apply no map. Sufficient is not synonymous."""

    artifact: TransformationArtifact

    @property
    def transformation_id(self) -> str:
        return self.artifact.transformation_id

    @property
    def source_space_hash(self) -> str:
        return self.artifact.spec.source_space_hash

    def transform(self, values: npt.ArrayLike) -> np.ndarray:
        matrix = as_vector_transformation(values)
        return np.asarray(matrix, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ConstantDelta:
    """One reusable direction: candidate = source + delta."""

    artifact: TransformationArtifact
    delta: np.ndarray = field(repr=False)

    @property
    def transformation_id(self) -> str:
        return self.artifact.transformation_id

    @property
    def source_space_hash(self) -> str:
        return self.artifact.spec.source_space_hash

    def transform(self, values: npt.ArrayLike) -> np.ndarray:
        matrix = as_vector_transformation(values)
        was_vector = matrix.ndim == 1
        rows = matrix[None, :] if was_vector else matrix
        if rows.shape[1] != self.delta.shape[0]:
            raise RelateError("input dimensions do not match the delta contract")
        out = rows + self.delta
        if not np.isfinite(out).all():
            raise RelateError("constant delta produced non-finite candidates")
        return out[0] if was_vector else out


@dataclass(frozen=True, slots=True)
class LinearOperator:
    """Candidate = XW, learned map without intercept."""

    artifact: TransformationArtifact
    mapping: np.ndarray = field(repr=False)

    @property
    def transformation_id(self) -> str:
        return self.artifact.transformation_id

    @property
    def source_space_hash(self) -> str:
        return self.artifact.spec.source_space_hash

    def transform(self, values: npt.ArrayLike) -> np.ndarray:
        matrix = as_vector_transformation(values)
        was_vector = matrix.ndim == 1
        rows = matrix[None, :] if was_vector else matrix
        if rows.shape[1] != self.mapping.shape[0]:
            raise RelateError("input dimensions do not match the linear contract")
        out = rows @ self.mapping
        if not np.isfinite(out).all():
            raise RelateError("linear operator produced non-finite candidates")
        return out[0] if was_vector else out


@dataclass(frozen=True, slots=True)
class AffineOperator:
    """Candidate = XW + b, the top of the supported ladder."""

    artifact: TransformationArtifact
    mapping: np.ndarray = field(repr=False)
    bias: np.ndarray = field(repr=False)

    @property
    def transformation_id(self) -> str:
        return self.artifact.transformation_id

    @property
    def source_space_hash(self) -> str:
        return self.artifact.spec.source_space_hash

    def transform(self, values: npt.ArrayLike) -> np.ndarray:
        matrix = as_vector_transformation(values)
        was_vector = matrix.ndim == 1
        rows = matrix[None, :] if was_vector else matrix
        if rows.shape[1] != self.mapping.shape[0]:
            raise RelateError("input dimensions do not match the affine contract")
        out = rows @ self.mapping + self.bias
        if not np.isfinite(out).all():
            raise RelateError("affine operator produced non-finite candidates")
        return out[0] if was_vector else out


def identity_map(
    *,
    relation: str,
    source_space_hash: str,
    train_case_set_hash: str = "",
) -> IdentityMap:
    """Nothing to learn: the rung everything else must beat on simplicity."""
    return IdentityMap(
        artifact=_artifact(
            kind="identity_map",
            relation=relation,
            source_space_hash=source_space_hash,
            parameters={},
            parameter_hash=hash_parameters({"operator": "identity_map"}),
            train_case_set_hash=train_case_set_hash,
            n_train_pairs=0,
        )
    )


def _anchor_matrices(source_anchors, target_anchors) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_anchors, dtype=np.float64)
    target = np.asarray(target_anchors, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2:
        raise RelateError("operator fitting needs anchor matrices")
    if source.shape[0] != target.shape[0] or source.shape[0] < 2:
        raise RelateError("operator fitting needs aligned pairs (>=2)")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise RelateError("operator fitting needs only finite values")
    return source, target


def fit_constant_delta(
    source_anchors: npt.ArrayLike,
    target_anchors: npt.ArrayLike,
    *,
    relation: str,
    source_space_hash: str,
    train_case_set_hash: str = "",
) -> ConstantDelta:
    """Delta = mean(target - source): is the edit one direction?"""
    source, target = _anchor_matrices(source_anchors, target_anchors)
    delta = np.asarray((target - source).mean(axis=0), dtype=np.float64)
    return ConstantDelta(
        artifact=_artifact(
            kind="constant_delta",
            relation=relation,
            source_space_hash=source_space_hash,
            parameters={},
            parameter_hash=_parameter_hash(delta),
            train_case_set_hash=train_case_set_hash,
            n_train_pairs=source.shape[0],
        ),
        delta=delta,
    )


def fit_linear_operator(
    source_anchors: npt.ArrayLike,
    target_anchors: npt.ArrayLike,
    *,
    relation: str,
    source_space_hash: str,
    train_case_set_hash: str = "",
) -> LinearOperator:
    """Least-squares map without intercept (NumPy only)."""
    source, target = _anchor_matrices(source_anchors, target_anchors)
    mapping, _, _, _ = np.linalg.lstsq(source, target, rcond=None)
    mapping = np.asarray(mapping, dtype=np.float64)
    return LinearOperator(
        artifact=_artifact(
            kind="linear",
            relation=relation,
            source_space_hash=source_space_hash,
            parameters={},
            parameter_hash=_parameter_hash(mapping),
            train_case_set_hash=train_case_set_hash,
            n_train_pairs=source.shape[0],
        ),
        mapping=mapping,
    )


def fit_affine_operator(
    source_anchors: npt.ArrayLike,
    target_anchors: npt.ArrayLike,
    *,
    relation: str,
    source_space_hash: str,
    train_case_set_hash: str = "",
) -> AffineOperator:
    """Least-squares map with intercept (NumPy only)."""
    source, target = _anchor_matrices(source_anchors, target_anchors)
    augmented = np.hstack([source, np.ones((source.shape[0], 1))])
    solution, _, _, _ = np.linalg.lstsq(augmented, target, rcond=None)
    mapping = np.asarray(solution[:-1], dtype=np.float64)
    bias = np.asarray(solution[-1], dtype=np.float64)
    return AffineOperator(
        artifact=_artifact(
            kind="affine",
            relation=relation,
            source_space_hash=source_space_hash,
            parameters={},
            parameter_hash=_parameter_hash(mapping, bias),
            train_case_set_hash=train_case_set_hash,
            n_train_pairs=source.shape[0],
        ),
        mapping=mapping,
        bias=bias,
    )


def select_simplest_passing(
    candidates: list[tuple[int, str, object]],
    *,
    scope: str,
    policy_hash: str = "",
) -> OperatorSelection:
    """Lowest-complexity profile passing ``scope``; verdicts only.

    ``candidates`` carries (complexity, operator_id, PreservationProfile).
    No reconstruction loss is compared here -- by the time selection
    runs, measurement has already spoken through verdicts.
    """
    if not scope:
        raise RelateError("selection needs a scope")
    evaluated = sorted({operator_id for _, operator_id, _ in candidates})
    passing = sorted(
        (complexity, operator_id)
        for complexity, operator_id, profile in candidates
        if profile.usable_for(scope)
    )
    if not passing:
        return OperatorSelection(
            relation="",
            outcome=OperatorSelectionOutcome.NONE_PASS,
            selected_operator_id=None,
            evaluated_operator_ids=tuple(evaluated),
            policy_hash=policy_hash,
        )
    _, selected = passing[0]
    return OperatorSelection(
        relation="",
        outcome=OperatorSelectionOutcome.SELECTED,
        selected_operator_id=selected,
        evaluated_operator_ids=tuple(evaluated),
        policy_hash=policy_hash,
    )
