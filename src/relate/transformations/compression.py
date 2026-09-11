"""Compression producers: PCA, random projection, prefix truncation.

Each producer satisfies the 5A ``VectorTransformation`` contract and
carries a ``TransformationArtifact`` (spec, parameter hash, provenance)
instead of a compression-specific identity system. Fit and transform
are distinct: PCA learns mean/components from a fit corpus; random
projection builds from a seed; prefix truncation slices with no
fitting at all.

Nothing here judges quality: explained variance, compression ratio,
and seeds are provenance metadata. Whether 32 dimensions are "enough"
is a preservation verdict, never a producer field.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

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

TRAINING_SUPPORTS = ("matryoshka", "unknown", "none")


def _parameter_hash(*arrays: np.ndarray, extra: object = None) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        digest.update(contiguous.tobytes())
        digest.update(
            json.dumps(list(contiguous.shape), separators=(",", ":")).encode()
        )
    digest.update(json.dumps(extra, sort_keys=True).encode())
    return digest.hexdigest()[:16]


def _artifact(
    *,
    kind: str,
    source_space_hash: str,
    parameters: dict,
    parameter_hash: str,
    seed: int | None,
    detail: dict,
) -> TransformationArtifact:
    if not source_space_hash:
        raise RelateError("compression needs an exact source space hash")
    spec = TransformationSpec(
        kind=kind, source_space_hash=source_space_hash, parameters=dict(parameters)
    )
    return TransformationArtifact(
        transformation_id=make_transformation_id(spec, parameter_hash),
        spec=spec,
        parameter_hash=parameter_hash,
        provenance=TransformationProvenance(
            source_space_hash=source_space_hash,
            kind=kind,
            code_identity=code_identity(),
            seed=seed,
            detail=dict(detail),
        ),
    )


@dataclass(frozen=True, slots=True)
class PCACompression:
    """PCA cartridge: center, project onto leading components."""

    artifact: TransformationArtifact
    mean: np.ndarray = field(repr=False)
    components: np.ndarray = field(repr=False)

    @property
    def transformation_id(self) -> str:
        return self.artifact.transformation_id

    @property
    def source_space_hash(self) -> str:
        return self.artifact.spec.source_space_hash

    @property
    def output_dimensions(self) -> int:
        return int(self.components.shape[0])

    @property
    def source_dimensions(self) -> int:
        return int(self.components.shape[1])

    def transform(self, values: npt.ArrayLike) -> np.ndarray:
        matrix = as_vector_transformation(values)
        was_vector = matrix.ndim == 1
        rows = matrix[None, :] if was_vector else matrix
        if rows.shape[1] != self.source_dimensions:
            raise RelateError("input dimensions do not match the PCA contract")
        out = (rows - self.mean) @ self.components.T
        if not np.isfinite(out).all():
            raise RelateError("PCA produced non-finite candidates")
        return out[0] if was_vector else out


@dataclass(frozen=True, slots=True)
class RandomProjection:
    """Seeded Gaussian map: the unstructured control with provenance."""

    artifact: TransformationArtifact
    matrix: np.ndarray = field(repr=False)

    @property
    def transformation_id(self) -> str:
        return self.artifact.transformation_id

    @property
    def source_space_hash(self) -> str:
        return self.artifact.spec.source_space_hash

    @property
    def output_dimensions(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def source_dimensions(self) -> int:
        return int(self.matrix.shape[0])

    def transform(self, values: npt.ArrayLike) -> np.ndarray:
        matrix = as_vector_transformation(values)
        was_vector = matrix.ndim == 1
        rows = matrix[None, :] if was_vector else matrix
        if rows.shape[1] != self.source_dimensions:
            raise RelateError("input dimensions do not match the projection contract")
        out = rows @ self.matrix
        if not np.isfinite(out).all():
            raise RelateError("random projection produced non-finite candidates")
        return out[0] if was_vector else out


@dataclass(frozen=True, slots=True)
class PrefixTruncation:
    """First-d coordinate slice. Named neutrally: slicing is not
    Matryoshka training unless provenance says the source earned it."""

    artifact: TransformationArtifact
    output_dimensions: int = 0

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
        if rows.shape[1] < self.output_dimensions:
            raise RelateError("input narrower than the truncation contract")
        out = np.ascontiguousarray(rows[:, : self.output_dimensions])
        return out[0] if was_vector else out


def fit_pca(
    source_vectors: npt.ArrayLike,
    *,
    output_dimensions: int,
    source_space_hash: str,
    center: bool = True,
    fit_corpus_hash: str = "",
    seed: int | None = None,
    coverage: dict | None = None,
) -> PCACompression:
    """Learn mean/components on a fit corpus; transform stays separate."""
    matrix = np.asarray(source_vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise RelateError("PCA fitting needs a matrix with at least two rows")
    if not np.isfinite(matrix).all():
        raise RelateError("PCA fitting needs only finite values")
    if (
        not isinstance(output_dimensions, int)
        or isinstance(output_dimensions, bool)
        or not 1 <= output_dimensions <= min(matrix.shape)
    ):
        raise RelateError("output_dimensions must fit inside the fit matrix")
    mean = matrix.mean(axis=0) if center else np.zeros(matrix.shape[1])
    _, singular, right = np.linalg.svd(matrix - mean, full_matrices=False)
    variance = (singular**2) / max(1, matrix.shape[0] - 1)
    total = float(variance.sum())
    if total <= 0.0:
        raise RelateError("PCA fitting needs non-constant data")
    components = np.asarray(right[:output_dimensions], dtype=np.float64)
    explained = [float(v) / total for v in variance[:output_dimensions]]
    parameter_hash = _parameter_hash(
        mean, components, extra={"output_dimensions": output_dimensions}
    )
    detail = {
        "fit_corpus_hash": fit_corpus_hash,
        "explained_variance_ratio": explained,
        "compression_ratio": output_dimensions / matrix.shape[1],
        "n_fit": matrix.shape[0],
        "center": bool(center),
        "coverage": dict(coverage or {}),
    }
    return PCACompression(
        artifact=_artifact(
            kind="pca",
            source_space_hash=source_space_hash,
            parameters={"output_dimensions": output_dimensions, "center": bool(center)},
            parameter_hash=parameter_hash,
            seed=seed,
            detail=detail,
        ),
        mean=np.asarray(mean, dtype=np.float64),
        components=components,
    )


def random_projection(
    source_dimensions: int,
    output_dimensions: int,
    *,
    seed: int,
    source_space_hash: str,
) -> RandomProjection:
    """Seeded Gaussian matrix: no fitting, full provenance."""
    for name, value in (
        ("source_dimensions", source_dimensions),
        ("output_dimensions", output_dimensions),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RelateError(f"{name} must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise RelateError("seed must be an integer")
    matrix = (
        np.random.default_rng(seed).normal(size=(source_dimensions, output_dimensions))
        / np.sqrt(output_dimensions)
    )
    parameter_hash = _parameter_hash(matrix)
    return RandomProjection(
        artifact=_artifact(
            kind="random_projection",
            source_space_hash=source_space_hash,
            parameters={"output_dimensions": output_dimensions},
            parameter_hash=parameter_hash,
            seed=seed,
            detail={"compression_ratio": output_dimensions / source_dimensions},
        ),
        matrix=np.asarray(matrix, dtype=np.float64),
    )


def prefix_truncation(
    source_dimensions: int,
    output_dimensions: int,
    *,
    source_space_hash: str,
    training_support: str = "unknown",
) -> PrefixTruncation:
    """First-d slice. ``training_support`` records Matryoshka provenance
    separately instead of baking a training claim into the method name."""
    for name, value in (
        ("source_dimensions", source_dimensions),
        ("output_dimensions", output_dimensions),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RelateError(f"{name} must be a positive integer")
    if output_dimensions > source_dimensions:
        raise RelateError("truncation cannot widen a representation")
    if training_support not in TRAINING_SUPPORTS:
        raise RelateError(f"unknown training support: {training_support}")
    parameter_hash = hash_parameters({"output_dimensions": output_dimensions})
    return PrefixTruncation(
        artifact=_artifact(
            kind="prefix_truncation",
            source_space_hash=source_space_hash,
            parameters={"output_dimensions": output_dimensions,
                        "training_support": training_support},
            parameter_hash=parameter_hash,
            seed=None,
            detail={
                "training_support": training_support,
                "compression_ratio": output_dimensions / source_dimensions,
            },
        ),
        output_dimensions=output_dimensions,
    )
