"""Relation-aware search over frozen embeddings.

The core idea is deliberately small: learn a linear projection from an existing
embedding space into externally defined relation coordinates, then search in
that relation space instead of assuming cosine similarity exposes every useful
relation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class RelateError(ValueError):
    """Raised when RELATE receives incompatible or invalid data."""


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One relation-ranked target."""

    index: int
    relation_distance: float
    cosine_distance: float


@dataclass(frozen=True, slots=True)
class RelationProjection:
    """A fitted linear projection from embeddings to relation coordinates."""

    coefficients: FloatArray
    intercept: FloatArray
    relation_median: FloatArray
    relation_scale: FloatArray
    embedding_mean: FloatArray
    alpha: float
    relation_names: tuple[str, ...]

    def __post_init__(self) -> None:
        coefficients = _matrix("coefficients", self.coefficients)
        intercept = _vector("intercept", self.intercept)
        median = _vector("relation_median", self.relation_median)
        scale = _vector("relation_scale", self.relation_scale)
        embedding_mean = _vector("embedding_mean", self.embedding_mean)

        relation_dimensions = coefficients.shape[1]
        if coefficients.shape[0] != embedding_mean.size:
            raise RelateError("coefficient and embedding dimensions differ")
        if any(value.size != relation_dimensions for value in (intercept, median, scale)):
            raise RelateError("relation parameter dimensions differ")
        if len(self.relation_names) != relation_dimensions:
            raise RelateError("relation_names must name every relation coordinate")
        if len(set(self.relation_names)) != len(self.relation_names):
            raise RelateError("relation_names must be unique")
        if np.any(scale <= 0.0):
            raise RelateError("relation_scale must be positive")
        if not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise RelateError("alpha must be finite and non-negative")

    @classmethod
    def fit(
        cls,
        embeddings: npt.ArrayLike,
        relation_coordinates: npt.ArrayLike,
        *,
        alpha: float = 1.0,
        relation_names: Sequence[str] | None = None,
        minimum_relation_scale: float = 1.0,
    ) -> "RelationProjection":
        """Fit independent ridge projections for one or more relations.

        Relation coordinates are robust-scaled using their median and
        interquartile range before fitting. ``minimum_relation_scale=1.0``
        preserves the scaling used by the original successful code result.
        """
        x = _matrix("embeddings", embeddings)
        y = _matrix("relation_coordinates", relation_coordinates)
        if x.shape[0] != y.shape[0]:
            raise RelateError("embeddings and relation coordinates need the same rows")
        if x.shape[0] < 2:
            raise RelateError("at least two training rows are required")
        if not np.isfinite(alpha) or alpha < 0.0:
            raise RelateError("alpha must be finite and non-negative")
        if not np.isfinite(minimum_relation_scale) or minimum_relation_scale <= 0.0:
            raise RelateError("minimum_relation_scale must be positive")

        names = _relation_names(relation_names, y.shape[1])
        median = np.median(y, axis=0)
        q25, q75 = np.percentile(y, (25.0, 75.0), axis=0)
        scale = np.maximum(q75 - q25, minimum_relation_scale)
        scaled_y = (y - median) / scale

        embedding_mean = x.mean(axis=0)
        output_mean = scaled_y.mean(axis=0)
        centered_x = x - embedding_mean
        centered_y = scaled_y - output_mean

        gram = centered_x.T @ centered_x
        regularized = gram.copy()
        regularized.flat[:: regularized.shape[0] + 1] += alpha
        cross = centered_x.T @ centered_y
        try:
            coefficients = np.linalg.solve(regularized, cross)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(regularized) @ cross
        intercept = output_mean - embedding_mean @ coefficients

        return cls(
            coefficients=np.asarray(coefficients, dtype=np.float64),
            intercept=np.asarray(intercept, dtype=np.float64),
            relation_median=np.asarray(median, dtype=np.float64),
            relation_scale=np.asarray(scale, dtype=np.float64),
            embedding_mean=np.asarray(embedding_mean, dtype=np.float64),
            alpha=float(alpha),
            relation_names=names,
        )

    @property
    def embedding_dimensions(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def relation_dimensions(self) -> int:
        return int(self.coefficients.shape[1])

    def project(self, embeddings: npt.ArrayLike) -> FloatArray:
        """Project one embedding or a matrix of embeddings into relation space."""
        array = np.asarray(embeddings, dtype=np.float64)
        was_vector = array.ndim == 1
        matrix = array[None, :] if was_vector else _matrix("embeddings", array)
        if matrix.ndim != 2:
            raise RelateError("embeddings must be a vector or matrix")
        if matrix.shape[1] != self.embedding_dimensions:
            raise RelateError("embedding dimensions do not match the fitted projection")
        if not np.isfinite(matrix).all():
            raise RelateError("embeddings must contain only finite values")
        projected = matrix @ self.coefficients + self.intercept
        return projected[0] if was_vector else projected

    def search(
        self,
        source_embedding: npt.ArrayLike,
        target_embeddings: npt.ArrayLike,
        *,
        k: int = 10,
        candidate_pool: int | None = None,
        exclude_index: int | None = None,
    ) -> tuple[SearchHit, ...]:
        """Rank targets by Chebyshev distance in learned relation space.

        When ``candidate_pool`` is supplied, cosine distance first selects that
        many candidates and RELATE reranks only that pool. This makes cosine a
        scalable candidate generator without letting it define the relation.
        """
        source = _vector("source_embedding", source_embedding)
        targets = _matrix("target_embeddings", target_embeddings)
        if source.size != self.embedding_dimensions or targets.shape[1] != source.size:
            raise RelateError("source, targets, and fitted projection dimensions must match")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise RelateError("k must be a positive integer")
        if targets.shape[0] == 0:
            return ()
        if exclude_index is not None and not 0 <= exclude_index < targets.shape[0]:
            raise RelateError("exclude_index is out of range")

        cosine = _cosine_distances(source, targets)
        available = np.arange(targets.shape[0], dtype=np.int64)
        if exclude_index is not None:
            available = available[available != exclude_index]
        if available.size == 0:
            return ()

        wanted = min(k, int(available.size))
        if candidate_pool is not None:
            if not isinstance(candidate_pool, int) or isinstance(candidate_pool, bool):
                raise RelateError("candidate_pool must be an integer")
            if candidate_pool < wanted:
                raise RelateError("candidate_pool must be at least k")
            pool_size = min(candidate_pool, int(available.size))
            cosine_order = np.lexsort((available, cosine[available]))
            selected = available[cosine_order[:pool_size]]
        else:
            selected = available

        source_relation = self.project(source)
        target_relations = self.project(targets[selected])
        relation = np.max(np.abs(target_relations - source_relation), axis=1)

        # Stable and deterministic: relation first, then cosine, then row index.
        order = np.lexsort((selected, cosine[selected], relation))[:wanted]
        return tuple(
            SearchHit(
                index=int(selected[position]),
                relation_distance=float(relation[position]),
                cosine_distance=float(cosine[selected[position]]),
            )
            for position in order
        )

    def save(self, path: str | Path) -> None:
        """Save the fitted projection as a pickle-free NumPy archive."""
        np.savez_compressed(
            Path(path),
            coefficients=self.coefficients,
            intercept=self.intercept,
            relation_median=self.relation_median,
            relation_scale=self.relation_scale,
            embedding_mean=self.embedding_mean,
            alpha=np.asarray([self.alpha], dtype=np.float64),
            relation_names=np.asarray(self.relation_names, dtype=np.str_),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RelationProjection":
        """Load a projection written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            alpha = np.asarray(archive["alpha"], dtype=np.float64)
            if alpha.shape != (1,):
                raise RelateError("saved alpha has an invalid shape")
            return cls(
                coefficients=np.asarray(archive["coefficients"], dtype=np.float64),
                intercept=np.asarray(archive["intercept"], dtype=np.float64),
                relation_median=np.asarray(archive["relation_median"], dtype=np.float64),
                relation_scale=np.asarray(archive["relation_scale"], dtype=np.float64),
                embedding_mean=np.asarray(archive["embedding_mean"], dtype=np.float64),
                alpha=float(alpha[0]),
                relation_names=tuple(str(value) for value in archive["relation_names"].tolist()),
            )


def _matrix(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise RelateError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(array).all():
        raise RelateError(f"{name} must contain only finite values")
    return array


def _vector(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise RelateError(f"{name} must be a one-dimensional vector")
    if not np.isfinite(array).all():
        raise RelateError(f"{name} must contain only finite values")
    return array


def _relation_names(values: Sequence[str] | None, dimensions: int) -> tuple[str, ...]:
    if values is None:
        return tuple(f"relation_{index}" for index in range(dimensions))
    names = tuple(str(value) for value in values)
    if len(names) != dimensions:
        raise RelateError("relation_names must name every relation coordinate")
    if any(not name for name in names):
        raise RelateError("relation_names cannot be empty")
    if len(set(names)) != len(names):
        raise RelateError("relation_names must be unique")
    return names


def _cosine_distances(source: FloatArray, targets: FloatArray) -> FloatArray:
    source_norm = float(np.linalg.norm(source))
    target_norms = np.linalg.norm(targets, axis=1)
    if source_norm == 0.0 or np.any(target_norms == 0.0):
        raise RelateError("cosine distance is undefined for zero-norm embeddings")
    similarity = (targets @ source) / (target_norms * source_norm)
    return 1.0 - np.clip(similarity, -1.0, 1.0)
