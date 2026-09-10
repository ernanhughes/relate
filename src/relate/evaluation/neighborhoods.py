"""Neighborhood diagnostics: stability of who stands next to whom.

Compares neighbor sets between two vector collections over the same IDs --
native vs translated, v1 vs v2, model A vs model B. No bridge logic lives
here: bridges will call these functions, never reimplement them.

Counterpart recovery (a translated point finds its own counterpart) and
neighborhood preservation (it keeps its surroundings) are separate reports
on purpose: a map can score near-perfectly on the first while failing the
second, and that split is the scientific result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from relate.evaluation.geometry import linear_cka
from relate.evaluation.hard_negatives import HardNegativeDelta
from relate.evaluation.metrics import mean, median
from relate.model import RelateError

NEIGHBORHOOD_METRICS = ("cosine", "euclidean")


@dataclass(frozen=True, slots=True)
class NeighborhoodObservation:
    item_id: str
    overlap: float
    top1_agrees: bool
    reference_neighbors: tuple = field(default_factory=tuple)
    candidate_neighbors: tuple = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class NeighborhoodReport:
    k: int
    n_queries: int
    mean_overlap: float
    median_overlap: float
    top1_agreement: float
    rank_correlation: float | None
    per_query: tuple = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CounterpartReport:
    n_queries: int
    top1: float
    recall_at_k: dict = field(default_factory=dict)
    mrr: float = 0.0


@dataclass(frozen=True, slots=True)
class GeometryComparison:
    """Coexisting measurements, never collapsed into one compatibility score."""

    reference_space_hash: str | None = None
    candidate_space_hash: str | None = None
    cosine_matrix_correlation: float | None = None
    distance_matrix_correlation: float | None = None
    neighborhood: NeighborhoodReport | None = None
    cka: float | None = None


@dataclass(frozen=True, slots=True)
class SpaceComparisonReport:
    """Evidence about two spaces. It does not decide compatibility.

    Policy (``usable_for``) is a later layer reading this report -- never a
    method on it. Measurement, then evidence, then policy.
    """

    source_space_hash: str
    target_space_hash: str
    geometry: GeometryComparison | None = None
    neighborhood: NeighborhoodReport | None = None
    counterpart: CounterpartReport | None = None
    hard_negatives: HardNegativeDelta | None = None


def _matrices(
    reference: npt.ArrayLike, candidate: npt.ArrayLike, ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.ndim != 2 or cand.ndim != 2:
        raise RelateError("reference and candidate must be matrices")
    if ref.shape[0] != cand.shape[0]:
        raise RelateError("reference and candidate need the same row count")
    if ref.shape[0] != len(ids):
        raise RelateError("ids must name every row")
    if len(set(ids)) != len(ids):
        raise RelateError("ids must be unique")
    if len(ids) < 3:
        raise RelateError("neighborhood comparison needs at least three items")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise RelateError("vectors must contain only finite values")
    return ref, cand


def _distance_matrix(matrix: np.ndarray) -> np.ndarray:
    """Euclidean distances via the Gram trick (no n-by-n-by-d temporary)."""
    gram = matrix @ matrix.T
    squared = (
        np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram
    )
    return np.sqrt(np.clip(squared, 0.0, None))


def _similarity(matrix: np.ndarray, metric: str) -> np.ndarray:
    if metric not in NEIGHBORHOOD_METRICS:
        raise RelateError(f"unknown neighborhood metric: {metric}")
    if metric == "cosine":
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0.0):
            raise RelateError("cosine neighborhoods need non-zero vectors")
        unit = matrix / norms[:, None]
        return unit @ unit.T
    return -_distance_matrix(matrix)


def _spearman(first: list[float], second: list[float]) -> float | None:
    """Spearman rank correlation (NumPy only); None when undefined."""
    if len(first) < 2:
        return None
    first_ranks = np.argsort(np.argsort(np.asarray(first, dtype=np.float64)))
    second_ranks = np.argsort(np.argsort(np.asarray(second, dtype=np.float64)))
    if np.std(first_ranks) == 0.0 or np.std(second_ranks) == 0.0:
        return None
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def compare_neighborhoods(
    reference_vectors: npt.ArrayLike,
    candidate_vectors: npt.ArrayLike,
    ids: list[str],
    *,
    k: int = 10,
    metric: str = "cosine",
    with_rank_correlation: bool = True,
) -> NeighborhoodReport:
    """Overlap, top-1 agreement, and rank stability of k-neighborhoods."""
    ref, cand = _matrices(reference_vectors, candidate_vectors, ids)
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise RelateError("k must be a positive integer")
    k = min(k, len(ids) - 1)
    ref_sim = _similarity(ref, metric)
    cand_sim = _similarity(cand, metric)
    np.fill_diagonal(ref_sim, -np.inf)
    np.fill_diagonal(cand_sim, -np.inf)
    ref_order = np.argsort(-ref_sim, axis=1)
    cand_order = np.argsort(-cand_sim, axis=1)

    observations: list[NeighborhoodObservation] = []
    correlations: list[float] = []
    for row, item_id in enumerate(ids):
        ref_top = tuple(ids[j] for j in ref_order[row, :k])
        cand_top = tuple(ids[j] for j in cand_order[row, :k])
        overlap = len(set(ref_top) & set(cand_top)) / k
        observations.append(
            NeighborhoodObservation(
                item_id=item_id,
                overlap=overlap,
                top1_agrees=bool(ref_top[0] == cand_top[0]),
                reference_neighbors=ref_top,
                candidate_neighbors=cand_top,
            )
        )
        if with_rank_correlation:
            shared = [name for name in ref_top if name in set(cand_top)]
            if len(shared) >= 2:
                ref_scores = [float(ref_sim[row, ids.index(name)]) for name in shared]
                cand_scores = [float(cand_sim[row, ids.index(name)]) for name in shared]
                value = _spearman(ref_scores, cand_scores)
                if value is not None and math.isfinite(value):
                    correlations.append(value)

    overlaps = [o.overlap for o in observations]
    return NeighborhoodReport(
        k=k,
        n_queries=len(ids),
        mean_overlap=mean(overlaps),
        median_overlap=median(overlaps),
        top1_agreement=sum(1 for o in observations if o.top1_agrees) / len(observations),
        rank_correlation=(mean(correlations) if correlations else None),
        per_query=tuple(observations),
    )


def counterpart_recovery(
    reference_vectors: npt.ArrayLike,
    candidate_vectors: npt.ArrayLike,
    ids: list[str],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> CounterpartReport:
    """For each reference row, where does its own candidate row rank?

    Cosine similarity from each reference row to every candidate row; the
    true counterpart is the same-index row. Recall@k and MRR follow.
    """
    ref, cand = _matrices(reference_vectors, candidate_vectors, ids)
    ref_norms = np.linalg.norm(ref, axis=1)
    cand_norms = np.linalg.norm(cand, axis=1)
    if np.any(ref_norms == 0.0) or np.any(cand_norms == 0.0):
        raise RelateError("counterpart recovery needs non-zero vectors")
    similarities = (ref / ref_norms[:, None]) @ (cand / cand_norms[:, None]).T
    order = np.argsort(-similarities, axis=1)
    positions = np.empty(len(ids), dtype=np.int64)
    for row in range(len(ids)):
        positions[row] = int(np.where(order[row] == row)[0][0]) + 1
    valid_ks = sorted({k for k in ks if isinstance(k, int) and not isinstance(k, bool) and k > 0})
    if not valid_ks:
        raise RelateError("ks must contain at least one positive integer")
    recall = {k: float(np.mean(positions <= k)) for k in valid_ks}
    return CounterpartReport(
        n_queries=len(ids),
        top1=float(recall[1]) if 1 in recall else float(np.mean(positions <= 1)),
        recall_at_k=recall,
        mrr=float(np.mean(1.0 / positions)),
    )


def _matrix_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    """Pearson correlation of flattened upper triangles; None if degenerate."""
    upper = np.triu_indices(first.shape[0], k=1)
    x, y = first[upper], second[upper]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def compare_geometry(
    reference_vectors: npt.ArrayLike,
    candidate_vectors: npt.ArrayLike,
    ids: list[str] | None = None,
    *,
    with_neighborhood: bool = True,
    neighborhood_k: int = 10,
    with_cka: bool = True,
    reference_space_hash: str | None = None,
    candidate_space_hash: str | None = None,
) -> GeometryComparison:
    """Pairwise-geometry agreement plus optional CKA and neighborhoods."""
    ref = np.asarray(reference_vectors, dtype=np.float64)
    cand = np.asarray(candidate_vectors, dtype=np.float64)
    if ref.shape != cand.shape:
        raise RelateError("geometry comparison needs identically shaped matrices")
    names = (
        list(ids)
        if ids is not None
        else [str(i) for i in range(ref.shape[0])]
    )
    _matrices(ref, cand, names)

    ref_norms = np.linalg.norm(ref, axis=1)
    cand_norms = np.linalg.norm(cand, axis=1)
    cosine_corr: float | None = None
    distance_corr: float | None = None
    if np.all(ref_norms > 0.0) and np.all(cand_norms > 0.0):
        cosine_corr = _matrix_correlation(
            (ref / ref_norms[:, None]) @ (ref / ref_norms[:, None]).T,
            (cand / cand_norms[:, None]) @ (cand / cand_norms[:, None]).T,
        )
    distance_corr = _matrix_correlation(
        _distance_matrix(ref),
        _distance_matrix(cand),
    )
    neighborhood = (
        compare_neighborhoods(ref, cand, names, k=neighborhood_k)
        if with_neighborhood
        else None
    )
    return GeometryComparison(
        reference_space_hash=reference_space_hash,
        candidate_space_hash=candidate_space_hash,
        cosine_matrix_correlation=cosine_corr,
        distance_matrix_correlation=distance_corr,
        neighborhood=neighborhood,
        cka=linear_cka(ref, cand) if with_cka else None,
    )
