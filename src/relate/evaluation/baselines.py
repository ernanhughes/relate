"""Baseline scorers: the discipline of always measuring against references.

Every scorer returns a *preference* score where higher means more preferred,
so distances are negated once here and the evaluator never branches on
geometry. No sklearn, no corpus knowledge, no bridge concepts yet.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

from relate.evaluation.metrics import (
    chebyshev_distance,
    cosine_similarity,
    euclidean_distance,
)
from relate.model import RelateError, RelationProjection

ScoreFn = Callable[[np.ndarray, np.ndarray], float]


def _tag(fn: ScoreFn, scorer_id: str) -> ScoreFn:
    fn.scorer_id = scorer_id  # type: ignore[attr-defined]
    return fn


def scorer_id_of(scorer: ScoreFn) -> str:
    """Provenance identity for a scorer; falls back to its __name__."""
    return str(getattr(scorer, "scorer_id", getattr(scorer, "__name__", "unknown")))


def cosine_scorer() -> ScoreFn:
    """Native cosine similarity; higher is more preferred."""

    def score(anchor: np.ndarray, candidate: np.ndarray) -> float:
        return cosine_similarity(anchor, candidate)

    return _tag(score, "cosine_similarity")


def euclidean_scorer() -> ScoreFn:
    """Negated native Euclidean distance; higher is more preferred."""

    def score(anchor: np.ndarray, candidate: np.ndarray) -> float:
        return -euclidean_distance(anchor, candidate)

    return _tag(score, "neg_euclidean_distance")


def random_scorer(seed: int = 0) -> ScoreFn:
    """Uniform random preference: the chance floor every method must beat."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise RelateError("seed must be an integer")
    rng = np.random.default_rng(seed)

    def score(anchor: np.ndarray, candidate: np.ndarray) -> float:
        return float(rng.uniform())

    return _tag(score, f"random_uniform:seed={seed}")


def relation_scorer(projection: RelationProjection) -> ScoreFn:
    """Negated Chebyshev distance in learned relation space.

    This is the supervised readout behind the original RELATE result: the
    same frozen embedding, a different question, a different readout.
    """

    def score(anchor: np.ndarray, candidate: np.ndarray) -> float:
        projected_anchor = np.asarray(projection.project(anchor), dtype=np.float64)
        projected_candidate = np.asarray(
            projection.project(candidate), dtype=np.float64
        )
        return -chebyshev_distance(projected_anchor, projected_candidate)

    name = ",".join(projection.relation_names)
    return _tag(score, f"relation_chebyshev:{name}")
