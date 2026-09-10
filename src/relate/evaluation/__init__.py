"""Evaluation package: generic science reused by every capability."""

from relate.evaluation.baselines import (
    ScoreFn,
    cosine_scorer,
    euclidean_scorer,
    random_scorer,
    relation_scorer,
    scorer_id_of,
)
from relate.evaluation.cards import EvaluationCard
from relate.evaluation.hard_negatives import (
    GroupResult,
    HardNegativeCase,
    HardNegativeDelta,
    HardNegativeObservation,
    HardNegativeReport,
    compare_reports,
    evaluate_hard_negatives,
    hard_negative_card,
)
from relate.evaluation.metrics import (
    chebyshev_distance,
    cosine_similarity,
    euclidean_distance,
    mean,
    median,
)
from relate.evaluation.preservation import (
    PreservationProfile,
    make_preservation_profile,
)

__all__ = [
    "EvaluationCard",
    "GroupResult",
    "HardNegativeCase",
    "HardNegativeDelta",
    "HardNegativeObservation",
    "HardNegativeReport",
    "PreservationProfile",
    "ScoreFn",
    "chebyshev_distance",
    "compare_reports",
    "cosine_scorer",
    "cosine_similarity",
    "euclidean_distance",
    "euclidean_scorer",
    "evaluate_hard_negatives",
    "hard_negative_card",
    "make_preservation_profile",
    "mean",
    "median",
    "random_scorer",
    "relation_scorer",
    "scorer_id_of",
]
