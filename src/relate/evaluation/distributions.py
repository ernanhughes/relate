"""Score distributions: the explicit inputs calibration reasons over.

Calibration belongs to a distribution, not to a model name. Random
unrelated negatives and adversarial negation/relation-swap negatives
produce radically different curves from the same scorer, so the
negative-set identity travels with the numbers -- never as a footnote.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from relate.model import RelateError

QUANTILE_KEYS = ("p5", "p25", "p50", "p75", "p95")
QUANTILE_LEVELS = (5.0, 25.0, 50.0, 75.0, 95.0)


@dataclass(frozen=True, slots=True)
class ScoreDistribution:
    """One side of a calibration comparison (positives or negatives)."""

    label: str
    n: int
    mean: float
    std: float
    quantiles: dict = field(default_factory=dict)
    scores: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.label:
            raise RelateError("label must be non-empty")


@dataclass(frozen=True, slots=True)
class NegativeSetDescriptor:
    """What the negatives were: random, mined, perturbed, and how many."""

    name: str
    kind: str
    n: int
    content_hash: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.kind:
            raise RelateError("name and kind must be non-empty")
        if (
            not isinstance(self.n, int)
            or isinstance(self.n, bool)
            or self.n <= 0
        ):
            raise RelateError("n must be a positive integer")


@dataclass(frozen=True, slots=True)
class CalibrationScope:
    """Scope is data, not branching: distinct scopes are distinct records."""

    task: str = ""
    domain: str = ""
    query_type: str = ""

    def as_dict(self) -> dict:
        return {"task": self.task, "domain": self.domain, "query_type": self.query_type}


def describe_scores(label: str, scores: list[float], *, keep_scores: bool = True) -> ScoreDistribution:
    """Summarize raw preference scores (higher = more positive)."""
    values = [float(s) for s in scores]
    if len(values) < 2:
        raise RelateError("at least two scores are required")
    if not all(math.isfinite(v) for v in values):
        raise RelateError("scores must contain only finite values")
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, np.array(QUANTILE_LEVELS) / 100.0)
    return ScoreDistribution(
        label=label,
        n=len(values),
        mean=float(array.mean()),
        std=float(array.std()),
        quantiles={key: float(value) for key, value in zip(QUANTILE_KEYS, quantiles)},
        scores=tuple(values) if keep_scores else (),
    )


def content_hash_of_scores(scores: list[float]) -> str:
    """Deterministic identity for a negative set's contents."""
    canonical = json.dumps([float(s) for s in scores], sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def roc_curve(
    positive_scores: list[float], negative_scores: list[float]
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """FAR/FRR at every distinct score threshold (accept iff score >= t).

    Returns (thresholds, fars, frrs) with thresholds descending. Higher
    score always means more positive -- the 3A preference convention, so
    distances enter negated and threshold direction cannot flip silently.
    """
    positives = [float(s) for s in positive_scores]
    negatives = [float(s) for s in negative_scores]
    if len(positives) < 1 or len(negatives) < 1:
        raise RelateError("need at least one positive and one negative score")
    if not all(math.isfinite(v) for v in positives + negatives):
        raise RelateError("scores must contain only finite values")
    thresholds = sorted(set(positives + negatives), reverse=True)
    pos = np.asarray(positives)
    neg = np.asarray(negatives)
    fars = [float(np.mean(neg >= t)) for t in thresholds]
    frrs = [float(np.mean(pos < t)) for t in thresholds]
    return tuple(thresholds), tuple(fars), tuple(frrs)


def roc_auc(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Exact P(positive > negative) + 0.5 * P(tie): global separability.

    Descriptive only -- a respectable AUC can coexist with an unusably
    large ambiguity region at the application's operating point.
    """
    positives = np.asarray([float(s) for s in positive_scores], dtype=np.float64)
    negatives = np.asarray([float(s) for s in negative_scores], dtype=np.float64)
    if positives.size < 1 or negatives.size < 1:
        raise RelateError("need at least one positive and one negative score")
    wins = np.sum(positives[:, None] > negatives[None, :])
    ties = np.sum(positives[:, None] == negatives[None, :])
    return float((wins + 0.5 * ties) / (positives.size * negatives.size))


def equal_error_point(
    thresholds: tuple[float, ...], fars: tuple[float, ...], frrs: tuple[float, ...]
) -> tuple[float, float]:
    """Threshold minimizing |FAR - FRR| and the mean rate there."""
    gaps = [abs(far - frr) for far, frr in zip(fars, frrs)]
    best = int(np.argmin(np.asarray(gaps)))
    return thresholds[best], float((fars[best] + frrs[best]) / 2.0)
