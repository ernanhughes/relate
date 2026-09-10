"""Calibration measurement: distributions become operating points.

Works over arbitrary preference scores (higher = more positive) -- cosine,
relation readouts, translated-space scores, cross-encoder scores -- exactly
like the 3A evaluator. AUC is reported as global separability; the
two-threshold operating point is the application decision. A respectable
AUC with an unusably large ambiguity region is the expected hard-negative
outcome, not a contradiction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from relate.evaluation.cards import EvaluationCard
from relate.evaluation.distributions import (
    CalibrationScope,
    NegativeSetDescriptor,
    ScoreDistribution,
    describe_scores,
    equal_error_point,
    roc_auc,
    roc_curve,
)
from relate.evaluation.hard_negatives import HardNegativeReport
from relate.model import RelateError

if TYPE_CHECKING:
    from relate.retrieval.calibration import CalibrationRecord


class CalibrationDecision(str, Enum):
    """Three-way outcome. No ``is_match`` shortcut: the ambiguity region stays."""

    ACCEPT = "accept"
    REJECT = "reject"
    ESCALATE = "escalate"


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    threshold_low: float
    threshold_high: float
    target: str
    false_accept_rate: float
    false_reject_rate: float
    ambiguity_fraction: float

    def __post_init__(self) -> None:
        if self.threshold_low > self.threshold_high:
            raise RelateError("threshold_low must not exceed threshold_high")


@dataclass(frozen=True, slots=True)
class CalibrationStaleness:
    stale: bool
    changed: tuple = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class PairedScoreDistributions:
    positive: tuple = field(default_factory=tuple)
    negative: tuple = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class CalibrationFit:
    positive: ScoreDistribution | None = None
    negative: ScoreDistribution | None = None
    thresholds: tuple = field(default_factory=tuple)
    fars: tuple = field(default_factory=tuple)
    frrs: tuple = field(default_factory=tuple)
    auc: float = 0.5
    eer_threshold: float = 0.0
    eer_rate: float = 0.0
    operating_point: OperatingPoint | None = None
    record: CalibrationRecord | None = None


def extract_margin_distributions(
    report: HardNegativeReport,
) -> PairedScoreDistributions:
    """Feed 3A observations into calibration without recomputing margins."""
    if not report.observations:
        raise RelateError("report contains no observations")
    return PairedScoreDistributions(
        positive=tuple(o.positive_score for o in report.observations),
        negative=tuple(o.negative_score for o in report.observations),
    )


def calibrate(
    positive_scores: list[float],
    negative_scores: list[float],
    *,
    far_target: float = 0.10,
    frr_target: float = 0.10,
    objective: str = "bounded_rates",
    negative_set: NegativeSetDescriptor | None = None,
    positive_set: str = "",
    space_hash: str = "",
    corpus: str = "",
    corpus_hash: str = "",
    scope: CalibrationScope | None = None,
    scorer: str = "",
    seed: int | None = None,
    evaluation_id: str = "",
) -> CalibrationFit:
    """Fit accept/reject/escalate thresholds at bounded error rates.

    ``high`` is the lowest threshold meeting ``far_target`` (accept above);
    ``low`` is the highest threshold meeting ``frr_target`` (reject below).
    Infeasible targets fail loudly instead of silently inverting the band.
    """
    from relate.retrieval.calibration import CalibrationRecord

    for name, target in (("far_target", far_target), ("frr_target", frr_target)):
        if not 0.0 < float(target) < 1.0:
            raise RelateError(f"{name} must be strictly between 0 and 1")
    positives = [float(s) for s in positive_scores]
    negatives = [float(s) for s in negative_scores]
    positive_dist = describe_scores("positive", positives)
    negative_dist = describe_scores("negative", negatives)
    thresholds, fars, frrs = roc_curve(positives, negatives)
    auc = roc_auc(positives, negatives)
    eer_threshold, eer_rate = equal_error_point(thresholds, fars, frrs)

    high_candidates = [t for t, far in zip(thresholds, fars) if far <= far_target]
    low_candidates = [t for t, frr in zip(thresholds, frrs) if frr <= frr_target]
    if not high_candidates:
        raise RelateError("far_target infeasible: even the top score misses it")
    if not low_candidates:
        raise RelateError("frr_target infeasible: even the bottom score misses it")
    # Anchor the reject bound first, then take the most lenient accept bound
    # at or above it. An unconstrained minimum would undercut the reject
    # bound on well-separated data and invert the band.
    low = max(low_candidates)
    feasible_high = [t for t in high_candidates if t >= low]
    if not feasible_high:
        raise RelateError(
            "infeasible operating point: no accept bound above the reject "
            "bound meets far_target; loosen a target or re-measure"
        )
    high = min(feasible_high)
    far_at = float(fars[thresholds.index(high)])
    frr_at = float(frrs[thresholds.index(low)])
    pooled = positives + negatives
    ambiguity = float(
        sum(1 for s in pooled if low <= s < high) / len(pooled)
    )
    operating_point = OperatingPoint(
        threshold_low=low,
        threshold_high=high,
        target=objective,
        false_accept_rate=far_at,
        false_reject_rate=frr_at,
        ambiguity_fraction=ambiguity,
    )
    resolved_scope = scope or CalibrationScope()
    record = CalibrationRecord(
        threshold=high,
        far=far_at,
        frr=frr_at,
        auc=auc,
        escalate_low=low,
        escalate_high=high,
        space_hash=space_hash,
        corpus=corpus,
        task=resolved_scope.task,
        corpus_hash=corpus_hash,
        scorer=scorer,
        scope=resolved_scope,
        negatives=negative_set,
        positive_set=positive_set,
        objective=objective,
    )
    return CalibrationFit(
        positive=positive_dist,
        negative=negative_dist,
        thresholds=thresholds,
        fars=fars,
        frrs=frrs,
        auc=auc,
        eer_threshold=eer_threshold,
        eer_rate=eer_rate,
        operating_point=operating_point,
        record=record,
    )


def calibration_card(
    fit: CalibrationFit,
    *,
    evaluation_id: str,
    corpus: str,
    corpus_hash: str,
    space_hash: str = "",
    scorer: str = "",
    seed: int | None = None,
) -> EvaluationCard:
    """Carry calibration evidence in an EvaluationCard (no parallel model)."""
    if not evaluation_id:
        raise RelateError("evaluation_id must be non-empty")
    if fit.operating_point is None or fit.record is None:
        raise RelateError("fit carries no operating point")
    point = fit.operating_point
    scope = getattr(fit.record, "scope", None)
    negatives = getattr(fit.record, "negatives", None)
    manifest = {
        "evaluation_id": evaluation_id,
        "task": "calibration",
        "space_hash": space_hash,
        "corpus": corpus,
        "corpus_hash": corpus_hash,
        "scorer": scorer or getattr(fit.record, "scorer", ""),
        "seed": seed,
        "scope": scope.as_dict() if scope is not None else {},
        "negatives": (
            {
                "name": negatives.name,
                "kind": negatives.kind,
                "n": negatives.n,
                "content_hash": negatives.content_hash,
            }
            if negatives is not None
            else None
        ),
        "objective": getattr(fit.record, "objective", ""),
    }
    return EvaluationCard(
        evaluation_id=evaluation_id,
        task="calibration",
        corpus=corpus,
        corpus_hash=corpus_hash,
        space_hash=space_hash,
        scorer=manifest["scorer"],
        metrics={
            "auc": fit.auc,
            "eer_rate": fit.eer_rate,
            "eer_threshold": fit.eer_threshold,
            "false_accept_rate": point.false_accept_rate,
            "false_reject_rate": point.false_reject_rate,
            "ambiguity_fraction": point.ambiguity_fraction,
            "threshold_low": point.threshold_low,
            "threshold_high": point.threshold_high,
            "n_positive": fit.positive.n,
            "n_negative": fit.negative.n,
        },
        manifest=json.dumps(manifest, sort_keys=True),
    )
