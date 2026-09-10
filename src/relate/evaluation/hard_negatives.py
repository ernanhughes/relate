"""Hard-negative evaluation: does a method preserve the distinction that matters?

Generic science, corpus-agnostic. The semantic meaning of a case is simply::

    anchor should prefer positive over negative

Scoring is injected (see :mod:`relate.evaluation.baselines`): the evaluator
never knows whether a score came from cosine, Euclidean distance, a
:class:`~relate.model.RelationProjection`, a bridge, or a future scorer.
Ties are explicit outcomes, because near-zero margins are later input to
calibration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from relate.evaluation.baselines import ScoreFn, scorer_id_of
from relate.evaluation.cards import EvaluationCard
from relate.evaluation.metrics import mean, median
from relate.model import RelateError

OUTCOMES = ("WIN", "TIE", "LOSS")


@dataclass(frozen=True, slots=True)
class HardNegativeCase:
    """One ordering judgment: anchor should prefer positive over negative."""

    case_id: str
    anchor_id: str
    positive_id: str
    negative_id: str
    relation: str
    negative_relation: str | None = None
    group: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("case_id", "anchor_id", "positive_id", "negative_id", "relation"):
            if not getattr(self, name):
                raise RelateError(f"HardNegativeCase.{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class HardNegativeObservation:
    """Evidence for one case, not just a percentage."""

    case_id: str
    positive_score: float
    negative_score: float
    margin: float
    outcome: str
    correct: bool

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise RelateError(f"unknown outcome: {self.outcome}")


@dataclass(frozen=True, slots=True)
class GroupResult:
    total: int
    wins: int
    ties: int
    losses: int
    accuracy: float
    mean_margin: float


@dataclass(frozen=True, slots=True)
class HardNegativeReport:
    total: int
    wins: int
    ties: int
    losses: int
    accuracy: float
    tie_rate: float
    loss_rate: float
    accuracy_excluding_ties: float | None
    mean_margin: float
    median_margin: float
    by_relation: dict = field(default_factory=dict)
    by_group: dict = field(default_factory=dict)
    worst_groups: tuple = field(default_factory=tuple)
    observations: tuple = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class HardNegativeDelta:
    """The primitive for native -> translated / compressed / v1 -> v2."""

    accuracy_delta: float
    mean_margin_delta: float
    median_margin_delta: float
    tie_rate_delta: float
    relation_deltas: dict = field(default_factory=dict)
    group_deltas: dict = field(default_factory=dict)


def _group_result(observations: list[HardNegativeObservation]) -> GroupResult:
    wins = sum(1 for o in observations if o.outcome == "WIN")
    ties = sum(1 for o in observations if o.outcome == "TIE")
    losses = len(observations) - wins - ties
    return GroupResult(
        total=len(observations),
        wins=wins,
        ties=ties,
        losses=losses,
        accuracy=wins / len(observations),
        mean_margin=mean([o.margin for o in observations]),
    )


def evaluate_hard_negatives(
    cases: list[HardNegativeCase],
    vectors: Mapping[str, np.ndarray],
    scorer: ScoreFn,
    *,
    tie_tol: float = 1e-9,
) -> HardNegativeReport:
    """Judge every case with an injected scorer.

    ``margin = positive_score - negative_score`` (scorers return preference,
    higher = more preferred). ``|margin| <= tie_tol`` is an explicit TIE.
    """
    if not cases:
        raise RelateError("at least one hard-negative case is required")
    if not np.isfinite(tie_tol) or tie_tol < 0.0:
        raise RelateError("tie_tol must be finite and non-negative")
    seen = set()
    observations: list[HardNegativeObservation] = []
    for case in cases:
        if case.case_id in seen:
            raise RelateError(f"duplicate case_id: {case.case_id}")
        seen.add(case.case_id)
        try:
            anchor = vectors[case.anchor_id]
            positive = vectors[case.positive_id]
            negative = vectors[case.negative_id]
        except KeyError as missing:
            raise RelateError(f"missing vector for id: {missing.args[0]}") from None
        positive_score = float(scorer(np.asarray(anchor), np.asarray(positive)))
        negative_score = float(scorer(np.asarray(anchor), np.asarray(negative)))
        if not math.isfinite(positive_score) or not math.isfinite(negative_score):
            raise RelateError(f"scorer returned a non-finite score for {case.case_id}")
        margin = positive_score - negative_score
        if margin > tie_tol:
            outcome = "WIN"
        elif margin < -tie_tol:
            outcome = "LOSS"
        else:
            outcome = "TIE"
        observations.append(
            HardNegativeObservation(
                case_id=case.case_id,
                positive_score=positive_score,
                negative_score=negative_score,
                margin=margin,
                outcome=outcome,
                correct=outcome == "WIN",
            )
        )

    wins = sum(1 for o in observations if o.outcome == "WIN")
    ties = sum(1 for o in observations if o.outcome == "TIE")
    losses = len(observations) - wins - ties
    decisive = wins + losses
    margins = [o.margin for o in observations]

    by_relation: dict[str, GroupResult] = {}
    for case, observation in zip(cases, observations):
        by_relation.setdefault(case.relation, []).append(observation)
    by_relation = {key: _group_result(value) for key, value in by_relation.items()}

    by_group: dict[str, GroupResult] = {}
    for case, observation in zip(cases, observations):
        if case.group is not None:
            by_group.setdefault(case.group, []).append(observation)
    by_group = {key: _group_result(value) for key, value in by_group.items()}

    ranked = sorted(
        ((key, result.accuracy) for key, result in {**by_relation, **by_group}.items()),
        key=lambda item: (item[1], item[0]),
    )

    return HardNegativeReport(
        total=len(observations),
        wins=wins,
        ties=ties,
        losses=losses,
        accuracy=wins / len(observations),
        tie_rate=ties / len(observations),
        loss_rate=losses / len(observations),
        accuracy_excluding_ties=(wins / decisive) if decisive else None,
        mean_margin=mean(margins),
        median_margin=median(margins),
        by_relation=by_relation,
        by_group=by_group,
        worst_groups=tuple(key for key, _ in ranked[:5]),
        observations=tuple(observations),
    )


def compare_reports(
    reference: HardNegativeReport, candidate: HardNegativeReport
) -> HardNegativeDelta:
    """Accuracy/margin deltas overall plus per-relation and per-group deltas."""
    relation_deltas = {
        key: candidate.by_relation[key].accuracy - reference.by_relation[key].accuracy
        for key in reference.by_relation
        if key in candidate.by_relation
    }
    group_deltas = {
        key: candidate.by_group[key].accuracy - reference.by_group[key].accuracy
        for key in reference.by_group
        if key in candidate.by_group
    }
    return HardNegativeDelta(
        accuracy_delta=candidate.accuracy - reference.accuracy,
        mean_margin_delta=candidate.mean_margin - reference.mean_margin,
        median_margin_delta=candidate.median_margin - reference.median_margin,
        tie_rate_delta=candidate.tie_rate - reference.tie_rate,
        relation_deltas=relation_deltas,
        group_deltas=group_deltas,
    )


def hard_negative_card(
    report: HardNegativeReport,
    *,
    evaluation_id: str,
    space_hash: str,
    corpus: str,
    corpus_hash: str,
    scorer: ScoreFn | str,
    task: str = "hard_negative_ordering",
    seed: int | None = None,
) -> EvaluationCard:
    """Carry a report in an :class:`EvaluationCard` (no second DTO).

    Summary statistics become ``metrics``/``per_relation``; the full evidence
    (observations, groups, deltas' inputs) stays readable from the report the
    caller already holds. Provenance travels in ``manifest`` as JSON.
    """
    if not evaluation_id:
        raise RelateError("evaluation_id must be non-empty")
    scorer_id = scorer if isinstance(scorer, str) else scorer_id_of(scorer)
    manifest = {
        "evaluation_id": evaluation_id,
        "task": task,
        "space_hash": space_hash,
        "corpus": corpus,
        "corpus_hash": corpus_hash,
        "scorer": scorer_id,
        "seed": seed,
        "total": report.total,
        "wins": report.wins,
        "ties": report.ties,
        "losses": report.losses,
        "worst_groups": list(report.worst_groups),
    }
    return EvaluationCard(
        evaluation_id=evaluation_id,
        task=task,
        corpus=corpus,
        corpus_hash=corpus_hash,
        space_hash=space_hash,
        scorer=scorer_id,
        metrics={
            "accuracy": report.accuracy,
            "tie_rate": report.tie_rate,
            "loss_rate": report.loss_rate,
            "accuracy_excluding_ties": report.accuracy_excluding_ties,
            "mean_margin": report.mean_margin,
            "median_margin": report.median_margin,
            "total": report.total,
        },
        per_relation={
            key: {
                "accuracy": value.accuracy,
                "mean_margin": value.mean_margin,
                "total": value.total,
            }
            for key, value in report.by_relation.items()
        },
        manifest=json.dumps(manifest, sort_keys=True),
    )
