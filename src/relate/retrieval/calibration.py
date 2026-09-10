"""Calibration: turn raw scores into task/corpus/space-bound operating points.

Thresholds are provenance-bound (space_hash + corpus + task). Transferring a
threshold across domains without re-measurement is denied: the book measured a
0.10 threshold shift across domains and an 86% escalate band on hard pairs.

Measurement lives in :mod:`relate.evaluation.calibration`; this record is
the evidence it produces. A threshold without provenance is incomplete:
:meth:`staleness_against` reports *which* dimensions changed.
"""

from __future__ import annotations

from dataclasses import dataclass

from relate.evaluation.calibration import CalibrationDecision, CalibrationStaleness
from relate.evaluation.distributions import CalibrationScope, NegativeSetDescriptor


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    threshold: float
    far: float
    frr: float
    auc: float = 0.5
    escalate_low: float = 0.0
    escalate_high: float = 0.0
    space_hash: str = ""
    corpus: str = ""
    task: str = ""
    corpus_hash: str = ""
    scorer: str = ""
    scope: CalibrationScope | None = None
    negatives: NegativeSetDescriptor | None = None
    positive_set: str = ""
    objective: str = ""

    def decide(self, score: float) -> CalibrationDecision:
        """Accept | reject | escalate. No ``is_match``: the band stays explicit."""
        if self.escalate_high > self.escalate_low:
            if score < self.escalate_low:
                return CalibrationDecision.REJECT
            if score >= self.escalate_high:
                return CalibrationDecision.ACCEPT
            return CalibrationDecision.ESCALATE
        return (
            CalibrationDecision.ACCEPT
            if score >= self.threshold
            else CalibrationDecision.REJECT
        )

    def is_stale_for(self, *, space_hash: str = "", corpus: str = "", task: str = "") -> bool:
        """Legacy boolean check; delegates to :meth:`staleness_against`."""
        return self.staleness_against(
            space_hash=space_hash, corpus=corpus, task=task
        ).stale

    def staleness_against(
        self,
        *,
        space_hash: str = "",
        corpus: str = "",
        corpus_hash: str = "",
        task: str = "",
        domain: str = "",
        query_type: str = "",
        scorer: str = "",
        negatives: str = "",
        negatives_hash: str = "",
    ) -> CalibrationStaleness:
        """Report *which* provenance dimensions changed, not just that some did.

        An unbound dimension on either side counts as changed: a threshold
        without provenance cannot confirm compatibility.
        """
        changed: list[str] = []

        def differs(bound: str, query: str) -> bool:
            return bool(query) and bound != query

        if differs(self.space_hash, space_hash):
            changed.append("space_hash")
        if differs(self.corpus, corpus):
            changed.append("corpus")
        if differs(self.corpus_hash, corpus_hash):
            changed.append("corpus_hash")
        if differs(self.task, task):
            changed.append("task")
        scope = self.scope or CalibrationScope()
        if differs(scope.domain, domain):
            changed.append("domain")
        if differs(scope.query_type, query_type):
            changed.append("query_type")
        if differs(self.scorer, scorer):
            changed.append("scorer")
        negatives_name = self.negatives.name if self.negatives is not None else ""
        if differs(negatives_name, negatives):
            changed.append("negatives")
        negatives_digest = (
            self.negatives.content_hash if self.negatives is not None else ""
        ) or ""
        if differs(negatives_digest, negatives_hash):
            changed.append("negatives_hash")
        return CalibrationStaleness(stale=bool(changed), changed=tuple(changed))
