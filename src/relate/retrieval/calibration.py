"""Calibration: turn raw scores into task/corpus/space-bound operating points.

Thresholds are provenance-bound (space_hash + corpus + task). Transferring a
threshold across domains without re-measurement is denied: the book measured a
0.10 threshold shift across domains and an 86% escalate band on hard pairs.
"""

from __future__ import annotations

from dataclasses import dataclass


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

    def decide(self, score: float) -> str:
        """Return accept | reject | escalate. Ambiguity region escalates."""
        if self.escalate_low <= score <= self.escalate_high:
            return "escalate"
        return "accept" if score >= self.threshold else "reject"

    def is_stale_for(self, *, space_hash: str = "", corpus: str = "", task: str = "") -> bool:
        """A threshold from another space/corpus/task is flagged stale."""
        if space_hash and self.space_hash and space_hash != self.space_hash:
            return True
        if corpus and self.corpus and corpus != self.corpus:
            return True
        if task and self.task and task != self.task:
            return True
        return False
