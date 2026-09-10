"""Retrieval policy: the consumer of bundles, not a field on them.

Routing thresholds live here, where policy belongs. A thin margin does
not authorize acceptance -- it triggers verification. Calibration, when
attached, overrides the margin route on escalate/reject.
"""

from __future__ import annotations

from dataclasses import dataclass

from relate.retrieval.calibration import CalibrationRecord
from relate.retrieval.signals import SignalBundle


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    name: str
    calibration: CalibrationRecord | None = None
    require_verify_on_escalate: bool = True
    accept_margin: float = 0.10
    verify_margin: float = 0.03

    def route(self, bundle: SignalBundle) -> str:
        """Margin route: accept | rerank | verify. No margin means verify."""
        if bundle.margin is None:
            return "verify"
        if bundle.margin >= self.accept_margin:
            return "accept"
        if bundle.margin >= self.verify_margin:
            return "rerank"
        return "verify"

    def decide(self, bundle: SignalBundle, score: float) -> str:
        route = self.route(bundle)
        if self.calibration is not None:
            calibrated = self.calibration.decide(score)
            if calibrated == "escalate":
                return "verify" if self.require_verify_on_escalate else "rerank"
            if calibrated == "reject":
                return "reject"
        return route
