"""Retrieval policy: what the retrieval chain returns and why."""

from __future__ import annotations

from dataclasses import dataclass

from relate.retrieval.calibration import CalibrationRecord
from relate.retrieval.signals import SignalBundle


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    name: str
    calibration: CalibrationRecord | None = None
    require_verify_on_escalate: bool = True

    def decide(self, bundle: SignalBundle, score: float) -> str:
        route = bundle.route()
        if self.calibration is not None:
            calibrated = self.calibration.decide(score)
            if calibrated == "escalate":
                return "verify" if self.require_verify_on_escalate else "rerank"
            if calibrated == "reject":
                return "reject"
        return route
