"""Signal bundle: per-result diagnostic vector.

Research measured score-only 0.76 -> 0.90 using geometric signals on hard
cases (margin, density, rank, hubness). The bundle exposes those signals so
policy can route to accept / rerank / verify instead of trusting raw score.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SignalBundle:
    score: float
    margin: float
    density: float = 0.0
    hubness: float = 0.0
    stability: float = 1.0

    def route(self, *, accept_margin: float = 0.10, verify_margin: float = 0.03) -> str:
        """Route a candidate: accept | rerank | verify.

        A high score with a thin margin does not authorize acceptance --
        it triggers verification.
        """
        if self.margin >= accept_margin:
            return "accept"
        if self.margin >= verify_margin:
            return "rerank"
        return "verify"
