"""Signal bundles: per-result diagnostic evidence, composed not computed.

A bundle carries already-measured evidence about one query/result pair:
the retrieval score, the 3A margin, 3B local geometry and neighborhood
facts, and the 3C calibration outcome. It never fits, searches, or
decides -- policy consumes bundles; bundles do not contain policy.

Missing and negative signals are structurally distinct: ``None`` means
unavailable, a float means measured (however bad). Embedding-derived
evidence and external verification travel in separate fields and are
never merged into one number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from relate.evaluation.calibration import CalibrationDecision
from relate.evaluation.hard_negatives import HardNegativeObservation
from relate.evaluation.neighborhoods import (
    Hubness,
    LocalDensity,
    NeighborhoodStability,
)
from relate.retrieval.calibration import CalibrationRecord


@dataclass(frozen=True, slots=True)
class ExternalSignals:
    """External verification, typed as external. Never merged with geometry."""

    source: str
    signals: dict = field(default_factory=dict)
    verdict: str = ""


@dataclass(frozen=True, slots=True)
class SignalProvenance:
    """What went into the bundle: evaluator identities and measurement specs."""

    scorer_id: str = ""
    space_hash: str = ""
    margin_case_id: str = ""
    density: LocalDensity | None = None
    hubness: Hubness | None = None
    stability: NeighborhoodStability | None = None
    calibration_id: str = ""
    external_source: str = ""


@dataclass(frozen=True, slots=True)
class SignalBundle:
    """One query/result pair's evidence. No scores, no decisions, no policy."""

    score: float
    margin: float | None = None
    local_density: float | None = None
    hubness: float | None = None
    neighborhood_stability: float | None = None
    calibration_decision: CalibrationDecision | None = None
    external: ExternalSignals | None = None
    provenance: SignalProvenance | None = None

    @property
    def available_signals(self) -> tuple[str, ...]:
        """Names of measured signals; absent names are unavailable, not bad."""
        present = ["score"]
        if self.margin is not None:
            present.append("margin")
        if self.local_density is not None:
            present.append("local_density")
        if self.hubness is not None:
            present.append("hubness")
        if self.neighborhood_stability is not None:
            present.append("neighborhood_stability")
        if self.calibration_decision is not None:
            present.append("calibration_decision")
        if self.external is not None:
            present.append("external")
        return tuple(present)


def build_signal_bundle(
    *,
    score: float,
    observation: HardNegativeObservation | None = None,
    margin: float | None = None,
    density: LocalDensity | None = None,
    hubness: Hubness | None = None,
    stability: NeighborhoodStability | None = None,
    calibration: CalibrationRecord | None = None,
    external: ExternalSignals | None = None,
    scorer_id: str = "",
    space_hash: str = "",
    calibration_id: str = "",
) -> SignalBundle:
    """Compose a bundle by extraction only.

    No neighbor search, no calibration fitting, no model training: every
    argument is already-measured evidence (or None when unavailable).
    ``observation`` supplies the 3A margin; an explicit ``margin`` covers
    presentation-signed reuse and must not disagree with the observation.
    """
    resolved_margin = observation.margin if observation is not None else margin
    if (
        observation is not None
        and margin is not None
        and observation.margin != margin
    ):
        from relate.model import RelateError

        raise RelateError("observation margin and explicit margin disagree")
    return SignalBundle(
        score=float(score),
        margin=resolved_margin,
        local_density=density.value if density is not None else None,
        hubness=hubness.normalized if hubness is not None else None,
        neighborhood_stability=stability.value if stability is not None else None,
        calibration_decision=(
            calibration.decide(score) if calibration is not None else None
        ),
        external=external,
        provenance=SignalProvenance(
            scorer_id=scorer_id,
            space_hash=space_hash,
            margin_case_id=observation.case_id if observation is not None else "",
            density=density,
            hubness=hubness,
            stability=stability,
            calibration_id=calibration_id,
            external_source=external.source if external is not None else "",
        ),
    )
