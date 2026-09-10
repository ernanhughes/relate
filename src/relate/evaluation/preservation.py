"""Preservation profiles: what a transformation preserved, for which scope.

Three visible steps, never collapsed into one compatibility score:

measurement (existing evaluators) -> comparison to an explicit
authority (reference frame) -> scope-specific verdicts from declared
policy.

A map proves coordinates can be transformed. A preservation profile
determines what that transformation is actually good for.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

from relate.evaluation.neighborhoods import SpaceComparisonReport
from relate.model import RelateError

REQUIREMENT_KINDS = (
    "min_value",
    "max_value",
    "min_ratio",
    "max_delta",
    "min_delta",
    "max_abs_delta",
)


class ReferenceFrame(str, Enum):
    """Whose behavior is authoritative for one measurement.

    Most A -> B claims use target-native behavior. Source-native is
    legitimate when the question is explicitly "did translation keep
    what the source did". Task-gold covers gold-labeled references.
    Never inferred silently: every result carries one.
    """

    TARGET_NATIVE = "target_native"
    SOURCE_NATIVE = "source_native"
    TASK_GOLD = "task_gold"


class PreservationVerdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class PreservationResult:
    capability: str
    metric: str
    value: float
    reference_value: float | None = None
    delta: float | None = None
    ratio: float | None = None
    reference_frame: ReferenceFrame = ReferenceFrame.TARGET_NATIVE
    evidence_id: str = ""

    def __post_init__(self) -> None:
        if not self.capability or not self.metric:
            raise RelateError("capability and metric must be non-empty")
        if isinstance(self.reference_frame, str) and not isinstance(
            self.reference_frame, ReferenceFrame
        ):
            try:
                object.__setattr__(
                    self, "reference_frame", ReferenceFrame(self.reference_frame)
                )
            except ValueError:
                raise RelateError(
                    f"unknown reference frame: {self.reference_frame}"
                ) from None


@dataclass(frozen=True, slots=True)
class Requirement:
    capability: str
    metric: str
    kind: str
    bound: float
    advisory: bool = False

    def __post_init__(self) -> None:
        if not self.capability or not self.metric:
            raise RelateError("requirement needs a capability and metric")
        if self.kind not in REQUIREMENT_KINDS:
            raise RelateError(f"unknown requirement kind: {self.kind}")

    def check(self, result: PreservationResult | None) -> bool | None:
        """True/False, or None when the capability was never measured."""
        if result is None:
            return None
        if self.kind == "min_value":
            return result.value is not None and result.value >= self.bound
        if self.kind == "max_value":
            return result.value is not None and result.value <= self.bound
        if self.kind == "min_ratio":
            return result.ratio is not None and result.ratio >= self.bound
        if self.kind == "max_delta":
            return result.delta is not None and result.delta <= self.bound
        if self.kind == "min_delta":
            return result.delta is not None and result.delta >= self.bound
        return result.delta is not None and abs(result.delta) <= self.bound

    def describe(self) -> str:
        return f"{self.metric} {self.kind} {self.bound:g}"


@dataclass(frozen=True, slots=True)
class PreservationPolicy:
    scope: str
    requirements: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.scope:
            raise RelateError("policy scope must be non-empty")

    @property
    def policy_id(self) -> str:
        """Deterministic, provenance-bound policy identity."""
        canonical = json.dumps(
            {
                "scope": self.scope,
                "requirements": [
                    {
                        "capability": r.capability,
                        "metric": r.metric,
                        "kind": r.kind,
                        "bound": r.bound,
                        "advisory": r.advisory,
                    }
                    for r in self.requirements
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ScopeVerdict:
    scope: str
    verdict: PreservationVerdict
    required_capabilities: tuple = field(default_factory=tuple)
    failed_capabilities: tuple = field(default_factory=tuple)
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class PreservationProvenance:
    source_space_hash: str = ""
    candidate_space_hash: str = ""
    target_space_hash: str = ""
    bridge_id: str = ""
    evaluation_correspondence_hash: str = ""
    scorer: str = ""
    evaluator_version: str = ""
    policy_ids: tuple = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class PreservationProfile:
    source_space_hash: str
    candidate_space_hash: str
    target_space_hash: str
    bridge_id: str = ""
    evaluation_correspondence_hash: str = ""
    results: tuple = field(default_factory=tuple)
    verdicts: tuple = field(default_factory=tuple)
    provenance: PreservationProvenance | None = None

    def result(self, capability: str, metric: str = "") -> PreservationResult | None:
        for item in self.results:
            if item.capability == capability and (not metric or item.metric == metric):
                return item
        return None

    def verdict_for(self, scope: str) -> ScopeVerdict | None:
        for verdict in self.verdicts:
            if verdict.scope == scope:
                return verdict
        return None

    def usable_for(self, scope: str) -> bool:
        """Lookup over an explicit verdict. Unknown scopes fail closed."""
        verdict = self.verdict_for(scope)
        return verdict is not None and verdict.verdict == PreservationVerdict.PASS

    @property
    def usable_scopes(self) -> tuple[str, ...]:
        return tuple(
            verdict.scope
            for verdict in self.verdicts
            if verdict.verdict == PreservationVerdict.PASS
        )

    @property
    def not_usable_for(self) -> tuple[str, ...]:
        return tuple(
            verdict.scope
            for verdict in self.verdicts
            if verdict.verdict != PreservationVerdict.PASS
        )

    def explain(self, scope: str) -> str:
        """Why a scope passed, warns, fails, or was never judged."""
        verdict = self.verdict_for(scope)
        if verdict is None:
            return f"UNKNOWN {scope}: no verdict measured"
        lines = [f"{verdict.verdict.value} {scope}:"]
        if verdict.rationale:
            lines.append(f"- {verdict.rationale}")
        elif verdict.verdict == PreservationVerdict.PASS:
            lines.append("- all required gates pass")
        return "\n".join(lines)


def _judge(
    policy: PreservationPolicy, results: dict[tuple[str, str], PreservationResult]
) -> ScopeVerdict:
    failed: list[str] = []
    advisory_missed: list[str] = []
    required: list[str] = []
    notes: list[str] = []
    for requirement in policy.requirements:
        outcome = requirement.check(results.get((requirement.capability, requirement.metric)))
        label = f"{requirement.capability}/{requirement.metric}"
        if requirement.advisory:
            if outcome is not True:
                advisory_missed.append(label)
                notes.append(f"advisory miss: {label} ({requirement.describe()})")
            continue
        required.append(label)
        if outcome is not True:
            failed.append(label)
            reason = "unmeasured" if outcome is None else "gate not met"
            notes.append(f"failed: {label} ({requirement.describe()}, {reason})")
    if failed:
        verdict = PreservationVerdict.FAIL
    elif advisory_missed:
        verdict = PreservationVerdict.WARN
    else:
        verdict = PreservationVerdict.PASS
    return ScopeVerdict(
        scope=policy.scope,
        verdict=verdict,
        required_capabilities=tuple(required),
        failed_capabilities=tuple(failed),
        rationale="; ".join(notes),
    )


def build_preservation_profile(
    *,
    source_space_hash: str,
    candidate_space_hash: str,
    target_space_hash: str,
    results: list[PreservationResult],
    policies: list[PreservationPolicy],
    bridge_id: str = "",
    evaluation_correspondence_hash: str = "",
    scorer: str = "",
    evaluator_version: str = "",
) -> PreservationProfile:
    """Assemble measured results plus declared-policy verdicts.

    No metric code lives here: results arrive measured, verdicts derive
    from policy. The same evidence under different policies may
    legitimately produce different verdicts.
    """
    if not source_space_hash or not candidate_space_hash or not target_space_hash:
        raise RelateError("profile needs source, candidate, and target hashes")
    keyed = {(result.capability, result.metric): result for result in results}
    verdicts = tuple(_judge(policy, keyed) for policy in policies)
    return PreservationProfile(
        source_space_hash=source_space_hash,
        candidate_space_hash=candidate_space_hash,
        target_space_hash=target_space_hash,
        bridge_id=bridge_id,
        evaluation_correspondence_hash=evaluation_correspondence_hash,
        results=tuple(results),
        verdicts=verdicts,
        provenance=PreservationProvenance(
            source_space_hash=source_space_hash,
            candidate_space_hash=candidate_space_hash,
            target_space_hash=target_space_hash,
            bridge_id=bridge_id,
            evaluation_correspondence_hash=evaluation_correspondence_hash,
            scorer=scorer,
            evaluator_version=evaluator_version,
            policy_ids=tuple(policy.policy_id for policy in policies),
        ),
    )


def _ratio(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0.0:
        return None
    return value / reference


def results_from_space_comparison(
    report: SpaceComparisonReport,
    *,
    frame: ReferenceFrame = ReferenceFrame.TARGET_NATIVE,
    evidence_prefix: str = "",
) -> list[PreservationResult]:
    """Translate a 4A comparison into preservation results (no measuring).

    Perfect agreement (1.0) is the reference for structural metrics;
    zero is the reference for deltas. Per-relation deltas are emitted
    individually so one overall number can never hide a failed class.
    """
    results: list[PreservationResult] = []

    def tag(capability: str) -> str:
        return f"{evidence_prefix}{capability}" if evidence_prefix else capability

    geometry = report.geometry
    if geometry is not None:
        for metric, value in (
            ("cka", geometry.cka),
            ("cosine_matrix_correlation", geometry.cosine_matrix_correlation),
            ("distance_matrix_correlation", geometry.distance_matrix_correlation),
        ):
            if value is not None:
                results.append(
                    PreservationResult(
                        capability=tag("representation_similarity"),
                        metric=metric,
                        value=float(value),
                        reference_value=1.0,
                        ratio=_ratio(float(value), 1.0),
                        reference_frame=frame,
                    )
                )
    neighborhood = report.neighborhood
    if neighborhood is not None:
        results.append(
            PreservationResult(
                capability=tag("neighborhood_structure"),
                metric=f"overlap_at_{neighborhood.k}",
                value=neighborhood.mean_overlap,
                reference_value=1.0,
                ratio=_ratio(neighborhood.mean_overlap, 1.0),
                reference_frame=frame,
            )
        )
        results.append(
            PreservationResult(
                capability=tag("neighborhood_structure"),
                metric="top1_agreement",
                value=neighborhood.top1_agreement,
                reference_value=1.0,
                ratio=_ratio(neighborhood.top1_agreement, 1.0),
                reference_frame=frame,
            )
        )
    counterpart = report.counterpart
    if counterpart is not None:
        results.append(
            PreservationResult(
                capability=tag("counterpart_recovery"),
                metric="recall_at_1",
                value=counterpart.top1,
                reference_value=1.0,
                ratio=_ratio(counterpart.top1, 1.0),
                reference_frame=frame,
            )
        )
        results.append(
            PreservationResult(
                capability=tag("counterpart_recovery"),
                metric="mrr",
                value=counterpart.mrr,
                reference_value=1.0,
                ratio=_ratio(counterpart.mrr, 1.0),
                reference_frame=frame,
            )
        )
    delta = report.hard_negatives
    if delta is not None:
        results.append(
            PreservationResult(
                capability=tag("hard_negative_ordering"),
                metric="accuracy_delta",
                value=delta.accuracy_delta,
                reference_value=0.0,
                delta=delta.accuracy_delta,
                reference_frame=frame,
            )
        )
        for relation in sorted(delta.relation_deltas):
            results.append(
                PreservationResult(
                    capability=tag(f"hard_negative_ordering/{relation}"),
                    metric="accuracy_delta",
                    value=delta.relation_deltas[relation],
                    reference_value=0.0,
                    delta=delta.relation_deltas[relation],
                    reference_frame=frame,
                )
            )
    return results


def calibration_transfer_results(
    *,
    native_fit,
    candidate_positive_scores: list[float],
    candidate_negative_scores: list[float],
    frame: ReferenceFrame = ReferenceFrame.TARGET_NATIVE,
) -> list[PreservationResult]:
    """Threshold transfer via 3C reuse: native point on candidate scores.

    Compares native operating behavior against translated-candidate
    behavior: FAR/FRR drift at the native thresholds, ambiguity drift,
    and the EER threshold shift (a fresh 3C fit on candidate scores
    supplies the candidate operating point -- no new threshold logic).
    """
    from relate.evaluation.calibration import calibrate

    native_point = native_fit.operating_point
    if native_point is None:
        raise RelateError("native fit carries no operating point")
    positives = [float(s) for s in candidate_positive_scores]
    negatives = [float(s) for s in candidate_negative_scores]
    if not positives or not negatives:
        raise RelateError("transfer needs candidate positives and negatives")
    far = float(sum(1 for s in negatives if s >= native_point.threshold_high) / len(negatives))
    frr = float(sum(1 for s in positives if s < native_point.threshold_low) / len(positives))
    pooled = positives + negatives
    ambiguity = float(
        sum(1 for s in pooled if native_point.threshold_low <= s < native_point.threshold_high)
        / len(pooled)
    )
    candidate_fit = calibrate(positives, negatives)
    shift = candidate_fit.eer_threshold - native_fit.eer_threshold
    native_ambiguity = native_point.ambiguity_fraction
    return [
        PreservationResult(
            capability="calibration_transfer",
            metric="far_increase",
            value=far,
            reference_value=native_point.false_accept_rate,
            delta=far - native_point.false_accept_rate,
            reference_frame=frame,
            evidence_id="native-operating-point",
        ),
        PreservationResult(
            capability="calibration_transfer",
            metric="frr_increase",
            value=frr,
            reference_value=native_point.false_reject_rate,
            delta=frr - native_point.false_reject_rate,
            reference_frame=frame,
            evidence_id="native-operating-point",
        ),
        PreservationResult(
            capability="calibration_transfer",
            metric="ambiguity_increase",
            value=ambiguity,
            reference_value=native_ambiguity,
            delta=ambiguity - native_ambiguity,
            reference_frame=frame,
            evidence_id="native-operating-point",
        ),
        PreservationResult(
            capability="calibration_transfer",
            metric="threshold_shift",
            value=candidate_fit.eer_threshold,
            reference_value=native_fit.eer_threshold,
            delta=shift,
            reference_frame=frame,
            evidence_id="eer-operating-point",
        ),
    ]


DEFAULT_POLICIES: tuple[PreservationPolicy, ...] = (
    PreservationPolicy(
        scope="retrieval",
        requirements=(
            Requirement("counterpart_recovery", "recall_at_1", "min_value", 0.90),
            Requirement(
                "neighborhood_structure",
                "overlap_at_10",
                "min_value",
                0.60,
                advisory=True,
            ),
        ),
    ),
    PreservationPolicy(
        scope="threshold_transfer",
        requirements=(
            Requirement("calibration_transfer", "far_increase", "max_delta", 0.05),
            Requirement(
                "calibration_transfer", "threshold_shift", "max_abs_delta", 0.05
            ),
        ),
    ),
)
