"""Preservation profiles: what a bridge/transformation preserved, per task.

A bridge can preserve retrieval while losing fine distinctions, thresholds
or neighborhoods. Preservation never collapses to one cosine number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_STATUSES = ("PASS", "WARN", "FAIL", "UNKNOWN")


@dataclass(frozen=True, slots=True)
class PreservationProfile:
    source_space_hash: str
    target_space_hash: str
    results: dict = field(default_factory=dict)
    native_reference: dict = field(default_factory=dict)
    random_floor: dict = field(default_factory=dict)

    def status(self, task: str) -> str:
        entry = self.results.get(task, {})
        status = entry.get("status", "UNKNOWN")
        return status if status in VALID_STATUSES else "UNKNOWN"

    def usable_for(self, scope: str) -> bool:
        """Fail-closed: only an explicit PASS authorizes use."""
        return self.status(scope) == "PASS"

    @property
    def usable_scopes(self) -> tuple[str, ...]:
        return tuple(
            task
            for task, entry in self.results.items()
            if entry.get("status") == "PASS"
        )

    @property
    def not_usable_for(self) -> tuple[str, ...]:
        return tuple(
            task
            for task, entry in self.results.items()
            if entry.get("status") in ("FAIL", "WARN")
        )


def make_preservation_profile(
    source_space_hash: str,
    target_space_hash: str,
    scores: dict,
    *,
    thresholds: dict | None = None,
    native_reference: dict | None = None,
    random_floor: dict | None = None,
) -> PreservationProfile:
    """Build a profile from raw per-task scores.

    A task PASSES when its score meets its threshold (default 0.8), WARNS
    within 0.1 below, otherwise FAILS. Missing thresholds fail closed only
    when no score exists (UNKNOWN).
    """
    thresholds = thresholds or {}
    results: dict = {}
    for task, score in scores.items():
        threshold = float(thresholds.get(task, 0.8))
        value = float(score)
        if value >= threshold:
            status = "PASS"
        elif value >= threshold - 0.10:
            status = "WARN"
        else:
            status = "FAIL"
        results[task] = {"score": value, "threshold": threshold, "status": status}
    return PreservationProfile(
        source_space_hash=source_space_hash,
        target_space_hash=target_space_hash,
        results=results,
        native_reference=dict(native_reference or {}),
        random_floor=dict(random_floor or {}),
    )
