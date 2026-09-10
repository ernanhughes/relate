"""Space comparison: deny-by-default cross-space gate.

Equal dimensions do not imply compatible spaces. A matching ``space_hash``
permits an operation; any mismatch requires a MEASURED bridge, never an
assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

from relate.model import RelateError
from relate.spaces.identity import SpaceIdentity


@dataclass(frozen=True, slots=True)
class SpaceComparison:
    space_a_hash: str
    space_b_hash: str
    identical: bool
    neighborhood_overlap: float | None = None
    linear_cka: float | None = None
    verdict: str = "DENIED"

    def require_compatible(self) -> None:
        if not self.identical:
            raise RelateError(
                "cross-space operation DENIED without a measured bridge: "
                f"{self.space_a_hash} != {self.space_b_hash}"
            )


def compare_spaces(
    a: SpaceIdentity,
    b: SpaceIdentity,
    *,
    neighborhood_overlap: float | None = None,
    linear_cka: float | None = None,
) -> SpaceComparison:
    identical = a.space_hash == b.space_hash
    verdict = "IDENTICAL" if identical else "DENIED_REQUIRES_BRIDGE"
    return SpaceComparison(
        space_a_hash=a.space_hash,
        space_b_hash=b.space_hash,
        identical=identical,
        neighborhood_overlap=neighborhood_overlap,
        linear_cka=linear_cka,
        verdict=verdict,
    )
