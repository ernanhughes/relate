"""Bridges: a map is a function; a bridge is the map plus its measured scope."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from relate.evaluation.preservation import PreservationProfile
from relate.model import RelateError

VALID_STATUSES = ("ACTIVE", "EXPERIMENTAL", "DEPRECATED")


@dataclass(frozen=True, slots=True)
class Bridge:
    source_space_hash: str
    target_space_hash: str
    direction: str
    method: str
    mapping: np.ndarray = field(repr=False)
    anchor_coverage: str = ""
    status: str = "EXPERIMENTAL"
    preservation: PreservationProfile | None = None

    def __post_init__(self) -> None:
        mapping = np.asarray(self.mapping, dtype=np.float64)
        if mapping.ndim != 2:
            raise RelateError("bridge mapping must be a two-dimensional matrix")
        if self.status not in VALID_STATUSES:
            raise RelateError(f"unknown bridge status: {self.status}")
        if not self.direction:
            raise RelateError("direction must be non-empty")

    def apply(self, vectors: npt.ArrayLike) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float64)
        was_vector = matrix.ndim == 1
        if was_vector:
            matrix = matrix[None, :]
        if matrix.shape[1] != self.mapping.shape[0]:
            raise RelateError("vector dimensions do not match bridge mapping")
        out = matrix @ self.mapping
        return out[0] if was_vector else out

    def usable_for(self, scope: str) -> bool:
        """Fail-closed: no preservation profile means not usable."""
        if self.preservation is None:
            return False
        return self.preservation.usable_for(scope)

    @property
    def not_usable_for(self) -> tuple[str, ...]:
        if self.preservation is None:
            return ("all",)
        return self.preservation.not_usable_for
