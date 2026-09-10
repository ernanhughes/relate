"""Preservation records for transformations: bridges, compression, operators.

Every transformation creates an obligation to measure what was preserved.
Global geometry preservation never implies faithfulness -- document drift
detected 0% of controlled corruption classes, and localized reversals
required external verification.
"""

from __future__ import annotations

from dataclasses import dataclass

from relate.evaluation.preservation import PreservationProfile

OPERATOR_CLASSES = ("IDENTITY", "CONSTANT_DELTA", "NONE", "RANK_R", "AFFINE", "MLP")


@dataclass(frozen=True, slots=True)
class CompressionRecord:
    source_space_hash: str
    derived_space_hash: str
    method: str
    preservation: PreservationProfile | None = None
    global_geometry_preserved: bool = False
    faithfulness_verified: bool = False


@dataclass(frozen=True, slots=True)
class TransformationRecord:
    relation: str
    operator_class: str
    reconstruction: float = 0.0
    collateral: float = 0.0
    source_space_hash: str = ""
    derived_space_hash: str = ""

    def __post_init__(self) -> None:
        if self.operator_class not in OPERATOR_CLASSES:
            raise ValueError(f"unknown operator class: {self.operator_class}")


def check_compression(
    *,
    source_space_hash: str,
    derived_space_hash: str,
    method: str,
    preservation: PreservationProfile | None = None,
    global_geometry_preserved: bool = False,
) -> CompressionRecord:
    """Fail-closed: geometry preservation never implies faithfulness."""
    return CompressionRecord(
        source_space_hash=source_space_hash,
        derived_space_hash=derived_space_hash,
        method=method,
        preservation=preservation,
        global_geometry_preserved=global_geometry_preserved,
        faithfulness_verified=False,
    )
