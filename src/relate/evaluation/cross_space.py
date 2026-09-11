"""Native cross-space comparison: what survives when representation changes.

Compares two independently produced embedding spaces over an explicit
shared-item correspondence. Comparing different spaces is allowed;
mixing (searching) them is denied without a measured bridge -- the
guard :func:`require_same_space_for_mixing` enforces that half.

No new metric definitions: structural similarity, neighborhoods, and
counterpart recovery come from 3B unchanged; hard-negative ordering
from 3A unchanged; calibration is not embedded here. No bridge fitting,
no compatibility score, no ``usable_for`` at this layer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from relate.evaluation.hard_negatives import (
    HardNegativeCase,
    compare_reports,
    evaluate_hard_negatives,
)
from relate.evaluation.baselines import ScoreFn
from relate.evaluation.neighborhoods import (
    SpaceComparisonReport,
    compare_geometry,
    compare_neighborhoods,
    counterpart_recovery,
)
from relate.model import RelateError


@dataclass(frozen=True, slots=True)
class CorrespondenceSet:
    """Explicit row correspondence between two matrices.

    Row order never defines correspondence silently: every comparison
    resolves rows through this object. Bijection is enforced so anchor
    sets stay reusable for bridge fitting and round-trip evaluation.
    """

    ids: tuple[str, ...]
    source_rows: tuple[int, ...]
    target_rows: tuple[int, ...]
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.ids:
            raise RelateError("correspondence needs at least one id")
        if not (
            len(self.ids) == len(self.source_rows) == len(self.target_rows)
        ):
            raise RelateError("ids and row maps need the same length")
        if len(set(self.ids)) != len(self.ids):
            raise RelateError("correspondence ids must be unique")
        if len(set(self.source_rows)) != len(self.source_rows):
            raise RelateError("source rows must be unique")
        if len(set(self.target_rows)) != len(self.target_rows):
            raise RelateError("target rows must be unique")
        if any(not isinstance(r, int) or isinstance(r, bool) or r < 0
               for r in (*self.source_rows, *self.target_rows)):
            raise RelateError("rows must be non-negative integers")
        canonical = json.dumps(
            sorted(
                (item_id, source_row, target_row)
                for item_id, source_row, target_row in zip(
                    self.ids, self.source_rows, self.target_rows
                )
            ),
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        if self.content_hash and self.content_hash != digest:
            raise RelateError("correspondence content_hash mismatch")
        if not self.content_hash:
            object.__setattr__(self, "content_hash", digest)

    @property
    def size(self) -> int:
        return len(self.ids)


def identity_correspondence(ids: list[str]) -> CorrespondenceSet:
    """Row-i to row-i correspondence over shared ids in order."""
    rows = tuple(range(len(ids)))
    return CorrespondenceSet(
        ids=tuple(ids), source_rows=rows, target_rows=rows
    )


def aligned_matrices(
    source_vectors: npt.ArrayLike,
    target_vectors: npt.ArrayLike,
    correspondence: CorrespondenceSet,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Resolve both matrices to correspondence order (copies)."""
    source = np.asarray(source_vectors, dtype=np.float64)
    target = np.asarray(target_vectors, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2:
        raise RelateError("source and target must be matrices")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise RelateError("vectors must contain only finite values")
    for name, matrix, rows in (
        ("source", source, correspondence.source_rows),
        ("target", target, correspondence.target_rows),
    ):
        if max(rows) >= matrix.shape[0]:
            raise RelateError(f"correspondence exceeds {name} row count")
    order = np.argsort(np.asarray(correspondence.ids))
    ordered_ids = tuple(np.asarray(correspondence.ids)[order].tolist())
    source_rows = np.asarray(correspondence.source_rows)[order]
    target_rows = np.asarray(correspondence.target_rows)[order]
    return source[source_rows], target[target_rows], ordered_ids


def row_permutation(n: int, seed: int) -> np.ndarray:
    """Deterministic derangement-ish control order (seeded shuffle)."""
    if n < 2:
        raise RelateError("permutation control needs at least two rows")
    return np.random.default_rng(seed).permutation(n)


def permuted_copy(vectors: npt.ArrayLike, seed: int) -> np.ndarray:
    """Row-shuffled copy: the chance-floor control for any comparison."""
    matrix = np.asarray(vectors, dtype=np.float64)
    return matrix[row_permutation(matrix.shape[0], seed)]


def require_same_space_for_mixing(
    source_space_hash: str, target_space_hash: str
) -> None:
    """Deny cross-space mixing without a measured bridge.

    Comparing spaces (this module) is allowed with distinct hashes;
    searching them together is not. A measured bridge authorizes mixing
    in 4C/4D -- never this function.
    """
    if source_space_hash != target_space_hash:
        raise RelateError(
            "cross-space mixing DENIED without a measured bridge: "
            f"{source_space_hash} != {target_space_hash}"
        )


def compare_native_spaces(
    *,
    source_vectors: npt.ArrayLike,
    target_vectors: npt.ArrayLike,
    correspondence: CorrespondenceSet,
    source_space_hash: str = "",
    target_space_hash: str = "",
    k: int = 10,
    metric: str = "cosine",
    counterpart_ks: tuple[int, ...] = (1, 5, 10),
    with_counterpart: bool = True,
    hard_negative_cases: list[HardNegativeCase] | None = None,
    hard_negative_vectors: tuple[dict, dict] | None = None,
    scorer: ScoreFn | None = None,
    scorer_id: str = "",
) -> SpaceComparisonReport:
    """Fill a SpaceComparisonReport from two native spaces.

    Distinct space hashes are expected, never an error. Hard-negative
    ordering is measured natively on each side with the same scorer and
    compared as a delta; supply cases plus both vector maps plus scorer
    together, or nothing at all.
    """
    source, target, ids = aligned_matrices(
        source_vectors, target_vectors, correspondence
    )
    names = list(ids)
    geometry = compare_geometry(
        source,
        target,
        names,
        neighborhood_k=k,
        reference_space_hash=source_space_hash or None,
        candidate_space_hash=target_space_hash or None,
    )
    neighborhood = compare_neighborhoods(source, target, names, k=k, metric=metric)
    counterpart = (
        counterpart_recovery(source, target, names, ks=counterpart_ks)
        if with_counterpart
        else None
    )

    delta = None
    if hard_negative_cases is not None:
        if hard_negative_vectors is None or scorer is None:
            raise RelateError(
                "hard-negative comparison needs cases, both vector maps, and a scorer"
            )
        source_map, target_map = hard_negative_vectors
        reference = evaluate_hard_negatives(hard_negative_cases, source_map, scorer)
        candidate = evaluate_hard_negatives(hard_negative_cases, target_map, scorer)
        delta = compare_reports(reference, candidate)

    return SpaceComparisonReport(
        source_space_hash=source_space_hash,
        target_space_hash=target_space_hash,
        geometry=geometry,
        neighborhood=neighborhood,
        counterpart=counterpart,
        hard_negatives=delta,
        correspondence_hash=correspondence.content_hash,
        scorer=scorer_id,
        metric=metric,
        k=k,
    )
