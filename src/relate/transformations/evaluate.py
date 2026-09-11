"""Generic transformation evaluation: orchestration without privilege.

The 4D bridge flow, de-bridged: transform held-out source rows through
any ``VectorTransformation``, compare candidates against the reference
authority with 4A machinery, convert to preservation results under an
explicit reference frame, and verdict under declared policy. Bridges
flow through here unchanged (their ``transformation_id`` is the bridge
id); compression and operators will follow in 5B/5C. No metric code
lives here -- only calls into existing evaluators.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from relate.evaluation.cross_space import CorrespondenceSet, compare_native_spaces
from relate.evaluation.preservation import (
    DEFAULT_POLICIES,
    PreservationPolicy,
    PreservationProfile,
    ReferenceFrame,
    build_preservation_profile,
    results_from_space_comparison,
)
from relate.model import RelateError, code_identity
from relate.transformations.contract import VectorTransformation


def evaluate_transformation(
    *,
    transformation: VectorTransformation,
    source_vectors: npt.ArrayLike,
    reference_vectors: npt.ArrayLike,
    correspondence: CorrespondenceSet,
    reference_frame: ReferenceFrame,
    candidate_space_hash: str = "",
    reference_space_hash: str = "",
    k: int = 10,
    with_counterpart: bool = True,
    hard_negative_cases=None,
    hard_negative_vectors=None,
    scorer=None,
    scorer_id: str = "",
    policies: list[PreservationPolicy] | None = None,
    evaluator_version: str | None = None,
) -> PreservationProfile:
    """Judge any vector transformation's candidates against an authority.

    ``reference_frame`` is required, never inferred: the same candidate
    vectors mean different things against target-native behavior,
    source-native behavior, or task gold. Identity derivation stays
    outside (callers resolve hashes via ``derived_transformation_space``
    or ``bridge_output_space``); this function only judges.
    """
    if not isinstance(transformation, VectorTransformation):
        raise RelateError("evaluation needs a VectorTransformation producer")
    if not isinstance(reference_frame, ReferenceFrame):
        raise RelateError("reference authority must be an explicit ReferenceFrame")
    source = np.asarray(source_vectors, dtype=np.float64)
    reference = np.asarray(reference_vectors, dtype=np.float64)
    candidate = np.asarray(transformation.transform(source), dtype=np.float64)
    if candidate.ndim != 2 or candidate.shape[0] != source.shape[0]:
        raise RelateError("transformation must preserve row count")
    comparison = compare_native_spaces(
        source_vectors=candidate,
        target_vectors=reference,
        correspondence=correspondence,
        source_space_hash=candidate_space_hash,
        target_space_hash=reference_space_hash,
        k=k,
        with_counterpart=with_counterpart,
        hard_negative_cases=hard_negative_cases,
        hard_negative_vectors=hard_negative_vectors,
        scorer=scorer,
        scorer_id=scorer_id,
    )
    results = results_from_space_comparison(comparison, frame=reference_frame)
    return build_preservation_profile(
        source_space_hash=transformation.source_space_hash,
        candidate_space_hash=candidate_space_hash,
        target_space_hash=reference_space_hash,
        results=results,
        policies=list(policies) if policies is not None else list(DEFAULT_POLICIES),
        bridge_id=transformation.transformation_id,
        evaluation_correspondence_hash=correspondence.content_hash,
        scorer=scorer_id,
        evaluator_version=(
            evaluator_version if evaluator_version is not None else code_identity()
        ),
    )
