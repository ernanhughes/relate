"""Observatory: compose spaces, relations, evidence and policy.

The runtime refuses to trust unmeasured things. Denial is the default;
permission must cite a record.

Identity is exact (``space_hash``), compatibility is empirical (measured
bridge), usability is policy (``usable_for(scope)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from relate.bridges.base import (
    Bridge,
    BridgeSpec,
    bridge_output_space,
    code_identity,
)
from relate.bridges.fit import fit_bridge
from relate.bridges.registry import BridgeRegistry
from relate.evaluation.baselines import ScoreFn, cosine_scorer
from relate.evaluation.cards import EvaluationCard
from relate.evaluation.cross_space import (
    CorrespondenceSet,
    compare_native_spaces,
    identity_correspondence,
)
from relate.evaluation.hard_negatives import HardNegativeObservation
from relate.evaluation.neighborhoods import (
    hubness_counts,
    local_density,
    make_hubness,
    shared_neighborhood_stability,
)
from relate.evaluation.preservation import (
    DEFAULT_POLICIES,
    PreservationProfile,
    build_preservation_profile,
    results_from_space_comparison,
)
from relate.model import RelateError, RelationProjection
from relate.relations.base import Relation, fit_relation
from relate.retrieval.calibration import CalibrationRecord
from relate.retrieval.signals import (
    ExternalSignals,
    SignalBundle,
    build_signal_bundle,
)
from relate.spaces.comparison import SpaceComparison, compare_spaces
from relate.spaces.identity import SpaceIdentity
from relate.spaces.registry import SpaceRegistry
from relate.transformations.records import CompressionRecord, check_compression


@dataclass
class Observatory:
    """Embedding runtime: spaces, relations, evidence, policy."""

    spaces: SpaceRegistry = field(default_factory=SpaceRegistry)
    bridges: BridgeRegistry = field(default_factory=BridgeRegistry)
    evaluations: list = field(default_factory=list)
    calibrations: list = field(default_factory=list)

    # -- spaces ---------------------------------------------------------
    def register_space(self, space: SpaceIdentity | None = None, **kwargs) -> SpaceIdentity:
        if space is None:
            space = SpaceIdentity(**kwargs)
        return self.spaces.register(space)

    def attach(self, embeddings, *, space: SpaceIdentity) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim != 2:
            raise RelateError("embeddings must be a two-dimensional matrix")
        if matrix.shape[1] != space.dimensions:
            raise RelateError("embedding dimensions do not match space")
        self.spaces.register(space)
        return matrix

    def inspect(self, vectors: np.ndarray) -> dict:
        matrix = np.asarray(vectors, dtype=np.float64)
        return {
            "count": int(matrix.shape[0]),
            "dimensions": int(matrix.shape[1]),
            "mean_norm": float(np.linalg.norm(matrix, axis=1).mean()),
        }

    def compare_spaces(
        self, a: SpaceIdentity, b: SpaceIdentity, **kwargs
    ) -> SpaceComparison:
        return compare_spaces(a, b, **kwargs)

    # -- relations ------------------------------------------------------
    def fit_relation(self, name, embeddings, coordinates, **kwargs) -> Relation:
        space_hash = kwargs.pop("space_hash", "")
        return fit_relation(name, embeddings, coordinates,
                            space_hash=space_hash, **kwargs)

    def search(self, query_vector, targets, *, relation: Relation, k: int = 10,
               **kwargs):
        return relation.projection.search(query_vector, targets, k=k, **kwargs)

    # -- bridges (producers only; judgment lives in evaluation) ------------
    def fit_bridge(
        self,
        source: np.ndarray,
        target: np.ndarray,
        *,
        source_space: SpaceIdentity,
        target_space: SpaceIdentity,
        correspondence: CorrespondenceSet,
        method: str = "procrustes",
        params: dict | None = None,
        coverage: dict | None = None,
    ) -> Bridge:
        """Fit a directional producer on explicit anchor correspondence."""
        spec = BridgeSpec(
            source_space_hash=source_space.space_hash,
            target_space_hash=target_space.space_hash,
            method=method,
            params=dict(params or {}),
        )
        bridge = fit_bridge(
            source_vectors=source,
            target_vectors=target,
            correspondence=correspondence,
            spec=spec,
            coverage=coverage,
        )
        return self.bridges.register(bridge)

    def bridge_space(
        self,
        source_space: SpaceIdentity,
        bridge: Bridge,
        target_reference: SpaceIdentity,
    ) -> SpaceIdentity:
        """Derived identity for a bridge's candidates (never the native hash)."""
        return bridge_output_space(source_space, bridge, target_reference)

    def evaluate_bridge(
        self,
        bridge: Bridge,
        evaluation_source,
        evaluation_target,
        *,
        correspondence=None,
        hard_negative_cases=None,
        hard_negative_vectors=None,
        scorer=None,
        scorer_id: str = "",
        policies=None,
        k: int = 10,
    ) -> PreservationProfile:
        """Judge a bridge's candidates through the measurement spine.

        Transforms held-out source rows, compares candidates against the
        native target with 4A machinery, converts the comparison into
        preservation results, and verdicts them under declared policy.
        Designed so 4D's ``usable_for``/``explain`` API falls out
        naturally; this method already returns the profile they read.
        """
        candidate = bridge.transform(np.asarray(evaluation_source, dtype=np.float64))
        target = np.asarray(evaluation_target, dtype=np.float64)
        if correspondence is None:
            correspondence = identity_correspondence(
                [str(i) for i in range(candidate.shape[0])]
            )
        try:
            source_space = self.spaces.require(bridge.source_space_hash)
            target_space = self.spaces.require(bridge.target_space_hash)
            candidate_space = bridge_output_space(source_space, bridge, target_space)
            candidate_hash = candidate_space.space_hash
        except RelateError:
            candidate_hash = f"derived:{bridge.bridge_id}"
        comparison = compare_native_spaces(
            source_vectors=candidate,
            target_vectors=target,
            correspondence=correspondence,
            source_space_hash=candidate_hash,
            target_space_hash=bridge.target_space_hash,
            k=k,
            hard_negative_cases=hard_negative_cases,
            hard_negative_vectors=hard_negative_vectors,
            scorer=scorer,
            scorer_id=scorer_id,
        )
        results = results_from_space_comparison(comparison)
        return build_preservation_profile(
            source_space_hash=bridge.source_space_hash,
            candidate_space_hash=candidate_hash,
            target_space_hash=bridge.target_space_hash,
            results=results,
            policies=list(policies) if policies is not None else list(DEFAULT_POLICIES),
            bridge_id=bridge.bridge_id,
            evaluation_correspondence_hash=correspondence.content_hash,
            scorer=scorer_id,
            evaluator_version=code_identity(),
        )

    # -- evidence -------------------------------------------------------
    def record_evaluation(self, card: EvaluationCard) -> EvaluationCard:
        self.evaluations.append(card)
        return card

    def record_calibration(self, record: CalibrationRecord) -> CalibrationRecord:
        self.calibrations.append(record)
        return record

    def check_compression(self, **kwargs) -> CompressionRecord:
        return check_compression(**kwargs)

    # -- result inspection ------------------------------------------------
    def inspect_result(
        self,
        *,
        query_vector,
        candidate_vector,
        context_vectors,
        candidate_index: int,
        k: int = 10,
        metric: str = "cosine",
        scorer: ScoreFn | None = None,
        scorer_id: str = "",
        space_hash: str = "",
        observation: HardNegativeObservation | None = None,
        margin: float | None = None,
        calibration: CalibrationRecord | None = None,
        calibration_id: str = "",
        external: ExternalSignals | None = None,
    ) -> SignalBundle:
        """Compose a SignalBundle for one retrieval result.

        Orchestration only: the score comes from the injected scorer
        (cosine by default), geometry from the existing neighborhood
        primitives, the margin from the supplied 3A observation, and the
        calibration outcome from the supplied record. Search behavior is
        unchanged; this is the inspect-after-search step.
        """
        scorer = scorer or cosine_scorer()
        query = np.asarray(query_vector, dtype=np.float64)
        candidate = np.asarray(candidate_vector, dtype=np.float64)
        context = np.asarray(context_vectors, dtype=np.float64)
        if context.ndim != 2 or not np.isfinite(context).all():
            raise RelateError("context_vectors must be a finite matrix")
        if not 0 <= candidate_index < context.shape[0]:
            raise RelateError("candidate_index is out of range")
        score = float(scorer(query, candidate))

        density = local_density(context, candidate_index, k=k, metric=metric)
        counts = hubness_counts(context, k=k, metric=metric)
        hubness = make_hubness(counts[candidate_index], context.shape[0], k)
        extended = np.vstack([context, query[None, :]])
        stability = shared_neighborhood_stability(
            extended, context.shape[0], candidate_index, k=k, metric=metric
        )
        return build_signal_bundle(
            score=score,
            observation=observation,
            margin=margin,
            density=density,
            hubness=hubness,
            stability=stability,
            calibration=calibration,
            external=external,
            scorer_id=scorer_id or getattr(scorer, "scorer_id", ""),
            space_hash=space_hash,
            calibration_id=calibration_id,
        )
