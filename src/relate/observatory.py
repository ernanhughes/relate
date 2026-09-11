"""Observatory: the one coherent runtime façade.

Canonical flow (each step delegates to the authoritative layer, none
reimplemented here):

producer (bridge, compression, operator)
  -> source representation -> transform -> derived candidate identity
  -> explicit authority/reference -> evaluate_transformation
  -> SpaceComparisonReport -> PreservationProfile
  -> usable_for(scope) / explain(scope).

Bridges, compression cartridges, and semantic operators differ in how
candidates are produced -- never in how candidates are identified,
measured, or judged. Lineage composes identity; preservation permission
never propagates transitively: every hop needs fresh measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from relate.bridges.base import (
    Bridge,
    BridgeMismatchError,
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
)
from relate.evaluation.hard_negatives import HardNegativeObservation
from relate.evaluation.neighborhoods import SpaceComparisonReport
from relate.evaluation.neighborhoods import (
    hubness_counts,
    local_density,
    make_hubness,
    shared_neighborhood_stability,
)
from relate.evaluation.preservation import (
    DEFAULT_POLICIES,
    PreservationProfile,
    ReferenceFrame,
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
from relate.spaces.identity import SpaceIdentity
from relate.spaces.registry import SpaceRegistry
from relate.transformations.contract import (
    DerivationRecord,
    VectorTransformation,
    derived_transformation_space,
)
from relate.transformations.records import CompressionRecord, check_compression


@dataclass(frozen=True, slots=True)
class TransformationEvaluation:
    """One transformation judgment: pure aggregate container.

    ``PreservationProfile`` stays the sole scoped verdict authority;
    this object only keeps the producer, the derived candidate and
    reference identities, the 4A comparison, and the profile together
    for one call chain.
    """

    transformation: object
    candidate_space: SpaceIdentity | None
    reference_space: SpaceIdentity | None
    comparison: SpaceComparisonReport
    profile: PreservationProfile


@dataclass(frozen=True, slots=True)
class BridgeEvaluation:
    """One bridge judgment: pure container over existing objects.

    ``PreservationProfile`` stays the sole scoped verdict authority;
    this object only keeps the bridge, its derived candidate identity,
    the 4A comparison, and the profile together for one call chain.
    """

    bridge: Bridge
    candidate_space: SpaceIdentity | None
    comparison: SpaceComparisonReport
    profile: PreservationProfile


@dataclass
class Observatory:
    """Embedding runtime: spaces, relations, evidence, policy."""

    spaces: SpaceRegistry = field(default_factory=SpaceRegistry)
    bridges: BridgeRegistry = field(default_factory=BridgeRegistry)
    evaluations: list = field(default_factory=list)
    calibrations: list = field(default_factory=list)
    derived: dict = field(default_factory=dict)

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
        self,
        source_vectors,
        target_vectors,
        *,
        correspondence: CorrespondenceSet,
        source_space: SpaceIdentity | None = None,
        target_space: SpaceIdentity | None = None,
        k: int = 10,
        hard_negative_cases=None,
        hard_negative_vectors=None,
        scorer=None,
        scorer_id: str = "",
    ) -> SpaceComparisonReport:
        """Native comparison through 4A machinery. Evidence only.

        Comparing spaces is allowed with distinct hashes; the report
        carries no compatibility verdict and no permission.
        """
        return compare_native_spaces(
            source_vectors=source_vectors,
            target_vectors=target_vectors,
            correspondence=correspondence,
            source_space_hash=source_space.space_hash if source_space else "",
            target_space_hash=target_space.space_hash if target_space else "",
            k=k,
            hard_negative_cases=hard_negative_cases,
            hard_negative_vectors=hard_negative_vectors,
            scorer=scorer,
            scorer_id=scorer_id,
        )

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
        """Fit a directional producer on explicit anchor correspondence.

        Source and target identities must already be registered: fitting
        against unknown spaces is refused, and a successful fit returns
        the registered artifact.
        """
        self.spaces.require(source_space.space_hash)
        self.spaces.require(target_space.space_hash)
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
        space = bridge_output_space(source_space, bridge, target_reference)
        self.spaces.register(space)
        self.derived[space.space_hash] = DerivationRecord(
            space_hash=space.space_hash,
            parent_hash=source_space.space_hash,
            transformation_id=bridge.bridge_id,
            kind="bridge",
            reference_hash=target_reference.space_hash,
        )
        return space

    def derive_transformation_space(
        self,
        *,
        parent: SpaceIdentity,
        transformation_id: str,
        parameters: dict | None,
        dimensions: int | None,
        kind: str = "transformation",
        reference_hash: str = "",
    ) -> SpaceIdentity:
        """Generic derived identity: the underlying mechanism for all kinds.

        ``bridge_output_space`` remains the bridge specialization (stable
        recipe, target-reference lineage); everything else flows through
        here. The space is registered and its lineage recorded.
        """
        space = derived_transformation_space(
            parent, transformation_id, parameters,
            dimensions=dimensions, kind=kind,
        )
        self.spaces.register(space)
        self.derived[space.space_hash] = DerivationRecord(
            space_hash=space.space_hash,
            parent_hash=parent.space_hash,
            transformation_id=transformation_id,
            kind=kind,
            reference_hash=reference_hash,
        )
        return space

    def lineage(self, space_hash: str) -> tuple[DerivationRecord, ...]:
        """Deterministic ancestry, candidate first: identity only.

        Carries no verdicts and authorizes nothing: preservation never
        propagates transitively, however clean the lineage looks.
        """
        chain: list[DerivationRecord] = []
        seen: set[str] = set()
        current: str = space_hash
        while current and current not in seen:
            seen.add(current)
            record = self.derived.get(current)
            if record is None:
                break
            chain.append(record)
            current = record.parent_hash
        return tuple(chain)

    def evaluate_bridge(
        self,
        bridge: Bridge,
        evaluation_source,
        evaluation_target,
        *,
        correspondence: CorrespondenceSet,
        source_space: SpaceIdentity | None = None,
        target_space: SpaceIdentity | None = None,
        hard_negative_cases=None,
        hard_negative_vectors=None,
        scorer=None,
        scorer_id: str = "",
        policies=None,
        k: int = 10,
    ) -> PreservationProfile:
        """Judge a bridge's candidates; identity mistakes fail first.

        A held-out correspondence is required -- training correspondence
        can never silently become evaluation correspondence. Supplied
        space identities must match the bridge direction, otherwise no
        measurement runs.
        """
        return self.evaluate_bridge_full(
            bridge,
            evaluation_source,
            evaluation_target,
            correspondence=correspondence,
            source_space=source_space,
            target_space=target_space,
            hard_negative_cases=hard_negative_cases,
            hard_negative_vectors=hard_negative_vectors,
            scorer=scorer,
            scorer_id=scorer_id,
            policies=policies,
            k=k,
        ).profile

    def evaluate_transformation(
        self,
        transformation: VectorTransformation,
        source_vectors,
        reference_vectors,
        *,
        correspondence: CorrespondenceSet,
        reference_frame: ReferenceFrame,
        candidate_space: SpaceIdentity | None = None,
        reference_space: SpaceIdentity | None = None,
        k: int = 10,
        with_counterpart: bool = True,
        hard_negative_cases=None,
        hard_negative_vectors=None,
        scorer=None,
        scorer_id: str = "",
        policies=None,
    ) -> TransformationEvaluation:
        """Canonical transformation evaluation: one path for every producer.

        The authority is required as a typed frame, never defaulted:
        friendly wrappers choose it only where the operation makes it
        unambiguous (bridge target, compression source, operator target).
        All measurement delegates to existing evaluators.
        """
        if not isinstance(transformation, VectorTransformation):
            raise RelateError("evaluation needs a VectorTransformation producer")
        candidate = transformation.transform(
            np.asarray(source_vectors, dtype=np.float64)
        )
        reference = np.asarray(reference_vectors, dtype=np.float64)
        candidate_hash = (
            candidate_space.space_hash
            if candidate_space is not None
            else f"derived:{transformation.transformation_id}"
        )
        comparison = compare_native_spaces(
            source_vectors=candidate,
            target_vectors=reference,
            correspondence=correspondence,
            source_space_hash=candidate_hash,
            target_space_hash=(
                reference_space.space_hash if reference_space is not None else ""
            ),
            k=k,
            with_counterpart=with_counterpart,
            hard_negative_cases=hard_negative_cases,
            hard_negative_vectors=hard_negative_vectors,
            scorer=scorer,
            scorer_id=scorer_id,
        )
        results = results_from_space_comparison(comparison, frame=reference_frame)
        profile = build_preservation_profile(
            source_space_hash=transformation.source_space_hash,
            candidate_space_hash=candidate_hash,
            target_space_hash=(
                reference_space.space_hash if reference_space is not None else ""
            ),
            results=results,
            policies=list(policies) if policies is not None else list(DEFAULT_POLICIES),
            bridge_id=transformation.transformation_id,
            evaluation_correspondence_hash=correspondence.content_hash,
            scorer=scorer_id,
            evaluator_version=code_identity(),
        )
        return TransformationEvaluation(
            transformation=transformation,
            candidate_space=candidate_space,
            reference_space=reference_space,
            comparison=comparison,
            profile=profile,
        )

    def evaluate_compression(
        self,
        transformation: VectorTransformation,
        source_vectors,
        reference_vectors,
        *,
        correspondence: CorrespondenceSet,
        reference_frame: ReferenceFrame = ReferenceFrame.SOURCE_NATIVE,
        candidate_space: SpaceIdentity | None = None,
        reference_space: SpaceIdentity | None = None,
        k: int = 10,
        hard_negative_cases=None,
        hard_negative_vectors=None,
        scorer=None,
        scorer_id: str = "",
        policies=None,
    ) -> TransformationEvaluation:
        """Compression façade: source-native authority, width-aware counterpart.

        Counterpart recovery needs shared coordinates, so width-changing
        candidates skip it automatically; every other measurement is the
        generic path unchanged.
        """
        candidate = transformation.transform(
            np.asarray(source_vectors, dtype=np.float64)
        )
        reference = np.asarray(reference_vectors, dtype=np.float64)
        with_counterpart = candidate.shape == reference.shape
        return self.evaluate_transformation(
            transformation,
            source_vectors,
            reference_vectors,
            correspondence=correspondence,
            reference_frame=reference_frame,
            candidate_space=candidate_space,
            reference_space=reference_space,
            k=k,
            with_counterpart=with_counterpart,
            hard_negative_cases=hard_negative_cases,
            hard_negative_vectors=hard_negative_vectors,
            scorer=scorer,
            scorer_id=scorer_id,
            policies=policies,
        )

    def evaluate_operator(
        self,
        transformation: VectorTransformation,
        source_vectors,
        reference_vectors,
        *,
        correspondence: CorrespondenceSet,
        reference_frame: ReferenceFrame = ReferenceFrame.TARGET_NATIVE,
        candidate_space: SpaceIdentity | None = None,
        reference_space: SpaceIdentity | None = None,
        k: int = 10,
        hard_negative_cases=None,
        hard_negative_vectors=None,
        scorer=None,
        scorer_id: str = "",
        policies=None,
    ) -> TransformationEvaluation:
        """Operator façade: relation-bound artifacts, transformed-content authority.

        The artifact must name the trained relation; the authority is the
        transformed content embedded normally. Then the generic path.
        """
        artifact = getattr(transformation, "artifact", None)
        relation = (
            artifact.spec.parameters.get("relation") if artifact is not None else ""
        )
        if not relation:
            raise RelateError(
                "operator evaluation needs a relation-bound artifact"
            )
        return self.evaluate_transformation(
            transformation,
            source_vectors,
            reference_vectors,
            correspondence=correspondence,
            reference_frame=reference_frame,
            candidate_space=candidate_space,
            reference_space=reference_space,
            k=k,
            hard_negative_cases=hard_negative_cases,
            hard_negative_vectors=hard_negative_vectors,
            scorer=scorer,
            scorer_id=scorer_id,
            policies=policies,
        )

    def evaluate_bridge_full(
        self,
        bridge: Bridge,
        evaluation_source,
        evaluation_target,
        *,
        correspondence: CorrespondenceSet,
        source_space: SpaceIdentity | None = None,
        target_space: SpaceIdentity | None = None,
        hard_negative_cases=None,
        hard_negative_vectors=None,
        scorer=None,
        scorer_id: str = "",
        policies=None,
        k: int = 10,
    ) -> BridgeEvaluation:
        """Bridge façade: identity checks, then the generic path.

        Behaviorally identical to the 4D implementation; only the
        science path moved underneath into ``evaluate_transformation``.
        """
        if source_space is not None and (
            source_space.space_hash != bridge.source_space_hash
        ):
            raise BridgeMismatchError(
                "BRIDGE SOURCE MISMATCH: bridge expects "
                f"{bridge.source_space_hash}, evaluation supplied "
                f"{source_space.space_hash}"
            )
        if target_space is not None and (
            target_space.space_hash != bridge.target_space_hash
        ):
            raise BridgeMismatchError(
                "BRIDGE TARGET MISMATCH: bridge expects "
                f"{bridge.target_space_hash}, evaluation supplied "
                f"{target_space.space_hash}"
            )
        try:
            registered_source = self.spaces.require(bridge.source_space_hash)
            registered_target = self.spaces.require(bridge.target_space_hash)
            resolved_space: SpaceIdentity | None = bridge_output_space(
                registered_source, bridge, registered_target
            )
        except RelateError:
            resolved_space = None
        generic = self.evaluate_transformation(
            bridge,
            evaluation_source,
            evaluation_target,
            correspondence=correspondence,
            reference_frame=ReferenceFrame.TARGET_NATIVE,
            candidate_space=resolved_space,
            reference_space=target_space,
            k=k,
            hard_negative_cases=hard_negative_cases,
            hard_negative_vectors=hard_negative_vectors,
            scorer=scorer,
            scorer_id=scorer_id,
            policies=policies,
        )
        return BridgeEvaluation(
            bridge=bridge,
            candidate_space=generic.candidate_space,
            comparison=generic.comparison,
            profile=generic.profile,
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
