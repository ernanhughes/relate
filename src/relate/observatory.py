"""Observatory: compose spaces, relations, evidence and policy.

The runtime refuses to trust unmeasured things. Denial is the default;
permission must cite a record.

Identity is exact (``space_hash``), compatibility is empirical (measured
bridge), usability is policy (``usable_for(scope)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from relate.bridges.base import Bridge
from relate.bridges.linear import fit_bridge
from relate.bridges.registry import BridgeRegistry
from relate.evaluation.cards import EvaluationCard
from relate.evaluation.preservation import PreservationProfile, make_preservation_profile
from relate.model import RelateError, RelationProjection
from relate.relations.base import Relation, fit_relation
from relate.retrieval.calibration import CalibrationRecord
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

    # -- bridges --------------------------------------------------------
    def fit_bridge(self, source: np.ndarray, target: np.ndarray, *,
                   source_space: SpaceIdentity, target_space: SpaceIdentity,
                   method: str = "procrustes", **kwargs) -> Bridge:
        bridge = fit_bridge(
            source, target, method=method,
            source_hash=source_space.space_hash,
            target_hash=target_space.space_hash,
            **kwargs,
        )
        return self.bridges.register(bridge)

    def evaluate_bridge(
        self,
        bridge: Bridge,
        *,
        source: np.ndarray,
        target: np.ndarray,
        thresholds: dict | None = None,
    ) -> PreservationProfile:
        """Measure counterpart recall + neighborhood agreement.

        Counterpart recovery != structural fidelity: a bridge can recover the
        paired target while only partly rebuilding the neighborhood.
        """
        src = np.asarray(source, dtype=np.float64)
        tgt = np.asarray(target, dtype=np.float64)
        mapped = bridge.apply(src)
        # counterpart Recall@1 (cosine)
        mapped_n = mapped / np.linalg.norm(mapped, axis=1, keepdims=True).clip(min=1e-12)
        tgt_n = tgt / np.linalg.norm(tgt, axis=1, keepdims=True).clip(min=1e-12)
        sims = mapped_n @ tgt_n.T
        top1 = np.argmax(sims, axis=1)
        recall_at_1 = float(np.mean(top1 == np.arange(src.shape[0])))
        # neighborhood agreement@5 (mapped vs native target neighborhoods)
        k = min(5, tgt.shape[0] - 1)
        agree: list[float] = []
        tgt_sims = tgt_n @ tgt_n.T
        map_sims = mapped_n @ mapped_n.T
        for i in range(tgt.shape[0]):
            native = set(np.argsort(-tgt_sims[i])[1 : k + 1].tolist())
            mapped_nb = set(np.argsort(-map_sims[i])[1 : k + 1].tolist())
            agree.append(len(native & mapped_nb) / max(1, k))
        profile = make_preservation_profile(
            bridge.source_space_hash,
            bridge.target_space_hash,
            {"retrieval": recall_at_1,
             "neighborhood": float(np.mean(agree)) if agree else 0.0},
            thresholds=thresholds or {"retrieval": 0.8, "neighborhood": 0.7},
        )
        # Bridges are frozen dataclasses; register a copy carrying the profile.
        bridged = Bridge(
            source_space_hash=bridge.source_space_hash,
            target_space_hash=bridge.target_space_hash,
            direction=bridge.direction,
            method=bridge.method,
            mapping=np.asarray(bridge.mapping),
            anchor_coverage=bridge.anchor_coverage,
            status=bridge.status,
            preservation=profile,
        )
        self.bridges.register(bridged)
        return profile

    # -- evidence -------------------------------------------------------
    def record_evaluation(self, card: EvaluationCard) -> EvaluationCard:
        self.evaluations.append(card)
        return card

    def record_calibration(self, record: CalibrationRecord) -> CalibrationRecord:
        self.calibrations.append(record)
        return record

    def check_compression(self, **kwargs) -> CompressionRecord:
        return check_compression(**kwargs)
