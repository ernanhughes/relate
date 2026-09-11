"""Search frozen embeddings by recoverable relation coordinates.

RELATE treats embedding geometry as measured evidence rather than permission.
"""

from relate.bridges import Bridge, BridgeMismatchError, BridgeRegistry, BridgeSpec, fit_bridge
from relate.evaluation import EvaluationCard, PreservationProfile
from relate.model import RelateError, RelationProjection, SearchHit
from relate.observatory import BridgeEvaluation, Observatory, TransformationEvaluation
from relate.python import PYTHON_RELATION_NAMES, PythonStructure, extract_python_structure
from relate.relations import Relation
from relate.retrieval import (
    CalibrationRecord,
    ExternalSignals,
    RetrievalPolicy,
    SignalBundle,
    SignalProvenance,
    build_signal_bundle,
)
from relate.spaces import SpaceIdentity, SpaceRegistry, compare_spaces
from relate.transformations import CompressionRecord, TransformationRecord

__all__ = [
    "PYTHON_RELATION_NAMES",
    "Bridge",
    "BridgeEvaluation",
    "BridgeMismatchError",
    "BridgeRegistry",
    "BridgeSpec",
    "CalibrationRecord",
    "CompressionRecord",
    "EvaluationCard",
    "ExternalSignals",
    "Observatory",
    "PreservationProfile",
    "PythonStructure",
    "RelateError",
    "Relation",
    "RelationProjection",
    "RetrievalPolicy",
    "SearchHit",
    "SignalBundle",
    "SignalProvenance",
    "SpaceIdentity",
    "SpaceRegistry",
    "TransformationEvaluation",
    "TransformationRecord",
    "build_signal_bundle",
    "compare_spaces",
    "extract_python_structure",
    "fit_bridge",
]
