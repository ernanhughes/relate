"""Search frozen embeddings by recoverable relation coordinates.

RELATE treats embedding geometry as measured evidence rather than permission.
"""

from relate.bridges import Bridge, BridgeRegistry, fit_bridge
from relate.evaluation import EvaluationCard, PreservationProfile
from relate.model import RelateError, RelationProjection, SearchHit
from relate.observatory import Observatory
from relate.python import PYTHON_RELATION_NAMES, PythonStructure, extract_python_structure
from relate.relations import Relation
from relate.retrieval import CalibrationRecord, RetrievalPolicy, SignalBundle
from relate.spaces import SpaceIdentity, SpaceRegistry, compare_spaces
from relate.transformations import CompressionRecord, TransformationRecord

__all__ = [
    "PYTHON_RELATION_NAMES",
    "Bridge",
    "BridgeRegistry",
    "CalibrationRecord",
    "CompressionRecord",
    "EvaluationCard",
    "Observatory",
    "PreservationProfile",
    "PythonStructure",
    "RelateError",
    "Relation",
    "RelationProjection",
    "RetrievalPolicy",
    "SearchHit",
    "SignalBundle",
    "SpaceIdentity",
    "SpaceRegistry",
    "TransformationRecord",
    "compare_spaces",
    "extract_python_structure",
    "fit_bridge",
]
