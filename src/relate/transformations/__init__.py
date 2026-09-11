"""Transformations package."""

from relate.transformations.compression import (
    PCACompression,
    PrefixTruncation,
    RandomProjection,
    fit_pca,
    prefix_truncation,
    random_projection,
)
from relate.transformations.contract import (
    ContentTransformation,
    TransformationArtifact,
    TransformationProvenance,
    TransformationSpec,
    VectorTransformation,
    as_vector_transformation,
    derived_transformation_space,
    hash_parameters,
    make_transformation_id,
)
from relate.transformations.evaluate import evaluate_transformation
from relate.transformations.records import (
    OPERATOR_CLASSES,
    CompressionRecord,
    TransformationRecord,
    check_compression,
)

__all__ = [
    "OPERATOR_CLASSES",
    "CompressionRecord",
    "ContentTransformation",
    "PCACompression",
    "PrefixTruncation",
    "RandomProjection",
    "TransformationArtifact",
    "TransformationProvenance",
    "TransformationRecord",
    "TransformationSpec",
    "VectorTransformation",
    "as_vector_transformation",
    "check_compression",
    "derived_transformation_space",
    "evaluate_transformation",
    "fit_pca",
    "hash_parameters",
    "make_transformation_id",
    "prefix_truncation",
    "random_projection",
]
