"""Transformation-contract tests: generic seam, bridges unchanged."""

from pathlib import Path

import numpy as np
import pytest

from relate import RelateError, SpaceIdentity
from relate.bridges import BridgeSpec, fit_bridge
from relate.evaluation import (
    DEFAULT_POLICIES,
    ReferenceFrame,
    compare_native_spaces,
    cosine_scorer,
    identity_correspondence,
    scorer_id_of,
)
from relate.transformations import (
    ContentTransformation,
    TransformationArtifact,
    TransformationProvenance,
    TransformationSpec,
    VectorTransformation,
    derived_transformation_space,
    evaluate_transformation,
    hash_parameters,
    make_transformation_id,
)


def _spaces(dims=8):
    return (SpaceIdentity(model="contract-a", dimensions=dims),
            SpaceIdentity(model="contract-b", dimensions=dims))


def test_spec_artifact_identity_deterministic():
    spec = TransformationSpec(kind="pca", source_space_hash="abc",
                              parameters={"dimensions": 32})
    first = make_transformation_id(spec, "params1")
    assert first == make_transformation_id(spec, "params1")
    assert first != make_transformation_id(spec, "params2")
    assert len(first) == 16
    artifact = TransformationArtifact(
        transformation_id=first, spec=spec, parameter_hash="params1",
        provenance=TransformationProvenance(source_space_hash="abc", kind="pca"),
    )
    assert artifact.provenance.kind == "pca"
    assert hash_parameters({"b": 1, "a": 2}) == hash_parameters({"a": 2, "b": 1})
    with pytest.raises(RelateError, match="non-empty"):
        TransformationSpec(kind="", source_space_hash="abc")
    with pytest.raises(RelateError, match="exact source"):
        TransformationSpec(kind="pca", source_space_hash="")


def test_bridge_satisfies_vector_contract_without_new_state():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(20, 8))
    target = rng.normal(size=(20, 8))
    ids = [f"a-{i}" for i in range(20)]
    bridge = fit_bridge(
        source_vectors=source, target_vectors=target,
        correspondence=identity_correspondence(ids),
        spec=BridgeSpec(source_space_hash="s", target_space_hash="t",
                        method="linear"),
    )
    assert isinstance(bridge, VectorTransformation)
    assert bridge.transformation_id == bridge.bridge_id
    assert not isinstance(bridge, ContentTransformation)
    assert not hasattr(bridge, "content_kind")


def test_derived_identity_generic_no_target_reference():
    parent, _ = _spaces(dims=32)
    first = derived_transformation_space(
        parent, "tr-1", {"dimensions": 8}, dimensions=8)
    assert first.space_hash != parent.space_hash
    assert first.derived_from == parent.space_hash
    assert first.dimensions == 8
    assert "tr-1" in first.derivation
    again = derived_transformation_space(
        parent, "tr-1", {"dimensions": 8}, dimensions=8)
    assert again.space_hash == first.space_hash
    other = derived_transformation_space(
        parent, "tr-2", {"dimensions": 8}, dimensions=8)
    assert other.space_hash != first.space_hash
    with pytest.raises(RelateError, match="non-empty"):
        derived_transformation_space(parent, "", {})


class _ScaleProducer:
    """Minimal non-bridge VectorTransformation for the generic path."""

    def __init__(self, factor, source_hash, transformation_id):
        self._factor = factor
        self.source_space_hash = source_hash
        self.transformation_id = transformation_id

    def transform(self, values):
        return np.asarray(values, dtype=np.float64) * self._factor


def _fixture():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(60, 8))
    target = source @ rng.normal(size=(8, 8)) + 0.05 * rng.normal(size=(60, 8))
    return source, target


def test_generic_path_reproduces_bridge_evaluation():
    from relate import Observatory

    source, target = _fixture()
    ids = [f"r-{i}" for i in range(60)]
    correspondence = identity_correspondence(ids)
    runtime = Observatory()
    space_a = runtime.register_space(model="a5", dimensions=8)
    space_b = runtime.register_space(model="b5", dimensions=8)
    bridge = runtime.fit_bridge(
        source, target, source_space=space_a, target_space=space_b,
        correspondence=correspondence, method="ridge", params={"alpha": 1.0},
    )
    full = runtime.evaluate_bridge_full(
        bridge, source, target, correspondence=correspondence,
        source_space=space_a, target_space=space_b,
        policies=list(DEFAULT_POLICIES),
    )
    generic = evaluate_transformation(
        transformation=bridge, source_vectors=source, reference_vectors=target,
        correspondence=correspondence, reference_frame=ReferenceFrame.TARGET_NATIVE,
        candidate_space_hash=full.profile.candidate_space_hash,
        reference_space_hash=space_b.space_hash,
        policies=list(DEFAULT_POLICIES),
    )
    assert generic.bridge_id == full.profile.bridge_id
    assert [v.scope for v in generic.verdicts] == [v.scope for v in full.profile.verdicts]
    assert [v.verdict for v in generic.verdicts] == [v.verdict for v in full.profile.verdicts]
    for result in full.profile.results:
        twin = generic.result(result.capability, result.metric)
        assert twin is not None
        assert twin.value == pytest.approx(result.value)
        assert twin.reference_frame == result.reference_frame


def test_generic_path_accepts_non_bridge_producer():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(60, 8))
    reference = source + 0.01 * rng.normal(size=(60, 8))
    ids = [f"r-{i}" for i in range(60)]
    producer = _ScaleProducer(1.0, "src-hash", "scale-1.0")
    assert isinstance(producer, VectorTransformation)
    profile = evaluate_transformation(
        transformation=producer, source_vectors=source, reference_vectors=reference,
        correspondence=identity_correspondence(ids),
        reference_frame=ReferenceFrame.SOURCE_NATIVE,
        candidate_space_hash="cand", reference_space_hash="ref",
        policies=list(DEFAULT_POLICIES),
    )
    assert profile.source_space_hash == "src-hash"
    assert profile.bridge_id == "scale-1.0"
    assert profile.usable_for("retrieval") is True
    with pytest.raises(RelateError, match="explicit ReferenceFrame"):
        evaluate_transformation(
            transformation=producer, source_vectors=source, reference_vectors=reference,
            correspondence=identity_correspondence(ids),
            reference_frame="target_native",
        )
    with pytest.raises(RelateError, match="VectorTransformation"):
        evaluate_transformation(
            transformation=object(), source_vectors=source, reference_vectors=reference,
            correspondence=identity_correspondence(ids),
            reference_frame=ReferenceFrame.SOURCE_NATIVE,
        )


def test_evaluate_bridge_full_unchanged():
    from relate import Observatory

    source, target = _fixture()
    ids = [f"r-{i}" for i in range(60)]
    runtime = Observatory()
    space_a = runtime.register_space(model="a6", dimensions=8)
    space_b = runtime.register_space(model="b6", dimensions=8)
    bridge = runtime.fit_bridge(
        source, target, source_space=space_a, target_space=space_b,
        correspondence=identity_correspondence(ids),
    )
    evaluation = runtime.evaluate_bridge_full(
        bridge, source, target, correspondence=identity_correspondence(ids),
        source_space=space_a, target_space=space_b,
    )
    assert evaluation.profile.candidate_space_hash == evaluation.candidate_space.space_hash
    assert evaluation.comparison.source_space_hash == evaluation.candidate_space.space_hash


def test_contract_module_measures_nothing():
    text = (Path(__file__).resolve().parent.parent / "src" / "relate"
            / "transformations" / "contract.py").read_text()
    for forbidden in ("compare_native_spaces", "build_preservation_profile",
                      "evaluate_hard_negatives", "linear_cka", "roc_curve",
                      "calibrate(", "import numpy.linalg", "argsort"):
        assert forbidden not in text, forbidden
