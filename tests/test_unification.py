"""Unification tests: one lifecycle for every producer family."""

import numpy as np
import pytest

from relate import (
    Observatory,
    RelateError,
    SpaceIdentity,
    TransformationEvaluation,
)
from relate.evaluation import (
    DEFAULT_POLICIES,
    ReferenceFrame,
    identity_correspondence,
)
from relate.transformations import (
    derived_transformation_space,
    fit_constant_delta,
    fit_pca,
    identity_map,
)


def _bridge_fixture(seed=11, n=60, dims=8):
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(n, dims))
    target = source @ rng.normal(size=(dims, dims)) + 0.05 * rng.normal(size=(n, dims))
    return source, target


def _runtime_pair(model_a="uni-a", model_b="uni-b", dims=8):
    runtime = Observatory()
    space_a = runtime.register_space(model=model_a, dimensions=dims)
    space_b = runtime.register_space(model=model_b, dimensions=dims)
    return runtime, space_a, space_b


def test_bridge_path_delegates_to_generic():
    source, target = _bridge_fixture()
    ids = [f"u-{i}" for i in range(len(source))]
    correspondence = identity_correspondence(ids)
    runtime, space_a, space_b = _runtime_pair()
    bridge = runtime.fit_bridge(
        source, target, source_space=space_a, target_space=space_b,
        correspondence=correspondence, method="ridge", params={"alpha": 1.0},
    )
    full = runtime.evaluate_bridge_full(
        bridge, source, target, correspondence=correspondence,
        source_space=space_a, target_space=space_b,
        policies=list(DEFAULT_POLICIES),
    )
    generic = runtime.evaluate_transformation(
        bridge, source, target, correspondence=correspondence,
        reference_frame=ReferenceFrame.TARGET_NATIVE,
        candidate_space=full.candidate_space, reference_space=space_b,
        policies=list(DEFAULT_POLICIES),
    )
    assert isinstance(generic, TransformationEvaluation)
    assert [v.scope for v in generic.profile.verdicts] == [
        v.scope for v in full.profile.verdicts]
    assert [v.verdict for v in generic.profile.verdicts] == [
        v.verdict for v in full.profile.verdicts]
    for result in full.profile.results:
        twin = generic.profile.result(result.capability, result.metric)
        assert twin is not None
        assert twin.value == pytest.approx(result.value)
    assert generic.comparison.geometry.cka == pytest.approx(
        full.comparison.geometry.cka)


def test_compression_through_canonical_path():
    rng = np.random.default_rng(12)
    vectors = rng.normal(size=(80, 16))
    ids = [f"c-{i}" for i in range(80)]
    runtime = Observatory()
    space = runtime.register_space(model="uni-pca", dimensions=16)
    cartridge = fit_pca(vectors, output_dimensions=8,
                        source_space_hash=space.space_hash)
    derived = runtime.derive_transformation_space(
        parent=space, transformation_id=cartridge.transformation_id,
        parameters={"method": "pca", "output_dimensions": 8}, dimensions=8)
    assert derived.space_hash != space.space_hash
    assert derived.derived_from == space.space_hash
    evaluation = runtime.evaluate_compression(
        cartridge, vectors, vectors, correspondence=identity_correspondence(ids),
        candidate_space=derived, reference_space=space,
        policies=list(DEFAULT_POLICIES),
    )
    assert evaluation.comparison.counterpart is None
    assert evaluation.profile.candidate_space_hash == derived.space_hash
    assert evaluation.reference_space.space_hash == space.space_hash


def test_operator_through_canonical_path():
    rng = np.random.default_rng(13)
    source = rng.normal(size=(50, 6))
    shift = np.zeros(6)
    shift[0] = 3.0
    target = source + shift
    runtime = Observatory()
    space = runtime.register_space(model="uni-op", dimensions=6)
    operator = fit_constant_delta(source[:35], target[:35], relation="weakened",
                                  source_space_hash=space.space_hash)
    correspondence = identity_correspondence([f"o-{i}" for i in range(50)])
    evaluation = runtime.evaluate_operator(
        operator, source, target, correspondence=correspondence,
        reference_space=space,
        policies=list(DEFAULT_POLICIES),
    )
    assert evaluation.profile.source_space_hash == space.space_hash
    plain = identity_map(relation="weakened", source_space_hash=space.space_hash)
    assert runtime.evaluate_operator(
        plain, source, target, correspondence=correspondence,
        reference_space=space, policies=list(DEFAULT_POLICIES)).profile is not None
    with pytest.raises(RelateError, match="relation-bound"):
        runtime.evaluate_operator(
            fit_pca(source, output_dimensions=3, source_space_hash=space.space_hash),
            source, target, correspondence=correspondence, reference_space=space)


def test_lineage_crosses_families_without_verdicts():
    rng = np.random.default_rng(14)
    native = rng.normal(size=(40, 16))
    runtime = Observatory()
    space = runtime.register_space(model="uni-lin", dimensions=16)
    cartridge = fit_pca(native, output_dimensions=8,
                        source_space_hash=space.space_hash)
    compressed = runtime.derive_transformation_space(
        parent=space, transformation_id=cartridge.transformation_id,
        parameters={"method": "pca", "output_dimensions": 8}, dimensions=8)
    narrowed = cartridge.transform(native)
    bridge = runtime.fit_bridge(
        narrowed, narrowed, source_space=compressed,
        target_space=runtime.register_space(model="uni-lin2", dimensions=8),
        correspondence=identity_correspondence([f"l-{i}" for i in range(40)]),
    )
    bridged = runtime.bridge_space(compressed, bridge,
                                   runtime.spaces.require(bridge.target_space_hash))
    chain = runtime.lineage(bridged.space_hash)
    assert [record.kind for record in chain] == ["bridge", "transformation"]
    assert chain[0].parent_hash == compressed.space_hash
    assert chain[1].parent_hash == space.space_hash
    assert runtime.lineage(space.space_hash) == ()
    for record in chain:
        assert not hasattr(record, "usable_for")
        assert not hasattr(record, "verdict")


def test_no_transitive_usability():
    from relate.transformations import VectorTransformation

    class _Composed:
        def __init__(self, first, second, transformation_id, source_space_hash):
            self._steps = (first, second)
            self.transformation_id = transformation_id
            self.source_space_hash = source_space_hash

        def transform(self, values):
            out = values
            for step in self._steps:
                out = step.transform(out)
            return np.asarray(out, dtype=np.float64)

    rng = np.random.default_rng(15)
    vectors = rng.normal(size=(30, 8))
    runtime = Observatory()
    space = runtime.register_space(model="uni-tr", dimensions=8)
    first = fit_pca(vectors, output_dimensions=8, source_space_hash=space.space_hash)
    second = fit_pca(vectors, output_dimensions=8, source_space_hash=space.space_hash)
    composed = _Composed(first, second, "composed:test", space.space_hash)
    assert isinstance(composed, VectorTransformation)
    fresh = runtime.derive_transformation_space(
        parent=space, transformation_id=composed.transformation_id,
        parameters={"composed_of": [first.transformation_id, second.transformation_id]},
        dimensions=8)
    assert fresh.space_hash not in (
        derived_transformation_space(
            space, first.transformation_id, {"m": "pca"}, dimensions=8).space_hash,
    )
    assert runtime.lineage(fresh.space_hash)[0].transformation_id == "composed:test"


def test_derived_hashes_stable_across_runs():
    parent = SpaceIdentity(model="pin-native", dimensions=32)
    first = derived_transformation_space(
        parent, "pin-tr", {"output_dimensions": 8}, dimensions=8)
    second = derived_transformation_space(
        parent, "pin-tr", {"output_dimensions": 8}, dimensions=8)
    assert first.space_hash == second.space_hash
    assert first.space_hash != parent.space_hash
