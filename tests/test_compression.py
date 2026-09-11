"""Compression tests: producers on the 5A contract, judgment elsewhere."""

from pathlib import Path

import numpy as np
import pytest

from relate import RelateError, SpaceIdentity
from relate.evaluation import ReferenceFrame
from relate.transformations import (
    VectorTransformation,
    ContentTransformation,
    derived_transformation_space,
    evaluate_transformation,
    fit_pca,
    prefix_truncation,
    random_projection,
)


def _matrix(seed=0, n=200, d=64):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, d))
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def test_producers_satisfy_vector_contract():
    space_hash = "srchash"
    pca = fit_pca(_matrix(), output_dimensions=16, source_space_hash=space_hash)
    rand = random_projection(64, 16, seed=0, source_space_hash=space_hash)
    prefix = prefix_truncation(64, 16, source_space_hash=space_hash)
    for producer in (pca, rand, prefix):
        assert isinstance(producer, VectorTransformation)
        assert not isinstance(producer, ContentTransformation)
        out = producer.transform(_matrix(seed=1))
        assert out.shape == (200, 16)
        assert np.isfinite(out).all()
        assert len(producer.transformation_id) == 16
        assert producer.source_space_hash == space_hash
    assert pca.output_dimensions == 16
    assert pca.source_dimensions == 64


def test_fit_transform_split_and_determinism():
    first = fit_pca(_matrix(), output_dimensions=16, source_space_hash="s")
    second = fit_pca(_matrix(), output_dimensions=16, source_space_hash="s")
    assert first.transformation_id == second.transformation_id
    assert first.transform(_matrix(seed=2)).shape == (200, 16)
    assert 0.0 < sum(first.artifact.provenance.detail["explained_variance_ratio"]) <= 1.0
    assert first.artifact.provenance.detail["compression_ratio"] == pytest.approx(0.25)
    assert first.artifact.spec.parameters["output_dimensions"] == 16
    same_seed = random_projection(64, 16, seed=7, source_space_hash="s")
    assert (same_seed.transform(_matrix())
            == random_projection(64, 16, seed=7,
                                 source_space_hash="s").transform(_matrix())).all()
    assert (same_seed.transform(_matrix())
            != random_projection(64, 16, seed=8,
                                 source_space_hash="s").transform(_matrix())).any()


def test_prefix_neutral_naming():
    prefix = prefix_truncation(64, 32, source_space_hash="s")
    assert prefix.artifact.spec.kind == "prefix_truncation"
    assert prefix.artifact.provenance.detail["training_support"] == "unknown"
    matryoshka = prefix_truncation(64, 32, source_space_hash="s",
                                   training_support="matryoshka")
    assert matryoshka.transformation_id != prefix.transformation_id
    with pytest.raises(RelateError, match="training support"):
        prefix_truncation(64, 32, source_space_hash="s", training_support="clip")
    with pytest.raises(RelateError, match="widen"):
        prefix_truncation(16, 32, source_space_hash="s")


def test_contract_failures_early():
    with pytest.raises(RelateError, match="fit inside"):
        fit_pca(_matrix(), output_dimensions=0, source_space_hash="s")
    with pytest.raises(RelateError, match="fit inside"):
        fit_pca(_matrix(), output_dimensions=65, source_space_hash="s")
    with pytest.raises(RelateError, match="integer"):
        random_projection(64, 16, seed=True, source_space_hash="s")
    pca = fit_pca(_matrix(), output_dimensions=16, source_space_hash="s")
    with pytest.raises(RelateError, match="contract"):
        pca.transform(np.ones((4, 32)))
    with pytest.raises(RelateError, match="finite"):
        bad = _matrix()
        bad[0, 0] = np.inf
        pca.transform(bad)


def test_derived_identity_for_compressed_space():
    parent = SpaceIdentity(model="native-128", dimensions=128)
    pca = fit_pca(_matrix(n=200, d=128), output_dimensions=32,
                  source_space_hash=parent.space_hash)
    derived = derived_transformation_space(
        parent, pca.transformation_id,
        {"method": "pca", "output_dimensions": 32}, dimensions=32)
    assert derived.space_hash != parent.space_hash
    assert derived.derived_from == parent.space_hash
    assert derived.dimensions == 32


def test_compression_flows_through_generic_path():
    from relate.evaluation import DEFAULT_POLICIES, identity_correspondence

    vectors = _matrix(n=120, d=32)
    ids = [f"c-{i}" for i in range(120)]
    pca = fit_pca(vectors, output_dimensions=16, source_space_hash="src")
    profile = evaluate_transformation(
        transformation=pca, source_vectors=vectors, reference_vectors=vectors,
        correspondence=identity_correspondence(ids),
        reference_frame=ReferenceFrame.SOURCE_NATIVE,
        candidate_space_hash="cand", reference_space_hash="src",
        with_counterpart=False, policies=list(DEFAULT_POLICIES),
    )
    assert profile.bridge_id == pca.transformation_id
    assert profile.source_space_hash == "src"
    # Counterpart is meaningless across widths: unmeasured, fail closed.
    assert profile.usable_for("retrieval") is False
    assert "unmeasured" in profile.explain("retrieval")


def test_compression_module_computes_no_metrics():
    text = (Path(__file__).resolve().parent.parent / "src" / "relate"
            / "transformations" / "compression.py").read_text()
    for forbidden in ("compare_neighborhoods", "linear_cka", "roc_curve",
                      "evaluate_hard_negatives", "hubness_counts", "calibrate(",
                      "usable_for", "PASS", "PreservationProfile"):
        assert forbidden not in text, forbidden
