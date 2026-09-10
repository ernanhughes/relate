"""Neighborhood tests: stability is measured, never decided."""

import numpy as np
import pytest

from relate import RelateError
from relate.evaluation import (
    SpaceComparisonReport,
    compare_geometry,
    compare_neighborhoods,
    counterpart_recovery,
)


def _space(seed=0, n=120, d=24):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, d))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _ids(n):
    return [f"item-{i}" for i in range(n)]


def test_identical_spaces_agree_everywhere():
    vectors = _space()
    ids = _ids(len(vectors))
    report = compare_neighborhoods(vectors, vectors, ids, k=10)
    assert report.mean_overlap == pytest.approx(1.0)
    assert report.top1_agreement == pytest.approx(1.0)
    assert report.rank_correlation == pytest.approx(1.0)
    assert report.per_query[0].overlap == pytest.approx(1.0)


def _clustered(seed=0, n_clusters=6, per_cluster=20, d=24):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, d)) * 3.0
    blocks = [c + 0.3 * rng.normal(size=(per_cluster, d)) for c in centers]
    vectors = np.vstack(blocks)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_neighbor_swap_keeps_neighborhoods_kills_counterpart():
    # Each candidate becomes its reference row's neighborhood centroid:
    # surrounded by the same cluster-mates (neighborhoods live) yet no
    # candidate is its own row's nearest match (counterpart dies).
    vectors = _clustered()
    ids = _ids(len(vectors))
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = unit @ unit.T
    np.fill_diagonal(sims, -np.inf)
    top = np.argsort(-sims, axis=1)[:, :10]
    rng = np.random.default_rng(2)
    centroids = np.array([vectors[rows].mean(axis=0) for rows in top])
    centroids += 0.01 * rng.normal(size=centroids.shape)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    neighborhood = compare_neighborhoods(vectors, centroids, ids, k=10)
    counterpart = counterpart_recovery(vectors, centroids, ids)
    assert neighborhood.mean_overlap > 0.5
    assert counterpart.top1 < 0.2


def test_noise_destroys_both_together():
    vectors = _space()
    ids = _ids(len(vectors))
    rng = np.random.default_rng(9)
    noise = rng.normal(size=vectors.shape)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    assert compare_neighborhoods(vectors, noise, ids, k=10).mean_overlap < 0.25
    assert counterpart_recovery(vectors, noise, ids).top1 < 0.1


def test_counterpart_recall_at_k_shape():
    vectors = _space(n=60)
    ids = _ids(len(vectors))
    report = counterpart_recovery(vectors, vectors, ids, ks=(1, 5, 10))
    assert report.top1 == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)
    assert report.recall_at_k[5] == pytest.approx(1.0)


def test_geometry_comparison_has_no_single_score():
    vectors = _space(n=80)
    ids = _ids(len(vectors))
    rng = np.random.default_rng(11)
    other = vectors + 0.05 * rng.normal(size=vectors.shape)
    comparison = compare_geometry(
        vectors, other, ids, reference_space_hash="a", candidate_space_hash="b"
    )
    assert comparison.cka is not None and comparison.cka > 0.9
    assert comparison.cosine_matrix_correlation is not None
    assert comparison.distance_matrix_correlation is not None
    assert comparison.neighborhood is not None
    assert not hasattr(comparison, "compatibility_score")


def test_space_report_is_evidence_not_permission():
    report = SpaceComparisonReport(
        source_space_hash="a",
        target_space_hash="b",
    )
    assert report.geometry is None
    assert report.counterpart is None
    assert report.hard_negatives is None
    assert not hasattr(report, "usable_for")


def test_neighborhood_failures_are_loud():
    vectors = _space(n=10)
    with pytest.raises(RelateError, match="same row count"):
        compare_neighborhoods(vectors, vectors[:5], _ids(10))
    with pytest.raises(RelateError, match="unique"):
        compare_neighborhoods(vectors, vectors, ["x"] * 10)
    with pytest.raises(RelateError, match="at least three"):
        compare_neighborhoods(vectors[:2], vectors[:2], ["a", "b"])
