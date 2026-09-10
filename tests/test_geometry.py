"""Geometry tests: invariants, determinism, and honest summaries."""

import numpy as np
import pytest

from relate import RelateError
from relate.evaluation import (
    PairSamplingSpec,
    describe_geometry,
    effective_rank,
    linear_cka,
    participation_ratio,
    twonn_estimate,
)


def _rng_space(seed=0, n=400, d=32, anisotropy=0.0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, d))
    if anisotropy:
        vectors[:, 0] *= 1.0 + anisotropy
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_rotation_preserves_geometry():
    vectors = _rng_space()
    rng = np.random.default_rng(1)
    rotation, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    rotated = vectors @ rotation
    base = describe_geometry(vectors, with_twonn=False)
    turned = describe_geometry(rotated, with_twonn=False)
    assert base.cosine_mean == pytest.approx(turned.cosine_mean, abs=1e-9)
    assert base.cosine_std == pytest.approx(turned.cosine_std, abs=1e-9)
    assert base.effective_rank == pytest.approx(turned.effective_rank, rel=1e-9)
    assert base.participation_ratio == pytest.approx(
        turned.participation_ratio, rel=1e-9
    )


def test_scaling_keeps_cosine_moves_euclidean():
    vectors = _rng_space() * 3.0 + 0.5
    scaled = 2.5 * vectors
    base = describe_geometry(vectors, with_twonn=False)
    moved = describe_geometry(scaled, with_twonn=False)
    assert base.cosine_mean == pytest.approx(moved.cosine_mean, abs=1e-9)
    assert moved.norm_mean == pytest.approx(2.5 * base.norm_mean, rel=1e-9)


def test_translation_changes_cosine():
    vectors = _rng_space(anisotropy=2.0)
    shifted = vectors + 5.0
    base = describe_geometry(vectors, with_twonn=False)
    moved = describe_geometry(shifted, with_twonn=False)
    assert abs(base.cosine_mean - moved.cosine_mean) > 0.05


def test_anisotropy_raises_random_pair_cosine():
    # Axis-stretch concentrates vectors bipolarly: |cosine| grows (std),
    # while the signed mean stays near zero -- magnitude is space-relative.
    isotropic = describe_geometry(_rng_space(), with_twonn=False)
    stretched = describe_geometry(_rng_space(anisotropy=6.0), with_twonn=False)
    assert stretched.cosine_std > isotropic.cosine_std + 0.2


def test_effective_rank_counts_directions():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=(500, 4)) @ rng.normal(size=(4, 32))
    signal += 1e-6 * rng.normal(size=(500, 32))
    assert effective_rank(signal) == pytest.approx(4.0, abs=0.5)
    assert participation_ratio(signal) == pytest.approx(4.0, abs=0.75)


def test_twonn_recovers_uniform_cube_dimension():
    rng = np.random.default_rng(4)
    cube = rng.uniform(size=(1500, 5))
    estimate = twonn_estimate(cube)
    assert estimate.method == "twonn"
    assert estimate.n_samples == 1500
    assert 3.0 < estimate.estimate < 7.0


def test_cka_is_one_for_self_and_rotations():
    vectors = _rng_space(n=200)
    assert linear_cka(vectors, vectors) == pytest.approx(1.0)
    rng = np.random.default_rng(5)
    rotation, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    assert linear_cka(vectors, vectors @ rotation) == pytest.approx(1.0, abs=1e-9)
    noise = rng.normal(size=vectors.shape)
    assert linear_cka(vectors, noise) < 0.2
    assert 0.0 <= linear_cka(vectors, noise) <= 1.0


def test_sampling_is_deterministic_and_provenance_bound():
    vectors = _rng_space(n=500)
    spec = PairSamplingSpec(max_pairs=10_000, seed=7)
    first = describe_geometry(vectors, sampling=spec, with_twonn=False)
    second = describe_geometry(vectors, sampling=spec, with_twonn=False)
    assert first.cosine_mean == second.cosine_mean
    assert first.sampling["n_pairs"] == 10_000
    assert first.sampling["census"] is False
    assert first.sampling["seed"] == 7
    census = describe_geometry(vectors[:20], with_twonn=False)
    assert census.sampling["census"] is True


def test_geometry_failures_are_loud():
    with pytest.raises(RelateError, match="at least two"):
        describe_geometry(np.ones((1, 4)))
    with pytest.raises(RelateError, match="positive integer"):
        PairSamplingSpec(max_pairs=0)
    with pytest.raises(RelateError, match="same row count"):
        linear_cka(np.ones((10, 4)), np.ones((12, 4)))
