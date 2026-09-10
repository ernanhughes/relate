"""Cross-space tests: correspondence is explicit, comparison reuses 3A/3B."""

import numpy as np
import pytest

from relate import RelateError
from relate.evaluation import (
    CorrespondenceSet,
    HardNegativeCase,
    aligned_matrices,
    compare_native_spaces,
    compare_neighborhoods,
    cosine_scorer,
    counterpart_recovery,
    evaluate_hard_negatives,
    identity_correspondence,
    linear_cka,
    permuted_copy,
    require_same_space_for_mixing,
)


def _spaces(seed=0, n=120, d=24):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, d))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    other = base + 0.05 * rng.normal(size=(n, d))
    other /= np.linalg.norm(other, axis=1, keepdims=True)
    return base, other


def _ids(n):
    return [f"item-{i}" for i in range(n)]


def test_correspondence_rejects_ambiguity():
    ids = _ids(8)
    good = identity_correspondence(ids)
    assert good.size == 8
    assert len(good.content_hash) == 16
    assert identity_correspondence(ids).content_hash == good.content_hash
    with pytest.raises(RelateError, match="same length"):
        CorrespondenceSet(ids=tuple(ids), source_rows=(0,), target_rows=(0,))
    with pytest.raises(RelateError, match="unique"):
        CorrespondenceSet(ids=tuple(ids), source_rows=tuple(range(8)),
                          target_rows=(0,) * 8)
    with pytest.raises(RelateError, match="unique"):
        CorrespondenceSet(ids=("a", "a"), source_rows=(0, 1), target_rows=(0, 1))
    with pytest.raises(RelateError, match="mismatch"):
        CorrespondenceSet(ids=tuple(ids), source_rows=tuple(range(8)),
                          target_rows=tuple(range(8)), content_hash="deadbeef")


def test_row_order_never_defines_correspondence():
    base, _ = _spaces()
    ids = _ids(len(base))
    shuffled_rows = np.random.default_rng(3).permutation(len(base))
    shuffled = base[shuffled_rows]
    correspondence = CorrespondenceSet(
        ids=tuple(ids), source_rows=tuple(range(len(base))),
        target_rows=tuple(int(r) for r in shuffled_rows),
    )
    aligned_source, aligned_target, ordered = aligned_matrices(
        base, shuffled, correspondence
    )
    assert ordered == tuple(sorted(ids))
    position = ordered.index("item-5")
    np.testing.assert_allclose(aligned_source[position], base[5])
    np.testing.assert_allclose(
        aligned_target[position], shuffled[shuffled_rows[5]]
    )
    # Same matrices, shifted correspondence: the mapping drives the result.
    shifted = CorrespondenceSet(
        ids=tuple(ids), source_rows=tuple(range(len(base))),
        target_rows=tuple((r + 1) % len(base) for r in range(len(base))),
    )
    same = compare_native_spaces(
        source_vectors=base, target_vectors=base,
        correspondence=identity_correspondence(ids),
        source_space_hash="a", target_space_hash="a",
    )
    moved = compare_native_spaces(
        source_vectors=base, target_vectors=base,
        correspondence=shifted,
        source_space_hash="a", target_space_hash="a",
    )
    assert same.neighborhood.mean_overlap == pytest.approx(1.0)
    assert moved.neighborhood.mean_overlap < 0.2


def test_compare_allows_distinct_hashes_mixing_denied():
    base, other = _spaces()
    ids = _ids(len(base))
    report = compare_native_spaces(
        source_vectors=base, target_vectors=other,
        correspondence=identity_correspondence(ids),
        source_space_hash="space-a", target_space_hash="space-b",
    )
    assert report.source_space_hash == "space-a"
    assert report.target_space_hash == "space-b"
    assert report.correspondence_hash == identity_correspondence(ids).content_hash
    require_same_space_for_mixing("space-a", "space-a")
    with pytest.raises(RelateError, match="DENIED without a measured bridge"):
        require_same_space_for_mixing("space-a", "space-b")


def test_comparison_reuses_3a_3b_unchanged():
    base, other = _spaces()
    ids = _ids(len(base))
    correspondence = identity_correspondence(ids)
    cases = [
        HardNegativeCase(
            case_id="c0", anchor_id=ids[0], positive_id=ids[1], negative_id=ids[2],
            relation="r",
        )
    ]
    vector_map = {name: base[i] for i, name in enumerate(ids)}
    other_map = {name: other[i] for i, name in enumerate(ids)}
    report = compare_native_spaces(
        source_vectors=base, target_vectors=other,
        correspondence=correspondence,
        source_space_hash="a", target_space_hash="b",
        hard_negative_cases=cases,
        hard_negative_vectors=(vector_map, other_map),
        scorer=cosine_scorer(),
        scorer_id="cosine_similarity",
    )
    assert report.geometry.cka == pytest.approx(linear_cka(base, other))
    direct_neighborhood = compare_neighborhoods(base, other, ids, k=10)
    assert report.neighborhood.mean_overlap == pytest.approx(
        direct_neighborhood.mean_overlap
    )
    direct_counterpart = counterpart_recovery(base, other, ids)
    assert report.counterpart.top1 == pytest.approx(direct_counterpart.top1)
    assert report.hard_negatives.accuracy_delta == pytest.approx(
        evaluate_hard_negatives(cases, other_map, cosine_scorer()).accuracy
        - evaluate_hard_negatives(cases, vector_map, cosine_scorer()).accuracy
    )
    assert report.scorer == "cosine_similarity"
    assert report.k == 10


def test_permuted_control_is_chance():
    base, _ = _spaces()
    ids = _ids(len(base))
    report = compare_native_spaces(
        source_vectors=base, target_vectors=permuted_copy(base, seed=5),
        correspondence=identity_correspondence(ids),
        source_space_hash="a", target_space_hash="b",
    )
    assert report.neighborhood.mean_overlap < 0.2
    assert report.counterpart.top1 < 0.1


def test_no_bridge_no_policy_at_this_layer():
    import relate.evaluation.cross_space as module

    for name in ("fit_bridge", "usable_for", "compatibility_score", "Bridge"):
        assert not hasattr(module, name)
    report = compare_native_spaces(
        source_vectors=_spaces(n=4)[0], target_vectors=_spaces(n=4)[1],
        correspondence=identity_correspondence(["a", "b", "c", "d"]),
        source_space_hash="a", target_space_hash="b",
    )
    assert not hasattr(report, "usable_for")


def test_partial_hard_negative_spec_fails_loudly():
    base, other = _spaces(n=10)
    ids = _ids(10)
    with pytest.raises(RelateError, match="both vector maps"):
        compare_native_spaces(
            source_vectors=base, target_vectors=other,
            correspondence=identity_correspondence(ids),
            hard_negative_cases=[
                HardNegativeCase(case_id="c", anchor_id=ids[0],
                                 positive_id=ids[1], negative_id=ids[2], relation="r")
            ],
        )
