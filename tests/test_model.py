from __future__ import annotations

import numpy as np
import pytest

from relate import RelateError, RelationProjection


def _projection() -> RelationProjection:
    embeddings = np.asarray(
        [
            [-2.0, 1.0],
            [-1.0, -2.0],
            [0.0, 1.5],
            [1.0, -1.0],
            [2.0, 2.5],
            [3.0, -0.5],
        ]
    )
    relations = embeddings[:, :1]
    return RelationProjection.fit(
        embeddings,
        relations,
        alpha=1e-10,
        relation_names=("first_coordinate",),
    )


def test_relation_search_finds_information_cosine_misses() -> None:
    model = _projection()
    source = np.asarray([1.0, 1.0])
    targets = np.asarray(
        [
            [1.0, -1.0],  # same relation, poor cosine
            [2.0, 2.0],   # perfect cosine, wrong relation
            [-1.0, -1.0],
        ]
    )

    hits = model.search(source, targets, k=3)

    assert hits[0].index == 0
    assert hits[1].index == 1
    assert hits[0].relation_distance < hits[1].relation_distance
    assert hits[1].cosine_distance < hits[0].cosine_distance


def test_cosine_can_generate_candidates_then_relation_reranks() -> None:
    model = _projection()
    source = np.asarray([1.0, 1.0])
    targets = np.asarray([[1.0, -1.0], [2.0, 2.0], [-1.0, -1.0]])

    hits = model.search(source, targets, k=1, candidate_pool=2)

    assert hits[0].index == 0


def test_projection_round_trip(tmp_path) -> None:
    model = _projection()
    path = tmp_path / "relation.npz"
    model.save(path)
    restored = RelationProjection.load(path)

    sample = np.asarray([[0.5, 4.0], [2.5, -3.0]])
    np.testing.assert_allclose(restored.project(sample), model.project(sample))
    assert restored.relation_names == model.relation_names


def test_search_can_exclude_the_source_row() -> None:
    model = _projection()
    targets = np.asarray([[1.0, 1.0], [1.0, -1.0], [2.0, 2.0]])

    hits = model.search(targets[0], targets, k=2, exclude_index=0)

    assert [hit.index for hit in hits] == [1, 2]


def test_invalid_shapes_fail_loudly() -> None:
    with pytest.raises(RelateError, match="same rows"):
        RelationProjection.fit(np.ones((3, 2)), np.ones((2, 1)))

    model = _projection()
    with pytest.raises(RelateError, match="candidate_pool must be at least k"):
        model.search(np.ones(2), np.ones((4, 2)), k=3, candidate_pool=2)
