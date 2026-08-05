from __future__ import annotations

from pathlib import Path

import numpy as np

import relate.export_space_artifact_exact as exact
from relate.model import RelationProjection


def _projection() -> RelationProjection:
    rng = np.random.default_rng(7)
    return RelationProjection(
        coefficients=rng.standard_normal((768, 3)),
        intercept=rng.standard_normal(3),
        relation_median=np.zeros(3),
        relation_scale=np.ones(3),
        embedding_mean=np.zeros(768),
        alpha=1.0,
        relation_names=("a", "b", "c"),
    )


def test_historical_independent_project_uses_one_vector_operation_per_relation() -> None:
    rng = np.random.default_rng(11)
    projection = _projection()
    embeddings = rng.standard_normal((32, 768))

    expected = np.column_stack(
        [
            embeddings @ projection.coefficients[:, index]
            + projection.intercept[index]
            for index in range(projection.relation_dimensions)
        ]
    )
    actual = exact.historical_independent_project(projection, embeddings)

    assert np.array_equal(actual, expected)


def test_export_wrapper_restores_normal_projection_method(
    monkeypatch,
    tmp_path: Path,
) -> None:
    projection = _projection()
    embeddings = np.arange(2 * 768, dtype=np.float64).reshape(2, 768)
    expected = exact.historical_independent_project(projection, embeddings)
    original = RelationProjection.project

    def fake_base_export(**kwargs):
        assert np.array_equal(projection.project(embeddings), expected)
        return {"verification": {}}

    monkeypatch.setattr(exact, "_base_export", fake_base_export)
    output = tmp_path / "projection.npz"
    result = exact.export_space_artifact(
        canonical_root=tmp_path / "canonical",
        cache_path=tmp_path / "cache.sqlite3",
        output_path=output,
    )

    assert RelationProjection.project is original
    assert output.with_suffix(".json").is_file()
    assert result["verification"]["prediction_execution"].startswith("three independent")
