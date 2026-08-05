from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from relate.export_space_artifact import export_space_artifact
from relate.model import RelationProjection
from relate.option_b_replay import array_sha256, file_sha256
from relate.python import PYTHON_RELATION_NAMES


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _text_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parameter_sha(value: np.ndarray | float) -> str:
    return array_sha256(np.asarray(value, dtype=np.float64))


def _build_fixture(root: Path) -> tuple[Path, Path]:
    canonical = root / "canonical"
    selection = canonical / "selection"
    reproduction = canonical / "embedding-reproduction-v2"
    probes = canonical / "probes-v1"
    for directory in (selection, reproduction, probes):
        directory.mkdir(parents=True)

    train_selected = [
        {"stable_key": f"t{index}", "code_sha256": str(index) * 64, "token_count": 10}
        for index in range(5)
    ]
    test_selected = [
        {"stable_key": f"q{index}", "code_sha256": str(index + 5) * 64, "token_count": 10}
        for index in range(2)
    ]
    train_primitives = np.asarray(
        [
            [1, 0, 1],
            [2, 1, 1],
            [3, 1, 2],
            [4, 2, 3],
            [5, 3, 5],
        ],
        dtype=np.float64,
    )
    primitive_rows = [
        {
            "stable_key": row["stable_key"],
            **{
                name: float(train_primitives[index, column])
                for column, name in enumerate(PYTHON_RELATION_NAMES)
            },
        }
        for index, row in enumerate(train_selected)
    ]

    artifacts: dict[str, object] = {}
    for split, selected_rows, primitive_values in (
        ("train", train_selected, primitive_rows),
        ("test", test_selected, []),
    ):
        selected_path = selection / f"option-b-selected-{split}-v2.jsonl"
        primitive_path = selection / f"option-b-primitives-{split}-v2.jsonl"
        _write_jsonl(selected_path, selected_rows)
        _write_jsonl(primitive_path, primitive_values)
        artifacts[split] = {
            "selected_manifest": {
                "sha256": _text_sha(selected_path),
                "rows": len(selected_rows),
            },
            "primitive_table": {
                "sha256": _text_sha(primitive_path),
                "rows": len(primitive_values),
            },
        }
    (selection / "option-b-canonical-row-selection-v2.json").write_text(
        json.dumps(
            {
                "status": "CANONICAL_ROW_SELECTION_V2_VERIFIED",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )

    train_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    test_x = np.asarray([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]], dtype=np.float32)
    fingerprints = {"train": "a" * 64, "test": "b" * 64}
    (reproduction / "option-b-independent-embedding-reproduction-v2.json").write_text(
        json.dumps(
            {
                "status": "CANONICAL_EMBEDDINGS_V2_REPRODUCED",
                "splits": {
                    "train": {
                        "dimensions": 3,
                        "array_sha256": array_sha256(train_x),
                        "extraction_fingerprint_sha256": fingerprints["train"],
                    },
                    "test": {
                        "dimensions": 3,
                        "array_sha256": array_sha256(test_x),
                        "extraction_fingerprint_sha256": fingerprints["test"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    median = np.median(train_primitives, axis=0)
    q25, q75 = np.percentile(train_primitives, (25.0, 75.0), axis=0)
    scale = np.maximum(q75 - q25, 1.0)
    scaled = (train_primitives - median) / scale
    primitive_contracts: dict[str, object] = {}
    test_predictions = np.empty((len(test_x), len(PYTHON_RELATION_NAMES)), dtype=np.float64)
    for index, name in enumerate(PYTHON_RELATION_NAMES):
        model = Ridge(alpha=1.0).fit(np.asarray(train_x, dtype=np.float64), scaled[:, index])
        primitive_contracts[name] = {
            "selected_alpha": 1.0,
            "final_coefficient_sha256": _parameter_sha(model.coef_),
            "final_intercept_sha256": _parameter_sha(float(model.intercept_)),
        }
        test_predictions[:, index] = model.predict(np.asarray(test_x, dtype=np.float64))

    bundle_path = probes / "option-b-primitive-probe-bundle-v1.json"
    bundle_path.write_text(
        json.dumps(
            {
                "status": "PRIMITIVE_PROBE_FIT_COMPLETE_PENDING_PUBLICATION_REVIEW",
                "contract": {"primitives": primitive_contracts},
            }
        ),
        encoding="utf-8",
    )
    prediction_path = probes / "option-b-predicted-test-queries-v1.npy"
    np.save(prediction_path, test_predictions, allow_pickle=False)
    (probes / "option-b-primitive-probe-publication-v1.json").write_text(
        json.dumps(
            {
                "predictions": {
                    "test_queries": {
                        "file_sha256": file_sha256(prediction_path),
                        "array_sha256": array_sha256(test_predictions),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    cache = root / "cache.sqlite3"
    connection = sqlite3.connect(cache)
    connection.execute(
        """
        CREATE TABLE embeddings_v2 (
            stable_key TEXT,
            source_sha256 TEXT,
            extraction_fingerprint_sha256 TEXT,
            dtype TEXT,
            dimensions INTEGER,
            embedding BLOB,
            payload_sha256 TEXT,
            array_sha256 TEXT,
            PRIMARY KEY (stable_key, source_sha256, extraction_fingerprint_sha256)
        )
        """
    )
    for split, selected_rows, matrix in (
        ("train", train_selected, train_x),
        ("test", test_selected, test_x),
    ):
        for selected_row, vector in zip(selected_rows, matrix, strict=True):
            raw = vector.tobytes()
            connection.execute(
                "INSERT INTO embeddings_v2 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    selected_row["stable_key"],
                    selected_row["code_sha256"],
                    fingerprints[split],
                    "float32",
                    3,
                    raw,
                    hashlib.sha256(raw).hexdigest(),
                    array_sha256(vector),
                ),
            )
    connection.commit()
    connection.close()
    return canonical, cache


def test_export_space_artifact_reproduces_published_predictions(tmp_path: Path) -> None:
    canonical, cache = _build_fixture(tmp_path)
    output = tmp_path / "space" / "assets" / "projection.npz"

    result = export_space_artifact(
        canonical_root=canonical,
        cache_path=cache,
        output_path=output,
    )

    assert result["status"] == "SPACE_PROJECTION_EXPORTED_AND_VERIFIED"
    assert result["verification"]["published_test_predictions_exact"] is True
    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    projection = RelationProjection.load(output)
    assert projection.embedding_dimensions == 3
    assert projection.relation_names == PYTHON_RELATION_NAMES
