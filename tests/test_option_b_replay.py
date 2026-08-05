from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from relate.option_b_replay import (
    PRIMITIVES,
    array_sha256,
    decision_from_scores,
    file_sha256,
    recompute_primary_scores,
    replay_option_b,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_fixture(root: Path) -> tuple[Path, Path, Path]:
    canonical = root / "canonical"
    selection = canonical / "selection"
    probe = canonical / "probes-v1"
    hard = canonical / "hard-negative-manifest-v1"
    method = canonical / "method-evaluation-v1"
    embedding_checkpoint = canonical / "embedding-reproduction-v2"
    cache_dir = root / "cache"
    for directory in (selection, probe, hard, method, embedding_checkpoint, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected = {
        "train": [
            {"stable_key": "t0", "code_sha256": "0" * 64, "token_count": 10},
            {"stable_key": "t1", "code_sha256": "1" * 64, "token_count": 20},
            {"stable_key": "t2", "code_sha256": "2" * 64, "token_count": 30},
        ],
        "validation": [
            {"stable_key": "v0", "code_sha256": "3" * 64, "token_count": 15}
        ],
        "test": [
            {"stable_key": "q0", "code_sha256": "4" * 64, "token_count": 11},
            {"stable_key": "q1", "code_sha256": "5" * 64, "token_count": 29},
        ],
    }
    primitives = {
        "train": [
            {"stable_key": "t0", **dict.fromkeys(PRIMITIVES, 0)},
            {"stable_key": "t1", **dict.fromkeys(PRIMITIVES, 1)},
            {"stable_key": "t2", **dict.fromkeys(PRIMITIVES, 2)},
        ],
        "validation": [{"stable_key": "v0", **dict.fromkeys(PRIMITIVES, 1)}],
        "test": [
            {"stable_key": "q0", **dict.fromkeys(PRIMITIVES, 0)},
            {"stable_key": "q1", **dict.fromkeys(PRIMITIVES, 2)},
        ],
    }
    selection_report: dict[str, object] = {
        "status": "CANONICAL_ROW_SELECTION_V2_VERIFIED",
        "artifacts": {},
    }
    artifacts = selection_report["artifacts"]
    assert isinstance(artifacts, dict)
    for split in selected:
        selected_path = selection / f"option-b-selected-{split}-v2.jsonl"
        primitive_path = selection / f"option-b-primitives-{split}-v2.jsonl"
        _write_jsonl(selected_path, selected[split])
        _write_jsonl(primitive_path, primitives[split])
        artifacts[split] = {
            "selected_manifest": {
                "sha256": _sha256(selected_path),
                "rows": len(selected[split]),
            },
            "primitive_table": {
                "sha256": _sha256(primitive_path),
                "rows": len(primitives[split]),
            },
        }
    (selection / "option-b-canonical-row-selection-v2.json").write_text(
        json.dumps(selection_report), encoding="utf-8"
    )

    embeddings = {
        "train": np.asarray([[1, 0], [0, 1], [-1, 0]], dtype=np.float32),
        "validation": np.asarray([[0, 1]], dtype=np.float32),
        "test": np.asarray([[1, 0], [-1, 0]], dtype=np.float32),
    }
    fingerprints = {"train": "a" * 64, "validation": "b" * 64, "test": "c" * 64}
    checkpoint = {
        "status": "CANONICAL_EMBEDDINGS_V2_REPRODUCED",
        "splits": {
            split: {
                "dimensions": 2,
                "array_sha256": array_sha256(matrix),
                "extraction_fingerprint_sha256": fingerprints[split],
            }
            for split, matrix in embeddings.items()
        },
    }
    (embedding_checkpoint / "option-b-independent-embedding-reproduction-v2.json").write_text(
        json.dumps(checkpoint), encoding="utf-8"
    )

    for database_name in ("a.sqlite3", "b.sqlite3"):
        connection = sqlite3.connect(cache_dir / database_name)
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
                PRIMARY KEY (
                    stable_key, source_sha256, extraction_fingerprint_sha256
                )
            )
            """
        )
        for split, matrix in embeddings.items():
            for manifest_row, vector in zip(selected[split], matrix, strict=True):
                raw = vector.tobytes()
                connection.execute(
                    "INSERT INTO embeddings_v2 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        manifest_row["stable_key"],
                        manifest_row["code_sha256"],
                        fingerprints[split],
                        "float32",
                        2,
                        raw,
                        hashlib.sha256(raw).hexdigest(),
                        array_sha256(vector),
                    ),
                )
        connection.commit()
        connection.close()

    train_predictions = np.asarray(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64
    )
    test_predictions = np.asarray([[0, 0, 0], [2, 2, 2]], dtype=np.float64)
    train_prediction_path = probe / "option-b-predicted-train-candidates-v1.npy"
    test_prediction_path = probe / "option-b-predicted-test-queries-v1.npy"
    np.save(train_prediction_path, train_predictions, allow_pickle=False)
    np.save(test_prediction_path, test_predictions, allow_pickle=False)
    probe_checkpoint = {
        "status": "PRIMITIVE_PROBE_ARTIFACTS_PUBLISHED_PENDING_REVIEW",
        "predictions": {
            "train_candidates": {
                "file_sha256": file_sha256(train_prediction_path),
                "array_sha256": array_sha256(train_predictions),
            },
            "test_queries": {
                "file_sha256": file_sha256(test_prediction_path),
                "array_sha256": array_sha256(test_predictions),
            },
        },
    }
    (probe / "option-b-primitive-probe-publication-v1.json").write_text(
        json.dumps(probe_checkpoint), encoding="utf-8"
    )

    _write_jsonl(
        hard / "option-b-hard-negative-queries-v1.jsonl",
        [{"selected_pair_count": 2}, {"selected_pair_count": 2}],
    )
    pairs: list[dict[str, object]] = []
    for query_index, (closer, farthers) in enumerate(((0, (1, 2)), (2, (1, 0)))):
        for pair_order, farther in enumerate(farthers):
            pairs.append(
                {
                    "query_index": query_index,
                    "pair_order": pair_order,
                    "query_stable_key": selected["test"][query_index]["stable_key"],
                    "closer_candidate_index": closer,
                    "farther_candidate_index": farther,
                    "closer_stable_key": selected["train"][closer]["stable_key"],
                    "farther_stable_key": selected["train"][farther]["stable_key"],
                }
            )
    uncompressed = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode() for row in pairs
    )
    pair_path = hard / "option-b-hard-negative-pairs-v1.jsonl.gz"
    with pair_path.open("wb") as output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0
        ) as archive:
            archive.write(uncompressed)
    hard_checkpoint = {
        "status": "HARD_NEGATIVE_MANIFEST_PUBLISHED_PENDING_REVIEW",
        "artifacts": {
            "queries": {
                "file_sha256": _sha256(hard / "option-b-hard-negative-queries-v1.jsonl")
            },
            "pairs": {
                "file_sha256": file_sha256(pair_path),
                "uncompressed_file_sha256": hashlib.sha256(uncompressed).hexdigest(),
            },
        },
    }
    (hard / "option-b-hard-negative-manifest-publication-v1.json").write_text(
        json.dumps(hard_checkpoint), encoding="utf-8"
    )

    closer = np.asarray([0, 0, 2, 2], dtype=np.int32)
    farther = np.asarray([1, 2, 1, 0], dtype=np.int32)
    offsets = np.asarray([0, 2, 4], dtype=np.int64)
    counts = np.asarray([2, 2], dtype=np.int64)
    scores = recompute_primary_scores(
        train_embeddings=embeddings["train"],
        test_embeddings=embeddings["test"],
        train_tokens=np.asarray([10, 20, 30]),
        test_tokens=np.asarray([11, 29]),
        train_true=np.asarray([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64),
        test_true=np.asarray([[0, 0, 0], [2, 2, 2]], dtype=np.float64),
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        closer=closer,
        farther=farther,
        offsets=offsets,
        progress_every=0,
    )
    published = {
        "status": "OPTION_B_PRIMARY_DECISION_INDEPENDENTLY_RECOMPUTED",
        "scientific_decision": decision_from_scores(scores, counts),
    }
    (method / "option-b-method-evaluation-independent-v1.json").write_text(
        json.dumps(published), encoding="utf-8"
    )
    return canonical, cache_dir / "a.sqlite3", cache_dir / "b.sqlite3"


def test_replay_option_b_reconstructs_both_caches_and_matches_publication(
    tmp_path: Path,
) -> None:
    canonical, cache_a, cache_b = _build_fixture(tmp_path)

    result = replay_option_b(
        canonical_root=canonical,
        cache_a=cache_a,
        cache_b=cache_b,
        progress_every=0,
    )

    assert result["status"] == "OPTION_B_HISTORICAL_REPLAY_COMPLETE"
    assert result["hard_negative_pairs"] == 4
    assert result["published_verification"]["exact_decision_match"] is True
    assert result["embedding_reconstruction"]["train"]["cache_a_cache_b_exact"] is True
    assert result["relate_e01_affected"] is False


def test_replay_option_b_rejects_cache_divergence(tmp_path: Path) -> None:
    canonical, cache_a, cache_b = _build_fixture(tmp_path)
    connection = sqlite3.connect(cache_b)
    connection.execute(
        "UPDATE embeddings_v2 SET embedding = ? WHERE stable_key = 't0'",
        (np.asarray([0, 0], dtype=np.float32).tobytes(),),
    )
    connection.commit()
    connection.close()

    import pytest

    with pytest.raises(ValueError, match="payload hash mismatch"):
        replay_option_b(
            canonical_root=canonical,
            cache_a=cache_a,
            cache_b=cache_b,
            progress_every=0,
        )
