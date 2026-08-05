from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from relate.replay import (
    PairExample,
    ReplayError,
    inspect_embedding_cache,
    inspect_npz,
    inventory_assets,
    replay_pair_benchmark,
)


def _write_cache(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE embedding_contracts (
            contract_sha256 TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            base_url TEXT NOT NULL,
            truncate INTEGER NOT NULL,
            normalization TEXT NOT NULL,
            stored_dtype TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE embeddings (
            contract_sha256 TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            text_value TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            vector BLOB NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (contract_sha256, text_sha256)
        );
        """
    )
    connection.execute(
        "INSERT INTO embedding_contracts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "a" * 64,
            1,
            "ollama",
            "demo",
            "http://localhost",
            1,
            "l2-v1",
            "float32",
            "now",
        ),
    )
    vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    connection.execute(
        "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?)",
        ("a" * 64, "b" * 64, "hello", 3, vector.tobytes(), "now"),
    )
    connection.commit()
    connection.close()


def test_inspect_embedding_cache_is_read_only_and_reports_contracts(tmp_path: Path) -> None:
    database = tmp_path / "cache.sqlite3"
    _write_cache(database)
    before = database.read_bytes()
    report = inspect_embedding_cache(database)
    assert report["embedding_rows"] == 1
    assert report["contracts"][0]["model"] == "demo"
    assert report["contracts"][0]["minimum_dimension"] == 3
    assert database.read_bytes() == before


def test_inspect_npz_reads_headers_and_classifies_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.npz"
    np.savez_compressed(
        path,
        texts=np.asarray(["a", "b"]),
        embeddings=np.eye(2),
        model=np.asarray("demo"),
        dataset_sha256=np.asarray("c" * 64),
    )
    report = inspect_npz(path)
    assert report["classification"] == "benchmark_embedding_snapshot"
    assert report["arrays"]["embeddings"]["shape"] == [2, 2]
    assert report["small_values"]["model"] == "demo"


def _write_replay_fixture(root: Path) -> tuple[Path, Path]:
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True)
    texts = [f"text-{index}" for index in range(24)]
    embeddings = np.random.default_rng(11).normal(size=(24, 6))

    splits: dict[str, list[PairExample]] = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        rows: list[PairExample] = []
        start = split_index * 8
        for index in range(8):
            label = index % 2
            left = texts[start + index]
            right_index = start + ((index + (0 if label else 1)) % 8)
            rows.append(
                PairExample(
                    f"{split}:{index}",
                    split,
                    left,
                    texts[right_index],
                    label,
                )
            )
        splits[split] = rows
        path = manifest_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")

    metadata = {
        "benchmark": "synthetic",
        "requested_revision": "main",
        "resolved_revision": "d" * 40,
        "limits": {split: 8 for split in splits},
        "seed": 118,
        "splits": {
            split: {
                "rows": len(rows),
                "label_counts": {"0": 4, "1": 4},
                "sha256": hashlib.sha256(
                    (manifest_dir / f"{split}.jsonl").read_bytes()
                ).hexdigest(),
            }
            for split, rows in splits.items()
        },
    }
    (manifest_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_hash = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    snapshot = root / "real_embeddings.npz"
    np.savez_compressed(
        snapshot,
        texts=np.asarray(texts),
        embeddings=embeddings,
        model=np.asarray("demo"),
        dataset_sha256=np.asarray(manifest_hash),
    )
    return snapshot, manifest_dir


def test_replay_pair_benchmark_runs_from_snapshot_and_manifests(tmp_path: Path) -> None:
    snapshot, manifests = _write_replay_fixture(tmp_path)
    result = replay_pair_benchmark(snapshot=snapshot, manifest_dir=manifests)
    assert result["status"] == "PAIR_BENCHMARK_REPLAY_COMPLETE"
    assert set(result["scores"]) == {
        "Cosine only",
        "Absolute difference",
        "Elementwise product",
        "Residual only",
        "Full pair",
        "Shuffled labels",
    }
    assert isinstance(
        result["primary_comparison"]["full_pair_minus_cosine_balanced_accuracy"],
        float,
    )


def test_replay_refuses_snapshot_manifest_mismatch(tmp_path: Path) -> None:
    snapshot, manifests = _write_replay_fixture(tmp_path)
    with np.load(snapshot, allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    values["dataset_sha256"] = np.asarray("0" * 64)
    np.savez_compressed(snapshot, **values)
    with pytest.raises(ReplayError, match="does not match"):
        replay_pair_benchmark(snapshot=snapshot, manifest_dir=manifests)


def test_inventory_finds_database_and_npz(tmp_path: Path) -> None:
    cache = tmp_path / ".writer" / "benchmarks" / "embedding-cache.sqlite3"
    cache.parent.mkdir(parents=True)
    _write_cache(cache)
    snapshot, _ = _write_replay_fixture(tmp_path / "run")
    report = inventory_assets(tmp_path)
    assert report["sqlite"]["embedding_rows"] == 1
    assert report["npz_count"] == 1
    assert report["npz"][0]["path"] == str(snapshot)
