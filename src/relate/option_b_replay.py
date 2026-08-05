"""Read-only reconstruction and replay of the historical Option B result.

The replay consumes the canonical artifacts from ``similarity_is_relative`` and
local SQLite embedding caches. It never regenerates embeddings and never writes
to the source caches or canonical research tree.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from relate.replay import ReplayError

PRIMITIVES = (
    "cyclomatic_complexity",
    "max_control_depth",
    "distinct_call_sites",
)
METHODS = (
    "raw_cosine",
    "raw_euclidean",
    "token_length",
    "true_oracle",
    "predicted_executor",
)
METHOD_INDEX = {name: index for index, name in enumerate(METHODS)}
THRESHOLD = 0.10

_SELECTION_REPORT = "option-b-canonical-row-selection-v2.json"
_EMBEDDING_CHECKPOINT = (
    "embedding-reproduction-v2/option-b-independent-embedding-reproduction-v2.json"
)
_PROBE_CHECKPOINT = "probes-v1/option-b-primitive-probe-publication-v1.json"
_HARD_NEGATIVE_CHECKPOINT = (
    "hard-negative-manifest-v1/option-b-hard-negative-manifest-publication-v1.json"
)
_PUBLISHED_VERIFICATION = (
    "method-evaluation-v1/option-b-method-evaluation-independent-v1.json"
)
_CACHE_TABLE = "embeddings_v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    payload = (
        json.dumps(
            {"dtype": str(array.dtype), "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + array.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReplayError(f"Required JSON artifact does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"JSON artifact must contain an object: {path}")
    return value


def _verify_text_sha256(path: Path, expected: str) -> None:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected:
        return
    normalized = raw.replace(b"\r\n", b"\n")
    if hashlib.sha256(normalized).hexdigest() != expected:
        raise ReplayError(f"Text artifact hash mismatch: {path}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ReplayError(f"JSONL row is not an object: {path}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"Cannot read JSONL artifact: {path}") from exc
    return rows


def load_selection_inputs(
    selection_dir: Path,
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load and verify the frozen row identities, token counts, and primitives."""
    report = _load_json(selection_dir / _SELECTION_REPORT)
    if report.get("status") != "CANONICAL_ROW_SELECTION_V2_VERIFIED":
        raise ReplayError("Canonical Option B row selection is not verified")

    keys_by_split: dict[str, tuple[str, ...]] = {}
    hashes_by_split: dict[str, tuple[str, ...]] = {}
    tokens_by_split: dict[str, np.ndarray] = {}
    true_by_split: dict[str, np.ndarray] = {}

    for split in ("train", "validation", "test"):
        selected_path = selection_dir / f"option-b-selected-{split}-v2.jsonl"
        primitive_path = selection_dir / f"option-b-primitives-{split}-v2.jsonl"
        expected = report["artifacts"][split]
        _verify_text_sha256(selected_path, str(expected["selected_manifest"]["sha256"]))
        _verify_text_sha256(primitive_path, str(expected["primitive_table"]["sha256"]))
        selected = _load_jsonl(selected_path)
        primitives = _load_jsonl(primitive_path)
        expected_rows = int(expected["selected_manifest"]["rows"])
        if len(selected) != expected_rows or len(primitives) != expected_rows:
            raise ReplayError(f"{split} selection row count mismatch")

        keys = tuple(str(row["stable_key"]) for row in selected)
        primitive_keys = tuple(str(row["stable_key"]) for row in primitives)
        if keys != primitive_keys or len(set(keys)) != len(keys):
            raise ReplayError(f"{split} stable-key order is invalid")
        source_hashes = tuple(str(row["code_sha256"]) for row in selected)
        if any(len(value) != 64 for value in source_hashes):
            raise ReplayError(f"{split} source hashes are invalid")

        keys_by_split[split] = keys
        hashes_by_split[split] = source_hashes
        tokens_by_split[split] = np.asarray(
            [int(row["token_count"]) for row in selected], dtype=np.int64
        )
        true_by_split[split] = np.asarray(
            [[float(row[name]) for name in PRIMITIVES] for row in primitives],
            dtype=np.float64,
        )

    train_true = true_by_split["train"]
    median = np.median(train_true, axis=0)
    q25, q75 = np.percentile(train_true, (25, 75), axis=0)
    scale = np.maximum(q75 - q25, 1.0)
    return (
        keys_by_split,
        hashes_by_split,
        tokens_by_split["train"],
        tokens_by_split["test"],
        (train_true - median) / scale,
        (true_by_split["test"] - median) / scale,
    )


def load_cache_matrix(
    cache_path: Path,
    *,
    stable_keys: Sequence[str],
    source_hashes: Sequence[str],
    fingerprint_sha256: str,
    dimensions: int,
    expected_array_sha256: str,
) -> np.ndarray:
    """Reconstruct one canonical matrix from a fingerprinted read-only cache."""
    if not cache_path.is_file():
        raise ReplayError(f"Option B cache does not exist: {cache_path}")
    if len(stable_keys) != len(source_hashes):
        raise ReplayError("Stable-key and source-hash counts differ")

    connection = sqlite3.connect(f"{cache_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if _CACHE_TABLE not in tables:
            raise ReplayError(f"Cache is missing {_CACHE_TABLE}: {cache_path}")
        matrix = np.empty((len(stable_keys), dimensions), dtype=np.float32)
        missing: list[str] = []
        for index, (stable_key, source_hash) in enumerate(
            zip(stable_keys, source_hashes, strict=True)
        ):
            row = connection.execute(
                f"""
                SELECT dtype, dimensions, embedding, payload_sha256, array_sha256
                FROM {_CACHE_TABLE}
                WHERE stable_key = ?
                  AND source_sha256 = ?
                  AND extraction_fingerprint_sha256 = ?
                """,
                (stable_key, source_hash, fingerprint_sha256),
            ).fetchone()
            if row is None:
                if len(missing) < 5:
                    missing.append(stable_key)
                continue
            dtype, stored_dimensions, payload, payload_sha, row_array_sha = row
            if str(dtype) != "float32" or int(stored_dimensions) != dimensions:
                raise ReplayError(f"Cached row shape or dtype mismatch: {stable_key}")
            raw = bytes(payload)
            if hashlib.sha256(raw).hexdigest() != str(payload_sha):
                raise ReplayError(f"Cached payload hash mismatch: {stable_key}")
            vector = np.frombuffer(raw, dtype=np.float32)
            if vector.shape != (dimensions,) or not np.isfinite(vector).all():
                raise ReplayError(f"Cached vector is invalid: {stable_key}")
            if array_sha256(vector) != str(row_array_sha):
                raise ReplayError(f"Cached array hash mismatch: {stable_key}")
            matrix[index] = vector
        if missing:
            raise ReplayError(
                f"Cache is missing canonical rows; first stable keys: {missing}"
            )
    finally:
        connection.close()

    actual = array_sha256(matrix)
    if actual != expected_array_sha256:
        raise ReplayError(
            f"Reconstructed matrix hash mismatch for {cache_path}: "
            f"{actual} != {expected_array_sha256}"
        )
    return matrix


def reconstruct_verified_embeddings(
    *,
    cache_a: Path,
    cache_b: Path,
    checkpoint_path: Path,
    keys_by_split: dict[str, tuple[str, ...]],
    hashes_by_split: dict[str, tuple[str, ...]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint = _load_json(checkpoint_path)
    if checkpoint.get("status") != "CANONICAL_EMBEDDINGS_V2_REPRODUCED":
        raise ReplayError("Canonical Option B embedding checkpoint is incomplete")

    retained: dict[str, np.ndarray] = {}
    reports: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        expected = checkpoint["splits"][split]
        dimensions = int(expected["dimensions"])
        array_hash = str(expected["array_sha256"])
        fingerprint = str(expected["extraction_fingerprint_sha256"])
        matrix_a = load_cache_matrix(
            cache_a,
            stable_keys=keys_by_split[split],
            source_hashes=hashes_by_split[split],
            fingerprint_sha256=fingerprint,
            dimensions=dimensions,
            expected_array_sha256=array_hash,
        )
        matrix_b = load_cache_matrix(
            cache_b,
            stable_keys=keys_by_split[split],
            source_hashes=hashes_by_split[split],
            fingerprint_sha256=fingerprint,
            dimensions=dimensions,
            expected_array_sha256=array_hash,
        )
        if not np.array_equal(matrix_a, matrix_b):
            raise ReplayError(f"Option B cache A/B matrices differ for {split}")
        reports[split] = {
            "rows": int(matrix_a.shape[0]),
            "dimensions": int(matrix_a.shape[1]),
            "dtype": str(matrix_a.dtype),
            "array_sha256": array_hash,
            "extraction_fingerprint_sha256": fingerprint,
            "cache_a_cache_b_exact": True,
        }
        if split in {"train", "test"}:
            retained[split] = matrix_a
    return retained["train"], retained["test"], reports


def load_prediction_inputs(
    probe_dir: Path,
    *,
    train_rows: int,
    test_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = _load_json(probe_dir / Path(_PROBE_CHECKPOINT).name)
    if checkpoint.get("status") != "PRIMITIVE_PROBE_ARTIFACTS_PUBLISHED_PENDING_REVIEW":
        raise ReplayError("Canonical Option B probe checkpoint is incomplete")

    result: dict[str, np.ndarray] = {}
    for split, role, filename, rows in (
        ("train", "train_candidates", "option-b-predicted-train-candidates-v1.npy", train_rows),
        ("test", "test_queries", "option-b-predicted-test-queries-v1.npy", test_rows),
    ):
        path = probe_dir / filename
        expected = checkpoint["predictions"][role]
        if file_sha256(path) != str(expected["file_sha256"]):
            raise ReplayError(f"{split} prediction file hash mismatch")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float64 or values.shape != (rows, len(PRIMITIVES)):
            raise ReplayError(f"{split} prediction shape or dtype mismatch")
        if array_sha256(np.asarray(values)) != str(expected["array_sha256"]):
            raise ReplayError(f"{split} prediction array hash mismatch")
        result[split] = values
    return result["train"], result["test"]


def load_pair_inputs(
    manifest_dir: Path,
    *,
    train_keys: Sequence[str],
    test_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = _load_json(manifest_dir / Path(_HARD_NEGATIVE_CHECKPOINT).name)
    if checkpoint.get("status") != "HARD_NEGATIVE_MANIFEST_PUBLISHED_PENDING_REVIEW":
        raise ReplayError("Canonical Option B hard-negative checkpoint is incomplete")

    query_path = manifest_dir / "option-b-hard-negative-queries-v1.jsonl"
    pair_path = manifest_dir / "option-b-hard-negative-pairs-v1.jsonl.gz"
    _verify_text_sha256(
        query_path, str(checkpoint["artifacts"]["queries"]["file_sha256"])
    )
    if file_sha256(pair_path) != str(checkpoint["artifacts"]["pairs"]["file_sha256"]):
        raise ReplayError("Hard-negative pair archive hash mismatch")

    queries = _load_jsonl(query_path)
    if len(queries) != len(test_keys):
        raise ReplayError("Hard-negative query count mismatch")
    counts = np.asarray([int(row["selected_pair_count"]) for row in queries], dtype=np.int64)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    closer = np.empty(int(offsets[-1]), dtype=np.int32)
    farther = np.empty(int(offsets[-1]), dtype=np.int32)

    digest = hashlib.sha256()
    position = 0
    try:
        with gzip.open(pair_path, "rb") as handle:
            for line in handle:
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                query_index = int(row["query_index"])
                expected_position = int(offsets[query_index]) + int(row["pair_order"])
                if expected_position != position:
                    raise ReplayError("Hard-negative pair order mismatch")
                closer_index = int(row["closer_candidate_index"])
                farther_index = int(row["farther_candidate_index"])
                if str(row["query_stable_key"]) != test_keys[query_index]:
                    raise ReplayError("Hard-negative query stable-key mismatch")
                if str(row["closer_stable_key"]) != train_keys[closer_index]:
                    raise ReplayError("Hard-negative closer stable-key mismatch")
                if str(row["farther_stable_key"]) != train_keys[farther_index]:
                    raise ReplayError("Hard-negative farther stable-key mismatch")
                closer[position] = closer_index
                farther[position] = farther_index
                position += 1
    except (OSError, json.JSONDecodeError, IndexError) as exc:
        raise ReplayError("Cannot parse hard-negative pair stream") from exc

    if position != len(closer):
        raise ReplayError("Hard-negative pair row count mismatch")
    expected_uncompressed = str(
        checkpoint["artifacts"]["pairs"]["uncompressed_file_sha256"]
    )
    if digest.hexdigest() != expected_uncompressed:
        raise ReplayError("Hard-negative uncompressed stream hash mismatch")
    return closer, farther, offsets, counts


def _pair_score(closer: np.ndarray, farther: np.ndarray) -> float:
    ties = closer == farther
    return float(np.mean(np.where(ties, 0.5, closer < farther)))


def recompute_primary_scores(
    *,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_tokens: np.ndarray,
    test_tokens: np.ndarray,
    train_true: np.ndarray,
    test_true: np.ndarray,
    train_predictions: np.ndarray,
    test_predictions: np.ndarray,
    closer: np.ndarray,
    farther: np.ndarray,
    offsets: np.ndarray,
    progress_every: int = 250,
) -> np.ndarray:
    scores = np.empty((len(test_embeddings), len(METHODS)), dtype=np.float64)
    eps = np.finfo(np.float64).eps
    for query_index in range(len(test_embeddings)):
        if progress_every > 0 and query_index % progress_every == 0:
            print(
                f"[option-b-replay] primary query {query_index}/{len(test_embeddings)}",
                file=sys.stderr,
                flush=True,
            )
        start = int(offsets[query_index])
        end = int(offsets[query_index + 1])
        if start == end:
            scores[query_index] = np.nan
            continue
        left = closer[start:end]
        right = farther[start:end]
        query_embedding = np.asarray(test_embeddings[query_index], dtype=np.float64)
        left_embedding = np.asarray(train_embeddings[left], dtype=np.float64)
        right_embedding = np.asarray(train_embeddings[right], dtype=np.float64)
        query_norm = float(np.sqrt(np.sum(query_embedding * query_embedding)))

        def cosine(values: np.ndarray) -> np.ndarray:
            dots = np.sum(values * query_embedding, axis=1, dtype=np.float64)
            norms = np.sqrt(np.sum(values * values, axis=1, dtype=np.float64))
            return 1.0 - dots / np.maximum(norms * query_norm, eps)

        def euclidean(values: np.ndarray) -> np.ndarray:
            delta = values - query_embedding
            return np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float64))

        left_distances = (
            cosine(left_embedding),
            euclidean(left_embedding),
            np.abs(train_tokens[left] - test_tokens[query_index]).astype(np.float64),
            np.max(np.abs(train_true[left] - test_true[query_index]), axis=1),
            np.max(np.abs(train_predictions[left] - test_predictions[query_index]), axis=1),
        )
        right_distances = (
            cosine(right_embedding),
            euclidean(right_embedding),
            np.abs(train_tokens[right] - test_tokens[query_index]).astype(np.float64),
            np.max(np.abs(train_true[right] - test_true[query_index]), axis=1),
            np.max(np.abs(train_predictions[right] - test_predictions[query_index]), axis=1),
        )
        scores[query_index] = [
            _pair_score(left_distance, right_distance)
            for left_distance, right_distance in zip(left_distances, right_distances, strict=True)
        ]
    return scores


def decision_from_scores(scores: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    eligible = counts > 0
    point = {
        method: float(np.mean(scores[eligible, METHOD_INDEX[method]])) for method in METHODS
    }
    raw_best = max(point["raw_cosine"], point["raw_euclidean"])
    gap = point["predicted_executor"] - raw_best
    return {
        "query_equal_weighted_accuracy": point,
        "queries_with_pairs": int(np.count_nonzero(eligible)),
        "queries_without_pairs": int(np.count_nonzero(~eligible)),
        "raw_best": raw_best,
        "raw_best_methods": [
            method
            for method in ("raw_cosine", "raw_euclidean")
            if point[method] == raw_best
        ],
        "gap": gap,
        "threshold": THRESHOLD,
        "outcome": "REAL_PREMISE_SUPPORTED" if gap >= THRESHOLD else "REAL_PREMISE_FAILED",
        "inconclusive_band": False,
        "secondary_metric_override": False,
    }


def replay_option_b(
    *,
    canonical_root: Path,
    cache_a: Path,
    cache_b: Path,
    progress_every: int = 250,
) -> dict[str, Any]:
    """Reconstruct both canonical embedding runs and replay the historical decision."""
    selection_dir = canonical_root / "selection"
    (
        keys_by_split,
        hashes_by_split,
        train_tokens,
        test_tokens,
        train_true,
        test_true,
    ) = load_selection_inputs(selection_dir)
    train_embeddings, test_embeddings, embedding_report = reconstruct_verified_embeddings(
        cache_a=cache_a,
        cache_b=cache_b,
        checkpoint_path=canonical_root / _EMBEDDING_CHECKPOINT,
        keys_by_split=keys_by_split,
        hashes_by_split=hashes_by_split,
    )
    train_predictions, test_predictions = load_prediction_inputs(
        canonical_root / "probes-v1",
        train_rows=len(keys_by_split["train"]),
        test_rows=len(keys_by_split["test"]),
    )
    closer, farther, offsets, counts = load_pair_inputs(
        canonical_root / "hard-negative-manifest-v1",
        train_keys=keys_by_split["train"],
        test_keys=keys_by_split["test"],
    )
    scores = recompute_primary_scores(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        train_tokens=train_tokens,
        test_tokens=test_tokens,
        train_true=train_true,
        test_true=test_true,
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        closer=closer,
        farther=farther,
        offsets=offsets,
        progress_every=progress_every,
    )
    decision = decision_from_scores(scores, counts)

    published_path = canonical_root / _PUBLISHED_VERIFICATION
    published = _load_json(published_path)
    if published.get("status") != "OPTION_B_PRIMARY_DECISION_INDEPENDENTLY_RECOMPUTED":
        raise ReplayError("Published historical Option B verification is incomplete")
    if decision != published.get("scientific_decision"):
        raise ReplayError("Replayed Option B decision differs from the published verification")

    return {
        "status": "OPTION_B_HISTORICAL_REPLAY_COMPLETE",
        "scientific_scope": "historical_option_b_only",
        "relate_e01_affected": False,
        "canonical_root": str(canonical_root),
        "cache_a": str(cache_a),
        "cache_b": str(cache_b),
        "embedding_reconstruction": embedding_report,
        "hard_negative_pairs": int(len(closer)),
        "primary_query_score_array_sha256": array_sha256(scores),
        "scientific_decision": decision,
        "published_verification": {
            "path": str(published_path),
            "file_sha256": file_sha256(published_path),
            "exact_decision_match": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-root",
        type=Path,
        required=True,
        help="Path to artifacts/canonical/option-b in similarity_is_relative.",
    )
    parser.add_argument(
        "--cache-a",
        type=Path,
        default=Path(".writer/option-b/cache/gpu-batch10-a.sqlite3"),
    )
    parser.add_argument(
        "--cache-b",
        type=Path,
        default=Path(".writer/option-b/cache/gpu-batch10-b.sqlite3"),
    )
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = replay_option_b(
        canonical_root=args.canonical_root,
        cache_a=args.cache_a,
        cache_b=args.cache_b,
        progress_every=args.progress_every,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
