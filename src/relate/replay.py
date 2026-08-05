"""Read-only inventory and deterministic replay of preserved RELATE assets.

This module deliberately does not generate embeddings or manage experiments. It
inspects the existing SQLite cache and replays the external pair benchmarks from
frozen NPZ snapshots plus their frozen JSONL manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class ReplayError(ValueError):
    """Raised when preserved replay inputs are missing, mismatched, or invalid."""


@dataclass(frozen=True, slots=True)
class PairExample:
    example_id: str
    split: str
    left: str
    right: str
    label: int


@dataclass(frozen=True, slots=True)
class PairScore:
    name: str
    selected_alpha: float
    validation_balanced_accuracy: float | None
    accuracy: float
    balanced_accuracy: float
    f1: float
    roc_auc: float
    log_loss: float


FEATURE_NAMES = (
    "Cosine only",
    "Absolute difference",
    "Elementwise product",
    "Residual only",
    "Full pair",
)
_ALPHA_CANDIDATES = (1e-5, 1e-4, 1e-3, 1e-2)
_REQUIRED_SNAPSHOT_KEYS = {"texts", "embeddings", "model", "dataset_sha256"}
_REQUIRED_MANIFEST_SPLITS = ("train", "validation", "test")


def inspect_embedding_cache(path: str | Path) -> dict[str, Any]:
    """Inspect the old benchmark SQLite cache without creating or modifying it."""

    database = Path(path)
    if not database.is_file():
        raise ReplayError(f"SQLite cache does not exist: {database}")

    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        }
        required = {"embedding_contracts", "embeddings"}
        missing = sorted(required - tables)
        if missing:
            raise ReplayError(f"SQLite cache is missing tables: {', '.join(missing)}")

        contract_rows = connection.execute(
            """
            SELECT
                c.contract_sha256,
                c.schema_version,
                c.provider,
                c.model,
                c.base_url,
                c.truncate,
                c.normalization,
                c.stored_dtype,
                COUNT(e.text_sha256) AS rows,
                MIN(e.dimension) AS minimum_dimension,
                MAX(e.dimension) AS maximum_dimension
            FROM embedding_contracts AS c
            LEFT JOIN embeddings AS e
              ON e.contract_sha256 = c.contract_sha256
            GROUP BY c.contract_sha256
            ORDER BY c.model, c.contract_sha256
            """
        ).fetchall()
        contracts = [
            {
                "contract_sha256": str(row[0]),
                "schema_version": int(row[1]),
                "provider": str(row[2]),
                "model": str(row[3]),
                "base_url": str(row[4]),
                "truncate": bool(row[5]),
                "normalization": str(row[6]),
                "stored_dtype": str(row[7]),
                "rows": int(row[8]),
                "minimum_dimension": None if row[9] is None else int(row[9]),
                "maximum_dimension": None if row[10] is None else int(row[10]),
            }
            for row in contract_rows
        ]
        total_rows = int(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
    finally:
        connection.close()

    return {
        "path": str(database),
        "size_bytes": database.stat().st_size,
        "tables": sorted(tables),
        "contracts": contracts,
        "contract_count": len(contracts),
        "embedding_rows": total_rows,
    }


def inspect_npz(path: str | Path) -> dict[str, Any]:
    """Read NPZ array headers without expanding large embedding matrices in memory."""

    archive = Path(path)
    if not archive.is_file():
        raise ReplayError(f"NPZ file does not exist: {archive}")

    arrays: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive) as handle:
        for member in sorted(handle.namelist()):
            if not member.endswith(".npy"):
                continue
            key = member[:-4]
            with handle.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version in {(2, 0), (3, 0)}:
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
                else:  # pragma: no cover - NumPy currently defines versions 1-3
                    raise ReplayError(f"Unsupported NPY version {version} in {archive}")
            arrays[key] = {
                "shape": list(shape),
                "dtype": str(dtype),
                "fortran_order": bool(fortran_order),
            }

    scalar_values: dict[str, Any] = {}
    with np.load(archive, allow_pickle=False) as payload:
        for key, metadata in arrays.items():
            shape = tuple(metadata["shape"])
            element_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
            if element_count <= 16 and key != "embeddings":
                value = payload[key]
                scalar_values[key] = _small_array_value(value)

    keys = set(arrays)
    classification = (
        "benchmark_embedding_snapshot"
        if _REQUIRED_SNAPSHOT_KEYS.issubset(keys)
        else "generic_npz"
    )
    return {
        "path": str(archive),
        "size_bytes": archive.stat().st_size,
        "classification": classification,
        "arrays": arrays,
        "small_values": scalar_values,
    }


def inventory_assets(
    root: str | Path,
    *,
    cache_db: str | Path | None = None,
) -> dict[str, Any]:
    """Inventory preserved SQLite and NPZ inputs beneath a local checkout."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise ReplayError(f"Inventory root does not exist: {root_path}")

    database = (
        Path(cache_db)
        if cache_db is not None
        else root_path / ".writer" / "benchmarks" / "embedding-cache.sqlite3"
    )
    npz_files = sorted(root_path.rglob("*.npz"))
    return {
        "root": str(root_path),
        "sqlite": inspect_embedding_cache(database) if database.is_file() else None,
        "npz": [inspect_npz(path) for path in npz_files],
        "npz_count": len(npz_files),
    }


def replay_pair_benchmark(
    *,
    snapshot: str | Path,
    manifest_dir: str | Path,
    seed: int = 118,
) -> dict[str, Any]:
    """Replay the old PAWS/BigClone comparison without generating embeddings."""

    snapshot_path = Path(snapshot)
    manifests_path = Path(manifest_dir)
    texts, embeddings, model, dataset_sha256 = _load_snapshot(snapshot_path)
    examples, metadata, manifest_sha256 = _load_and_verify_manifests(manifests_path)
    if dataset_sha256 != manifest_sha256:
        raise ReplayError(
            "Snapshot dataset_sha256 does not match the frozen manifest metadata: "
            f"{dataset_sha256} != {manifest_sha256}"
        )

    embedding_by_text = dict(zip(texts, embeddings, strict=True))
    required_texts = {
        value
        for rows in examples.values()
        for row in rows
        for value in (row.left, row.right)
    }
    missing_texts = required_texts - embedding_by_text.keys()
    if missing_texts:
        preview = sorted(missing_texts)[:3]
        raise ReplayError(
            f"Snapshot is missing {len(missing_texts)} manifest texts; examples: {preview}"
        )

    split_features: dict[str, dict[str, np.ndarray]] = {}
    split_labels: dict[str, np.ndarray] = {}
    for split, rows in examples.items():
        left, right, labels = _pair_arrays(rows, embedding_by_text)
        split_features[split] = _build_features(left, right)
        split_labels[split] = labels

    scores = _evaluate_feature_sets(
        train_features=split_features["train"],
        validation_features=split_features["validation"],
        test_features=split_features["test"],
        train_labels=split_labels["train"],
        validation_labels=split_labels["validation"],
        test_labels=split_labels["test"],
        seed=seed,
    )
    by_name = {score.name: score for score in scores}
    return {
        "status": "PAIR_BENCHMARK_REPLAY_COMPLETE",
        "snapshot": str(snapshot_path),
        "manifest_dir": str(manifests_path),
        "manifest_sha256": manifest_sha256,
        "model": model,
        "embedding_dimension": int(embeddings.shape[1]),
        "unique_texts": len(texts),
        "seed": int(seed),
        "benchmark": metadata.get("benchmark"),
        "resolved_revision": metadata.get("resolved_revision"),
        "split_rows": {split: len(rows) for split, rows in examples.items()},
        "scores": {score.name: asdict(score) for score in scores},
        "primary_comparison": {
            "full_pair_minus_cosine_balanced_accuracy": (
                by_name["Full pair"].balanced_accuracy
                - by_name["Cosine only"].balanced_accuracy
            )
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and replay preserved RELATE assets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Inspect SQLite and NPZ assets.")
    inventory.add_argument("root", type=Path)
    inventory.add_argument("--cache-db", type=Path)
    inventory.add_argument("--output", type=Path)

    replay = subparsers.add_parser(
        "replay-pairs",
        help="Replay a frozen PAWS/BigClone benchmark from NPZ and manifests.",
    )
    replay.add_argument("--snapshot", type=Path, required=True)
    replay.add_argument("--manifests", type=Path, required=True)
    replay.add_argument("--seed", type=int, default=118)
    replay.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        payload = inventory_assets(args.root, cache_db=args.cache_db)
    elif args.command == "replay-pairs":
        payload = replay_pair_benchmark(
            snapshot=args.snapshot,
            manifest_dir=args.manifests,
            seed=args.seed,
        )
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled command: {args.command}")

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


def _load_snapshot(path: Path) -> tuple[list[str], np.ndarray, str, str]:
    if not path.is_file():
        raise ReplayError(f"Snapshot does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(_REQUIRED_SNAPSHOT_KEYS - set(payload.files))
        if missing:
            raise ReplayError(f"Snapshot is missing arrays: {', '.join(missing)}")
        texts = payload["texts"].astype(str).tolist()
        embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
        model = str(payload["model"].item())
        dataset_sha256 = str(payload["dataset_sha256"].item())

    if not texts or len(set(texts)) != len(texts):
        raise ReplayError("Snapshot texts must be non-empty and unique")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ReplayError("Snapshot embedding rows do not match snapshot texts")
    if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
        raise ReplayError("Snapshot embeddings must be a finite non-empty matrix")
    if not model or len(dataset_sha256) != 64:
        raise ReplayError("Snapshot model or dataset hash is invalid")
    return texts, embeddings, model, dataset_sha256


def _load_and_verify_manifests(
    manifest_dir: Path,
) -> tuple[dict[str, list[PairExample]], dict[str, Any], str]:
    metadata_path = manifest_dir / "metadata.json"
    if not metadata_path.is_file():
        raise ReplayError(f"Manifest metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    examples: dict[str, list[PairExample]] = {}
    all_ids: set[str] = set()

    recorded_splits = metadata.get("splits")
    if not isinstance(recorded_splits, dict):
        raise ReplayError("Manifest metadata is missing split records")

    for split in _REQUIRED_MANIFEST_SPLITS:
        path = manifest_dir / f"{split}.jsonl"
        if not path.is_file():
            raise ReplayError(f"Manifest split does not exist: {path}")
        recorded = recorded_splits.get(split)
        if not isinstance(recorded, dict):
            raise ReplayError(f"Manifest metadata is missing split {split}")
        recorded_hash = str(recorded.get("sha256", ""))
        actual_hash = _sha256(path)
        if recorded_hash != actual_hash:
            raise ReplayError(f"Manifest hash changed for {split}")

        rows = list(_read_pair_rows(path, split))
        if len(rows) != int(recorded.get("rows", -1)):
            raise ReplayError(f"Manifest row count changed for {split}")
        labels = {row.label for row in rows}
        if labels != {0, 1}:
            raise ReplayError(f"Manifest split {split} must contain both binary labels")
        duplicate = all_ids.intersection(row.example_id for row in rows)
        if duplicate:
            raise ReplayError(f"Manifest example IDs cross splits: {sorted(duplicate)[:3]}")
        all_ids.update(row.example_id for row in rows)
        examples[split] = rows

    manifest_hash = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return examples, metadata, manifest_hash


def _read_pair_rows(path: Path, split: str) -> Iterable[PairExample]:
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = PairExample(**json.loads(line))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ReplayError(f"Invalid row in {path}:{line_number}") from exc
            if row.split != split:
                raise ReplayError(f"Row split mismatch in {path}:{line_number}")
            if row.label not in {0, 1}:
                raise ReplayError(f"Non-binary label in {path}:{line_number}")
            if not row.left.strip() or not row.right.strip():
                raise ReplayError(f"Empty pair text in {path}:{line_number}")
            if row.example_id in seen:
                raise ReplayError(f"Duplicate example ID in {path}: {row.example_id}")
            seen.add(row.example_id)
            yield row


def _pair_arrays(
    rows: list[PairExample],
    embedding_by_text: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = _normalize_rows(np.vstack([embedding_by_text[row.left] for row in rows]))
    right = _normalize_rows(np.vstack([embedding_by_text[row.right] for row in rows]))
    labels = np.asarray([row.label for row in rows], dtype=np.int64)
    return left, right, labels


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ReplayError("Expected a two-dimensional embedding matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(matrix).all():
        raise ReplayError("Expected finite non-zero embedding rows")
    return matrix / norms


def _build_features(left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
    cosine = np.sum(left * right, axis=1, keepdims=True, dtype=np.float32)
    absolute_difference = np.abs(left - right).astype(np.float32, copy=False)
    product = (left * right).astype(np.float32, copy=False)
    residual = (right - cosine * left).astype(np.float32, copy=False)
    return {
        "Cosine only": cosine,
        "Absolute difference": absolute_difference,
        "Elementwise product": product,
        "Residual only": residual,
        "Full pair": np.hstack([absolute_difference, product]).astype(np.float32, copy=False),
    }


def _evaluate_feature_sets(
    *,
    train_features: dict[str, np.ndarray],
    validation_features: dict[str, np.ndarray],
    test_features: dict[str, np.ndarray],
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> list[PairScore]:
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        roc_auc_score,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def build_model(alpha: float):
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                max_iter=2_000,
                tol=1e-4,
                random_state=seed,
                average=True,
            ),
        )

    def score_model(
        *,
        name: str,
        alpha: float,
        validation_score: float | None,
        model: Any,
        test_x: np.ndarray,
    ) -> PairScore:
        probabilities = model.predict_proba(test_x)[:, 1]
        predictions = (probabilities >= 0.5).astype(np.int64)
        return PairScore(
            name=name,
            selected_alpha=float(alpha),
            validation_balanced_accuracy=validation_score,
            accuracy=float(accuracy_score(test_labels, predictions)),
            balanced_accuracy=float(balanced_accuracy_score(test_labels, predictions)),
            f1=float(f1_score(test_labels, predictions)),
            roc_auc=float(roc_auc_score(test_labels, probabilities)),
            log_loss=float(log_loss(test_labels, probabilities, labels=[0, 1])),
        )

    results: list[PairScore] = []
    for name in FEATURE_NAMES:
        candidates: list[tuple[float, float]] = []
        for alpha in _ALPHA_CANDIDATES:
            model = build_model(alpha)
            model.fit(train_features[name], train_labels)
            predictions = model.predict(validation_features[name])
            score = float(balanced_accuracy_score(validation_labels, predictions))
            candidates.append((score, alpha))
        validation_score, selected_alpha = max(
            candidates, key=lambda item: (item[0], -item[1])
        )
        model = build_model(selected_alpha)
        model.fit(
            np.vstack([train_features[name], validation_features[name]]),
            np.concatenate([train_labels, validation_labels]),
        )
        results.append(
            score_model(
                name=name,
                alpha=selected_alpha,
                validation_score=validation_score,
                model=model,
                test_x=test_features[name],
            )
        )

    full_result = next(score for score in results if score.name == "Full pair")
    shuffled_labels = np.concatenate([train_labels, validation_labels]).copy()
    np.random.default_rng(seed).shuffle(shuffled_labels)
    shuffled_model = build_model(full_result.selected_alpha)
    shuffled_model.fit(
        np.vstack([train_features["Full pair"], validation_features["Full pair"]]),
        shuffled_labels,
    )
    results.append(
        score_model(
            name="Shuffled labels",
            alpha=full_result.selected_alpha,
            validation_score=None,
            model=shuffled_model,
            test_x=test_features["Full pair"],
        )
    )
    return results


def _small_array_value(value: np.ndarray) -> Any:
    if value.shape == ():
        return _json_scalar(value.item())
    return [_json_scalar(item) for item in value.reshape(-1).tolist()]


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
