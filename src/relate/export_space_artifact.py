"""Export the verified historical Option B projection for a Hugging Face demo.

The exporter reconstructs the frozen training and test embedding matrices from
one read-only SQLite cache, refits the three published ridge readouts, verifies
their coefficient/intercept hashes and the frozen published prediction artifact,
then verifies that the compact runtime projection reproduces those predictions
either bit-for-bit or within an explicit strict numerical-equivalence contract.
It then writes one small pickle-free projection archive for the Space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from relate.model import RelationProjection
from relate.option_b_replay import array_sha256, file_sha256, load_cache_matrix
from relate.python import PYTHON_RELATION_NAMES

MODEL_ID = "microsoft/codebert-base"
MODEL_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"
MAX_LENGTH = 256
POOLING_POLICY = "attention-mask mean pooling"

# The historical prediction artifact is preserved by exact file/array hashes.
# Runtime predictions may follow a different BLAS/vectorization path, so their
# scientific equivalence is verified separately under this strict contract.
PREDICTION_RTOL = 1e-12
PREDICTION_ATOL = 1e-12


class SpaceArtifactError(ValueError):
    """Raised when a frozen input cannot produce the published projection."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SpaceArtifactError(f"required JSON artifact does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpaceArtifactError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise SpaceArtifactError(f"JSON artifact must contain an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SpaceArtifactError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise SpaceArtifactError(f"cannot read JSONL artifact: {path}") from exc
    return rows


def _verify_text_sha256(path: Path, expected: str) -> None:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected:
        return
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != expected:
        raise SpaceArtifactError(f"text artifact hash mismatch: {path}")


def _parameter_sha256(value: np.ndarray | float) -> str:
    return array_sha256(np.asarray(value, dtype=np.float64))

def _prediction_diff_summary(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    stable_keys: tuple[str, ...],
    relation_names: tuple[str, ...],
    max_examples: int = 10,
) -> str:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)

    lines = [
        f"refitted shape:  {actual.shape}",
        f"published shape: {expected.shape}",
        f"refitted array sha256:  {array_sha256(actual)}",
        f"published array sha256: {array_sha256(expected)}",
    ]

    if actual.shape != expected.shape:
        lines.append("cannot compute elementwise diff because shapes differ")
        return "\n".join(lines)

    mismatch = actual != expected
    mismatch_count = int(np.count_nonzero(mismatch))
    total = int(actual.size)

    abs_diff = np.abs(actual - expected)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = abs_diff / np.maximum(np.abs(expected), np.finfo(np.float64).tiny)

    lines.extend(
        [
            f"exact mismatches: {mismatch_count}/{total}",
            f"allclose rtol={PREDICTION_RTOL:.0e} "
            f"atol={PREDICTION_ATOL:.0e}: "
            f"{np.allclose(actual, expected, rtol=PREDICTION_RTOL, atol=PREDICTION_ATOL)}",
            f"allclose rtol=1e-10 atol=1e-12: "
            f"{np.allclose(actual, expected, rtol=1e-10, atol=1e-12)}",
            f"max absolute diff: {float(np.max(abs_diff)):.17g}",
            f"max relative diff: {float(np.max(rel_diff)):.17g}",
        ]
    )

    lines.append("per relation:")
    for column, relation_name in enumerate(relation_names):
        relation_mismatch = mismatch[:, column]
        count = int(np.count_nonzero(relation_mismatch))
        relation_abs = abs_diff[:, column]

        lines.append(
            f"  {relation_name}: mismatches={count}/{actual.shape[0]}, "
            f"max_abs_diff={float(np.max(relation_abs)):.17g}"
        )

    if mismatch_count:
        lines.append(f"largest {min(max_examples, mismatch_count)} mismatches:")

        mismatch_flat = np.flatnonzero(mismatch.ravel())
        order = mismatch_flat[
            np.argsort(abs_diff.ravel()[mismatch_flat])[::-1]
        ]

        for flat_index in order[:max_examples]:
            row, column = np.unravel_index(flat_index, actual.shape)

            key = (
                stable_keys[row]
                if row < len(stable_keys)
                else f"<row {row}>"
            )
            relation = (
                relation_names[column]
                if column < len(relation_names)
                else f"<column {column}>"
            )

            lines.append(
                "  "
                f"row={row} key={key} relation={relation} "
                f"refitted={actual[row, column]:.17g} "
                f"published={expected[row, column]:.17g} "
                f"abs_diff={abs_diff[row, column]:.17g} "
                f"rel_diff={rel_diff[row, column]:.17g}"
            )

    return "\n".join(lines)

def _selection_rows(
    canonical_root: Path,
    split: str,
    *,
    include_primitives: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray | None]:
    selection_dir = canonical_root / "selection"
    report = _load_json(selection_dir / "option-b-canonical-row-selection-v2.json")
    if report.get("status") != "CANONICAL_ROW_SELECTION_V2_VERIFIED":
        raise SpaceArtifactError("canonical Option B row selection is not verified")

    expected = report["artifacts"][split]
    selected_path = selection_dir / f"option-b-selected-{split}-v2.jsonl"
    _verify_text_sha256(
        selected_path,
        str(expected["selected_manifest"]["sha256"]),
    )
    selected = _load_jsonl(selected_path)
    expected_rows = int(expected["selected_manifest"]["rows"])
    if len(selected) != expected_rows:
        raise SpaceArtifactError(f"{split} selected-manifest row count mismatch")

    stable_keys = tuple(str(row["stable_key"]) for row in selected)
    source_hashes = tuple(str(row["code_sha256"]) for row in selected)
    if len(set(stable_keys)) != len(stable_keys):
        raise SpaceArtifactError(f"{split} stable keys are not unique")
    if any(len(value) != 64 for value in source_hashes):
        raise SpaceArtifactError(f"{split} source hashes are invalid")

    if not include_primitives:
        return stable_keys, source_hashes, None

    primitive_path = selection_dir / f"option-b-primitives-{split}-v2.jsonl"
    _verify_text_sha256(
        primitive_path,
        str(expected["primitive_table"]["sha256"]),
    )
    primitive_rows = _load_jsonl(primitive_path)
    if len(primitive_rows) != expected_rows:
        raise SpaceArtifactError(f"{split} primitive-table row count mismatch")
    primitive_keys = tuple(str(row["stable_key"]) for row in primitive_rows)
    if primitive_keys != stable_keys:
        raise SpaceArtifactError(f"{split} primitive order differs from selection")
    primitives = np.asarray(
        [
            [float(row[name]) for name in PYTHON_RELATION_NAMES]
            for row in primitive_rows
        ],
        dtype=np.float64,
    )
    if not np.isfinite(primitives).all():
        raise SpaceArtifactError(f"{split} primitives contain non-finite values")
    return stable_keys, source_hashes, primitives


def _embedding_expectation(canonical_root: Path, split: str) -> dict[str, Any]:
    checkpoint = _load_json(
        canonical_root
        / "embedding-reproduction-v2"
        / "option-b-independent-embedding-reproduction-v2.json"
    )
    if checkpoint.get("status") != "CANONICAL_EMBEDDINGS_V2_REPRODUCED":
        raise SpaceArtifactError("canonical Option B embeddings are not reproduced")
    return dict(checkpoint["splits"][split])


def _fit_verified_projection(
    train_embeddings: np.ndarray,
    train_primitives: np.ndarray,
    probe_bundle: dict[str, Any],
) -> RelationProjection:
    median = np.median(train_primitives, axis=0)
    q25, q75 = np.percentile(train_primitives, (25.0, 75.0), axis=0)
    scale = np.maximum(q75 - q25, 1.0)
    scaled = (train_primitives - median) / scale

    contract = probe_bundle.get("contract", {})
    primitive_contracts = contract.get("primitives", {})
    coefficients = np.empty(
        (train_embeddings.shape[1], len(PYTHON_RELATION_NAMES)),
        dtype=np.float64,
    )
    intercept = np.empty(len(PYTHON_RELATION_NAMES), dtype=np.float64)
    selected_alphas: list[float] = []

    train_x = np.asarray(train_embeddings, dtype=np.float64)
    for index, relation_name in enumerate(PYTHON_RELATION_NAMES):
        frozen = primitive_contracts.get(relation_name)
        if not isinstance(frozen, dict):
            raise SpaceArtifactError(f"probe bundle is missing {relation_name}")
        alpha = float(frozen["selected_alpha"])
        model = Ridge(alpha=alpha).fit(train_x, scaled[:, index])
        coefficient = np.asarray(model.coef_, dtype=np.float64)
        model_intercept = float(model.intercept_)
        if _parameter_sha256(coefficient) != str(frozen["final_coefficient_sha256"]):
            raise SpaceArtifactError(
                f"{relation_name} coefficient hash differs from the frozen probe bundle"
            )
        if _parameter_sha256(model_intercept) != str(frozen["final_intercept_sha256"]):
            raise SpaceArtifactError(
                f"{relation_name} intercept hash differs from the frozen probe bundle"
            )
        coefficients[:, index] = coefficient
        intercept[index] = model_intercept
        selected_alphas.append(alpha)

    if len(set(selected_alphas)) != 1:
        raise SpaceArtifactError(
            "the current RelationProjection archive requires one shared alpha"
        )

    return RelationProjection(
        coefficients=coefficients,
        intercept=intercept,
        relation_median=np.asarray(median, dtype=np.float64),
        relation_scale=np.asarray(scale, dtype=np.float64),
        embedding_mean=np.asarray(train_x.mean(axis=0), dtype=np.float64),
        alpha=selected_alphas[0],
        relation_names=tuple(PYTHON_RELATION_NAMES),
    )


def export_space_artifact(
    *,
    canonical_root: Path,
    cache_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create and verify the tiny projection archive consumed by the Space."""
    train_keys, train_hashes, train_primitives = _selection_rows(
        canonical_root,
        "train",
        include_primitives=True,
    )
    assert train_primitives is not None
    train_expected = _embedding_expectation(canonical_root, "train")
    train_embeddings = load_cache_matrix(
        cache_path,
        stable_keys=train_keys,
        source_hashes=train_hashes,
        fingerprint_sha256=str(train_expected["extraction_fingerprint_sha256"]),
        dimensions=int(train_expected["dimensions"]),
        expected_array_sha256=str(train_expected["array_sha256"]),
    )

    probe_dir = canonical_root / "probes-v1"
    probe_bundle_path = probe_dir / "option-b-primitive-probe-bundle-v1.json"
    probe_bundle = _load_json(probe_bundle_path)
    if probe_bundle.get("status") != "PRIMITIVE_PROBE_FIT_COMPLETE_PENDING_PUBLICATION_REVIEW":
        raise SpaceArtifactError("canonical primitive-probe bundle is incomplete")
    projection = _fit_verified_projection(
        train_embeddings,
        train_primitives,
        probe_bundle,
    )

    test_keys, test_hashes, _ = _selection_rows(
        canonical_root,
        "test",
        include_primitives=False,
    )
    test_expected = _embedding_expectation(canonical_root, "test")
    test_embeddings = load_cache_matrix(
        cache_path,
        stable_keys=test_keys,
        source_hashes=test_hashes,
        fingerprint_sha256=str(test_expected["extraction_fingerprint_sha256"]),
        dimensions=int(test_expected["dimensions"]),
        expected_array_sha256=str(test_expected["array_sha256"]),
    )
    predictions = np.asarray(projection.project(test_embeddings), dtype=np.float64)

    publication = _load_json(probe_dir / "option-b-primitive-probe-publication-v1.json")
    expected_predictions = publication["predictions"]["test_queries"]
    prediction_path = probe_dir / "option-b-predicted-test-queries-v1.npy"
    if file_sha256(prediction_path) != str(expected_predictions["file_sha256"]):
        raise SpaceArtifactError("published test prediction file hash mismatch")
    published_predictions = np.load(prediction_path, allow_pickle=False)

    # First verify the historical publication artifact itself exactly. This is
    # the archival contract: the checked-in .npy must still be the frozen array
    # named by the publication metadata.
    published_prediction_hash = array_sha256(published_predictions)
    expected_published_prediction_hash = str(expected_predictions["array_sha256"])
    if published_prediction_hash != expected_published_prediction_hash:
        raise SpaceArtifactError(
            "loaded published prediction array does not match its frozen array hash"
        )

    # Then compare the compact RelationProjection runtime path with that frozen
    # artifact. Exact equality is recorded when available, but a different
    # BLAS/vectorization accumulation order may legitimately change low-order
    # float64 bits without changing the fitted model or scientific result.
    predictions_exact = np.array_equal(predictions, published_predictions)
    predictions_numerically_equivalent = np.allclose(
        predictions,
        published_predictions,
        rtol=PREDICTION_RTOL,
        atol=PREDICTION_ATOL,
    )

    prediction_abs_diff = np.abs(predictions - published_predictions)
    prediction_max_abs_diff = float(np.max(prediction_abs_diff))
    prediction_mismatch_count = int(
        np.count_nonzero(predictions != published_predictions)
    )

    if not predictions_numerically_equivalent:
        details = _prediction_diff_summary(
            predictions,
            published_predictions,
            stable_keys=test_keys,
            relation_names=projection.relation_names,
        )
        raise SpaceArtifactError(
            "refitted projection materially differs from the published "
            f"test predictions\n\n{details}"
        )

    runtime_prediction_hash = array_sha256(predictions)
    prediction_reproduction = (
        "BIT_EXACT"
        if predictions_exact
        else "NUMERICALLY_EQUIVALENT_NON_BIT_EXACT"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    projection.save(output_path)
    sidecar_path = output_path.with_suffix(".json")
    result = {
        "artifact_id": "relate-option-b-space-projection-v1",
        "status": "SPACE_PROJECTION_EXPORTED_AND_VERIFIED",
        "verification_mode": prediction_reproduction,
        "scientific_scope": "historical_option_b_live_readout",
        "model": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "max_length": MAX_LENGTH,
            "pooling": POOLING_POLICY,
        },
        "projection": {
            "path": str(output_path),
            "file_sha256": file_sha256(output_path),
            "embedding_dimensions": projection.embedding_dimensions,
            "relation_names": list(projection.relation_names),
            "selected_alpha": projection.alpha,
            "coefficient_sha256": array_sha256(projection.coefficients),
            "intercept_sha256": array_sha256(projection.intercept),
            "relation_median_sha256": array_sha256(projection.relation_median),
            "relation_scale_sha256": array_sha256(projection.relation_scale),
        },
        "verification": {
            "train_embedding_array_sha256": str(train_expected["array_sha256"]),
            "test_embedding_array_sha256": str(test_expected["array_sha256"]),
            "probe_bundle_file_sha256": file_sha256(probe_bundle_path),

            # Historical publication artifact: exact archival identity.
            "published_test_prediction_file_sha256": file_sha256(prediction_path),
            "published_test_prediction_array_sha256": published_prediction_hash,
            "published_prediction_artifact_exact": True,

            # Current compact runtime path: distinguish bit identity from strict
            # numerical equivalence instead of conflating the two.
            "runtime_test_prediction_array_sha256": runtime_prediction_hash,
            "runtime_predictions_bit_exact": predictions_exact,
            "runtime_predictions_numerically_equivalent": (
                predictions_numerically_equivalent
            ),
            "runtime_prediction_reproduction": prediction_reproduction,
            "runtime_prediction_rtol": PREDICTION_RTOL,
            "runtime_prediction_atol": PREDICTION_ATOL,
            "runtime_prediction_exact_mismatch_count": prediction_mismatch_count,
            "runtime_prediction_max_abs_diff": prediction_max_abs_diff,
        },
        "relate_e01_affected": False,
    }
    sidecar_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-root",
        type=Path,
        required=True,
        help="Path to artifacts/canonical/option-b in similarity_is_relative.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(".writer/option-b/cache/gpu-batch10-a.sqlite3"),
        help="One verified Option B embeddings_v2 SQLite cache.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("space/assets/option-b-demo-projection.npz"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_space_artifact(
        canonical_root=args.canonical_root,
        cache_path=args.cache,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
