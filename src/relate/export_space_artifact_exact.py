"""Export the Option B Space artifact using the exact historical prediction path.

The historical predicted executor fitted and evaluated three independent
single-target Ridge models.  Predicting all three columns with one matrix-matrix
multiplication is mathematically equivalent, but BLAS accumulation order can
change the final floating-point bits.  The publication hashes therefore require
three independent matrix-vector operations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from relate.export_space_artifact import export_space_artifact as _base_export
from relate.model import RelationProjection, RelateError


def historical_independent_project(
    projection: RelationProjection,
    embeddings: Any,
) -> np.ndarray:
    """Project each relation independently, matching ``Ridge.predict`` shape."""

    array = np.asarray(embeddings, dtype=np.float64)
    was_vector = array.ndim == 1
    matrix = array[None, :] if was_vector else array
    if matrix.ndim != 2:
        raise RelateError("embeddings must be a vector or matrix")
    if matrix.shape[1] != projection.embedding_dimensions:
        raise RelateError("embedding dimensions do not match the fitted projection")
    if not np.isfinite(matrix).all():
        raise RelateError("embeddings must contain only finite values")

    predicted = np.empty(
        (matrix.shape[0], projection.relation_dimensions),
        dtype=np.float64,
    )
    for index in range(projection.relation_dimensions):
        predicted[:, index] = (
            matrix @ projection.coefficients[:, index]
            + projection.intercept[index]
        )
    return predicted[0] if was_vector else predicted


def export_space_artifact(
    *,
    canonical_root: Path,
    cache_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the verified exporter with historical independent predictions."""

    original = RelationProjection.project

    def exact_project(self: RelationProjection, embeddings: Any) -> np.ndarray:
        return historical_independent_project(self, embeddings)

    RelationProjection.project = exact_project  # type: ignore[method-assign]
    try:
        result = _base_export(
            canonical_root=canonical_root,
            cache_path=cache_path,
            output_path=output_path,
        )
    finally:
        RelationProjection.project = original  # type: ignore[method-assign]

    result["verification"]["prediction_execution"] = (
        "three independent float64 matrix-vector operations"
    )
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(".writer/option-b/cache/gpu-batch10-a.sqlite3"),
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
