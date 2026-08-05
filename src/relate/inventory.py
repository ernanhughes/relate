"""Read-only inventory of preserved RELATE SQLite, NPZ, and NPY assets."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from relate.replay import ReplayError, inspect_embedding_cache


_REQUIRED_SNAPSHOT_KEYS = {"texts", "embeddings", "model", "dataset_sha256"}


def inspect_npy(path: str | Path) -> dict[str, Any]:
    """Read one NPY header without loading the full array into memory."""

    array_path = Path(path)
    if not array_path.is_file():
        raise ReplayError(f"NPY file does not exist: {array_path}")
    with array_path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:  # pragma: no cover
            raise ReplayError(f"Unsupported NPY version {version} in {array_path}")
    return {
        "path": str(array_path),
        "size_bytes": array_path.stat().st_size,
        "shape": list(shape),
        "dtype": str(dtype),
        "fortran_order": bool(fortran_order),
        "contains_objects": bool(dtype.hasobject),
    }


def inspect_npz(path: str | Path) -> dict[str, Any]:
    """Inspect NPZ members safely, including legacy object-dtype arrays.

    Array shape and dtype are read directly from each embedded NPY header. Small
    non-object arrays are loaded with ``allow_pickle=False`` so useful scalar
    metadata remains visible. Object arrays are never materialized or unpickled.
    """

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
                else:  # pragma: no cover
                    raise ReplayError(f"Unsupported NPY version {version} in {archive}")
            arrays[key] = {
                "shape": list(shape),
                "dtype": str(dtype),
                "fortran_order": bool(fortran_order),
                "contains_objects": bool(dtype.hasobject),
            }

    scalar_values: dict[str, Any] = {}
    skipped_values: list[dict[str, str]] = []
    with np.load(archive, allow_pickle=False) as payload:
        for key, metadata in arrays.items():
            shape = tuple(metadata["shape"])
            element_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
            if element_count > 16 or key == "embeddings":
                continue
            if metadata["contains_objects"]:
                skipped_values.append(
                    {
                        "key": key,
                        "reason": "object_dtype_requires_pickle",
                    }
                )
                continue
            try:
                scalar_values[key] = _small_array_value(payload[key])
            except ValueError as exc:
                skipped_values.append({"key": key, "reason": str(exc)})

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
        "skipped_values": skipped_values,
    }


def inventory_assets(
    root: str | Path,
    *,
    cache_db: str | Path | None = None,
) -> dict[str, Any]:
    """Inventory preserved assets without reading full embedding arrays."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise ReplayError(f"Inventory root does not exist: {root_path}")
    database = (
        Path(cache_db)
        if cache_db is not None
        else root_path / ".writer" / "benchmarks" / "embedding-cache.sqlite3"
    )
    npz_files = sorted(root_path.rglob("*.npz"))
    npy_files = sorted(root_path.rglob("*.npy"))
    return {
        "root": str(root_path),
        "sqlite": inspect_embedding_cache(database) if database.is_file() else None,
        "npz": [inspect_npz(path) for path in npz_files],
        "npz_count": len(npz_files),
        "npy": [inspect_npy(path) for path in npy_files],
        "npy_count": len(npy_files),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--cache-db", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = inventory_assets(args.root, cache_db=args.cache_db)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
