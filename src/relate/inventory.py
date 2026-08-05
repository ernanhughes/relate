"""Read-only inventory of preserved RELATE SQLite, NPZ, and NPY assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from relate.replay import ReplayError, inspect_embedding_cache, inspect_npz


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


if __name__ == "__main__":
    raise SystemExit(main())
