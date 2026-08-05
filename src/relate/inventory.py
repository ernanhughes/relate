"""Read-only inventory of preserved RELATE SQLite, NPZ, and NPY assets."""

from __future__ import annotations

import argparse
import ast
import json
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Callable

import numpy as np

from relate.replay import ReplayError, inspect_embedding_cache


_REQUIRED_SNAPSHOT_KEYS = {"texts", "embeddings", "model", "dataset_sha256"}
_MAGIC_PREFIX = b"\x93NUMPY"
_MAX_HEADER_BYTES = 10_000_000


def inspect_npy(path: str | Path) -> dict[str, Any]:
    """Read one NPY header without loading or coercing the stored array dtype."""

    array_path = Path(path)
    if not array_path.is_file():
        raise ReplayError(f"NPY file does not exist: {array_path}")
    with array_path.open("rb") as stream:
        metadata = _read_npy_header(stream, source=array_path)
    return {
        "path": str(array_path),
        "size_bytes": array_path.stat().st_size,
        **metadata,
    }


def inspect_npz(path: str | Path) -> dict[str, Any]:
    """Inspect NPZ members without unpickling or requiring local dtype support.

    Array metadata is parsed from each embedded NPY header as a Python literal.
    This avoids NumPy coercing descriptors such as ``<f16`` on platforms where
    that floating-point dtype is unavailable. Small safe arrays are loaded only
    when the local NumPy build supports their dtype.
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
                arrays[key] = _read_npy_header(
                    stream,
                    source=f"{archive}!{member}",
                )

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
            if not metadata["dtype_supported"]:
                skipped_values.append(
                    {
                        "key": key,
                        "reason": "dtype_not_supported_by_local_numpy",
                    }
                )
                continue
            try:
                scalar_values[key] = _small_array_value(payload[key])
            except (TypeError, ValueError) as exc:
                skipped_values.append(
                    {
                        "key": key,
                        "reason": f"array_value_unavailable: {exc}",
                    }
                )

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
    npz_reports = [_inspect_or_record_error(inspect_npz, path) for path in npz_files]
    npy_reports = [_inspect_or_record_error(inspect_npy, path) for path in npy_files]
    return {
        "root": str(root_path),
        "sqlite": inspect_embedding_cache(database) if database.is_file() else None,
        "npz": npz_reports,
        "npz_count": len(npz_files),
        "npz_error_count": sum("inspection_error" in item for item in npz_reports),
        "npy": npy_reports,
        "npy_count": len(npy_files),
        "npy_error_count": sum("inspection_error" in item for item in npy_reports),
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


def _read_npy_header(stream: BinaryIO, *, source: str | Path) -> dict[str, Any]:
    """Parse an NPY header without constructing its dtype."""

    magic = stream.read(len(_MAGIC_PREFIX))
    if magic != _MAGIC_PREFIX:
        raise ReplayError(f"Invalid NPY magic in {source}")
    version_bytes = stream.read(2)
    if len(version_bytes) != 2:
        raise ReplayError(f"Truncated NPY version in {source}")
    version = (int(version_bytes[0]), int(version_bytes[1]))
    if version == (1, 0):
        length_size = 2
        encoding = "latin1"
    elif version == (2, 0):
        length_size = 4
        encoding = "latin1"
    elif version == (3, 0):
        length_size = 4
        encoding = "utf-8"
    else:
        raise ReplayError(f"Unsupported NPY version {version} in {source}")

    length_bytes = stream.read(length_size)
    if len(length_bytes) != length_size:
        raise ReplayError(f"Truncated NPY header length in {source}")
    header_length = int.from_bytes(length_bytes, byteorder="little", signed=False)
    if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
        raise ReplayError(f"Invalid NPY header length {header_length} in {source}")
    raw_header = stream.read(header_length)
    if len(raw_header) != header_length:
        raise ReplayError(f"Truncated NPY header in {source}")

    try:
        header = ast.literal_eval(raw_header.decode(encoding).strip())
    except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
        raise ReplayError(f"Invalid NPY header literal in {source}") from exc
    if not isinstance(header, dict):
        raise ReplayError(f"NPY header is not a dictionary in {source}")

    shape = header.get("shape")
    fortran_order = header.get("fortran_order")
    descriptor = header.get("descr")
    if (
        not isinstance(shape, tuple)
        or any(not isinstance(value, int) or value < 0 for value in shape)
    ):
        raise ReplayError(f"Invalid NPY shape in {source}: {shape!r}")
    if not isinstance(fortran_order, bool):
        raise ReplayError(f"Invalid NPY fortran_order in {source}")
    if descriptor is None:
        raise ReplayError(f"NPY header is missing descr in {source}")

    contains_objects = _descriptor_contains_objects(descriptor)
    dtype_supported = True
    dtype_error: str | None = None
    try:
        dtype_display = str(np.dtype(descriptor))
    except (TypeError, ValueError) as exc:
        dtype_supported = False
        dtype_display = descriptor if isinstance(descriptor, str) else repr(descriptor)
        dtype_error = str(exc)

    result: dict[str, Any] = {
        "npy_version": list(version),
        "shape": list(shape),
        "dtype": dtype_display,
        "dtype_descriptor": _json_descriptor(descriptor),
        "dtype_supported": dtype_supported,
        "fortran_order": fortran_order,
        "contains_objects": contains_objects,
    }
    if dtype_error is not None:
        result["dtype_error"] = dtype_error
    return result


def _descriptor_contains_objects(descriptor: Any) -> bool:
    if isinstance(descriptor, str):
        normalized = descriptor[1:] if descriptor[:1] in "<>=|" else descriptor
        return normalized.startswith("O")
    if isinstance(descriptor, dict):
        return any(_descriptor_contains_objects(value) for value in descriptor.values())
    if isinstance(descriptor, list):
        return any(_descriptor_contains_objects(value) for value in descriptor)
    if isinstance(descriptor, tuple):
        # Structured dtype fields are (name, format) or (name, format, shape).
        if len(descriptor) >= 2 and isinstance(descriptor[0], (str, tuple)):
            return _descriptor_contains_objects(descriptor[1])
        return any(_descriptor_contains_objects(value) for value in descriptor)
    return False


def _json_descriptor(descriptor: Any) -> Any:
    if isinstance(descriptor, tuple):
        return [_json_descriptor(value) for value in descriptor]
    if isinstance(descriptor, list):
        return [_json_descriptor(value) for value in descriptor]
    if isinstance(descriptor, dict):
        return {str(key): _json_descriptor(value) for key, value in descriptor.items()}
    if isinstance(descriptor, (str, int, float, bool)) or descriptor is None:
        return descriptor
    return repr(descriptor)


def _inspect_or_record_error(
    inspector: Callable[[Path], dict[str, Any]],
    path: Path,
) -> dict[str, Any]:
    try:
        return inspector(path)
    except (OSError, ReplayError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "inspection_error": f"{type(exc).__name__}: {exc}",
        }


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
