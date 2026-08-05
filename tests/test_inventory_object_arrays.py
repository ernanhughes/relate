from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from relate.inventory import inspect_npy, inspect_npz, inventory_assets


def test_inspect_npz_skips_object_arrays_without_pickle(tmp_path: Path) -> None:
    archive = tmp_path / "legacy.npz"
    np.savez_compressed(
        archive,
        metadata=np.asarray({"old": "payload"}, dtype=object),
        model=np.asarray("demo-model"),
        embeddings=np.eye(2),
    )

    report = inspect_npz(archive)

    assert report["arrays"]["metadata"]["dtype"] == "object"
    assert report["arrays"]["metadata"]["contains_objects"] is True
    assert report["small_values"]["model"] == "demo-model"
    assert report["skipped_values"] == [
        {
            "key": "metadata",
            "reason": "object_dtype_requires_pickle",
        }
    ]


def test_inventory_continues_past_legacy_object_archive(tmp_path: Path) -> None:
    archive = tmp_path / "legacy.npz"
    np.savez_compressed(
        archive,
        labels=np.asarray([{"relation": "same"}], dtype=object),
    )

    report = inventory_assets(tmp_path)

    assert report["npz_count"] == 1
    assert report["npz"][0]["path"] == str(archive)
    assert report["npz"][0]["skipped_values"][0]["key"] == "labels"


def test_raw_f16_header_does_not_require_platform_dtype_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member = _npy_v1_bytes("<f16", shape=(1,), data=b"\x00" * 16)
    archive = tmp_path / "legacy-f16.npz"
    standalone = tmp_path / "legacy-f16.npy"
    standalone.write_bytes(member)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("values.npy", member)

    real_dtype = np.dtype

    def reject_f16(value, *args, **kwargs):
        if value == "<f16":
            raise TypeError("data type '<f16' not understood")
        return real_dtype(value, *args, **kwargs)

    monkeypatch.setattr(np, "dtype", reject_f16)

    npy_report = inspect_npy(standalone)
    npz_report = inspect_npz(archive)

    assert npy_report["dtype"] == "<f16"
    assert npy_report["dtype_descriptor"] == "<f16"
    assert npy_report["dtype_supported"] is False
    assert "not understood" in npy_report["dtype_error"]
    assert npz_report["arrays"]["values"]["dtype"] == "<f16"
    assert npz_report["arrays"]["values"]["dtype_supported"] is False
    assert npz_report["skipped_values"] == [
        {
            "key": "values",
            "reason": "dtype_not_supported_by_local_numpy",
        }
    ]


def test_inventory_records_bad_archive_and_continues(tmp_path: Path) -> None:
    broken = tmp_path / "broken.npz"
    valid = tmp_path / "valid.npz"
    broken.write_bytes(b"not a zip archive")
    np.savez_compressed(valid, values=np.asarray([1, 2, 3]))

    report = inventory_assets(tmp_path)

    assert report["npz_count"] == 2
    assert report["npz_error_count"] == 1
    by_path = {item["path"]: item for item in report["npz"]}
    assert "inspection_error" in by_path[str(broken)]
    assert by_path[str(valid)]["small_values"]["values"] == [1, 2, 3]


def _npy_v1_bytes(descriptor: str, *, shape: tuple[int, ...], data: bytes) -> bytes:
    header_literal = repr(
        {
            "descr": descriptor,
            "fortran_order": False,
            "shape": shape,
        }
    ).encode("latin1")
    prefix_size = len(b"\x93NUMPY") + 2 + 2
    padding = (-(prefix_size + len(header_literal) + 1)) % 64
    header = header_literal + (b" " * padding) + b"\n"
    return (
        b"\x93NUMPY"
        + bytes((1, 0))
        + len(header).to_bytes(2, byteorder="little", signed=False)
        + header
        + data
    )
