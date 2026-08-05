from __future__ import annotations

from pathlib import Path

import numpy as np

from relate.inventory import inspect_npz, inventory_assets


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
