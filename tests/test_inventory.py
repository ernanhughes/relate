from pathlib import Path

import numpy as np

from relate.inventory import inspect_npy, inventory_assets


def test_inspect_npy_reports_header(tmp_path: Path) -> None:
    path = tmp_path / "matrix.npy"
    np.save(path, np.zeros((7, 5), dtype=np.float32))
    report = inspect_npy(path)
    assert report["shape"] == [7, 5]
    assert report["dtype"] == "float32"


def test_inventory_includes_npy_and_npz(tmp_path: Path) -> None:
    npy = tmp_path / "queries.npy"
    npz = tmp_path / "bundle.npz"
    np.save(npy, np.zeros((3, 4), dtype=np.float32))
    np.savez_compressed(npz, values=np.ones((2, 2), dtype=np.float64))
    report = inventory_assets(tmp_path)
    assert report["npy_count"] == 1
    assert report["npz_count"] == 1
    assert report["npy"][0]["path"] == str(npy)
    assert report["npz"][0]["path"] == str(npz)
