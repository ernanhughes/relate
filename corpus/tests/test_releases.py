"""Tier-2 corpus release tests — frozen-release continuity, not runtime behavior.

Run from the corpus directory (stdlib only, no NumPy, no installed package):

    python -m pytest corpus/tests        # from the repo root
    python corpus/tests/test_releases.py # stdlib fallback runner

Gates (Step 2 acceptance):
- frozen corpus hashes match exactly (0.1.0 / 0.2.0 / doc-0.1.0);
- every file recorded in each manifest exists with identical sha256;
- counts, IDs membership and task views unchanged;
- relate-0.2.0 items/pairs byte-identical to relate-0.1.0 (v0.1 + hard queries).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # corpus/

RELEASES = {
    "relate-0.1.0": {"items": 1173, "pairs": 1181, "queries": 269},
    "relate-0.2.0": {"items": 1173, "pairs": 1181, "queries": 411},
}

DOC_RELEASE = "doc/relate-doc-0.1.0"
DOC_HASH = "111cc2bb9008557d632061f98a0e4f847b91293e31bcdb2d78285a35de7f7914"
HASH_010 = "8cad6816d90e06bc49e5b0b64cd460945e17ea4b1ec3c46054409669eda525b3"
HASH_020 = "695654cbd0bdadcbe0140f85927ad8799a6a3306ab431bc1ec48fdec195f7719"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_dir(name: str) -> Path:
    d = HERE / name
    assert d.is_dir(), f"missing release dir: {d}"
    return d


def _manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text())


def _find(directory: Path, basename: str) -> Path:
    hits = [p for p in directory.rglob(basename) if p.is_file()]
    assert len(hits) == 1, f"expected one {basename} under {directory}, found {len(hits)}"
    return hits[0]


def _check_manifest_hashes(name: str) -> None:
    directory = _release_dir(name)
    manifest = _manifest(directory)
    frozen = (directory / "corpus_hash.txt").read_text().strip()
    assert frozen == manifest["corpus_hash"], f"{name}: corpus_hash.txt != manifest"
    for basename, expected in manifest["files"].items():
        assert _sha256(_find(directory, basename)) == expected, f"{name}: {basename} differs"


def _count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def test_hash_010_matches() -> None:
    assert (_release_dir("relate-0.1.0") / "corpus_hash.txt").read_text().strip() == HASH_010


def test_hash_020_matches() -> None:
    assert (_release_dir("relate-0.2.0") / "corpus_hash.txt").read_text().strip() == HASH_020


def test_hash_doc_matches() -> None:
    assert (_release_dir(DOC_RELEASE) / "corpus_hash.txt").read_text().strip() == DOC_HASH


def test_manifest_hashes_010() -> None:
    _check_manifest_hashes("relate-0.1.0")


def test_manifest_hashes_020() -> None:
    _check_manifest_hashes("relate-0.2.0")


def test_manifest_hashes_doc() -> None:
    _check_manifest_hashes(DOC_RELEASE)


def test_counts_unchanged() -> None:
    for name, expected in RELEASES.items():
        directory = _release_dir(name)
        assert _count_lines(directory / "items.jsonl") == expected["items"], name
        assert _count_lines(directory / "pairs.jsonl") == expected["pairs"], name
        assert _count_lines(directory / "queries.jsonl") == expected["queries"], name
    doc = _release_dir(DOC_RELEASE)
    assert _count_lines(doc / "documents.jsonl") == 45
    assert _count_lines(doc / "compressions.jsonl") == 595
    assert _count_lines(doc / "transformation_pairs.jsonl") == 260


def test_v02_items_pairs_byte_identical_to_v01() -> None:
    for leaf in ("items.jsonl", "pairs.jsonl"):
        a = HERE / "relate-0.1.0" / leaf
        b = HERE / "relate-0.2.0" / leaf
        assert _sha256(a) == _sha256(b), leaf


def test_views_present_both_releases() -> None:
    views = (
        "retrieval-v0.1.json",
        "relation-v0.1.json",
        "hard-negative-v0.1.json",
        "calibration-v0.1.json",
        "dimensionality-v0.1.json",
        "alignment-v0.1.json",
    )
    for name in ("relate-0.1.0", "relate-0.2.0"):
        for view in views:
            assert _find(_release_dir(name), view).is_file()


def run() -> int:
    fns = sorted(
        ((name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")),
    )
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
