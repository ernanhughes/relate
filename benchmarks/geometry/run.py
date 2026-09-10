"""Geometry benchmark: shape differs across spaces; invariants hold exactly.

    python benchmarks/geometry/run.py          # write expected/
    python benchmarks/geometry/run.py --check  # verify byte-identical

Three synthetic spaces (seeded, n=1200, nominal d=256):

- ISO: isotropic Gaussian, unit norm -- random-pair cosine near zero.
- ANISO: exponentially decaying axis scales -- concentrated like the
  high-baseline encoders in the book (raw magnitude is space-relative).
- LOWRANK: 6-dim latent structure + noise -- nominal 256, effective rank
  and TwoNN ID in the single digits.

Plus the invariant checks: rotation preserves everything, scaling preserves
cosine but moves Euclidean norms, translation moves cosine. No semantic
claim is made about any space; geometry diagnoses geometry.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from relate.evaluation import PairSamplingSpec, describe_geometry  # noqa: E402
from relate.spaces import SpaceIdentity  # noqa: E402

BENCHMARK_ID = "geometry-v1"
SEED = 0
N = 1200
D = 256


def build_spaces(seed: int = SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    iso = rng.normal(size=(N, D))
    iso /= np.linalg.norm(iso, axis=1, keepdims=True)
    scales = 1.0 + 8.0 * np.exp(-np.arange(D) / 6.0)
    base = rng.normal(size=(N, D)) * scales
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    direction = np.zeros(D)
    direction[0] = 1.0
    aniso = base + direction
    aniso /= np.linalg.norm(aniso, axis=1, keepdims=True)
    latent = rng.uniform(-3.0, 3.0, size=(N, 6))
    lowrank = latent @ rng.normal(size=(6, D)) + 0.05 * rng.normal(size=(N, D))
    return {"iso": iso, "aniso": aniso, "lowrank": lowrank}


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    spaces = build_spaces()
    sampling = PairSamplingSpec(max_pairs=100_000, seed=SEED)
    reports = {
        name: describe_geometry(vectors, sampling=sampling)
        for name, vectors in spaces.items()
    }

    rng = np.random.default_rng(1)
    rotation, _ = np.linalg.qr(rng.normal(size=(D, D)))
    rotated = spaces["iso"] @ rotation
    rotated_report = describe_geometry(rotated, sampling=sampling)
    base = reports["iso"]
    invariants = {
        "rotation": {
            "cosine_mean_drift": abs(base.cosine_mean - rotated_report.cosine_mean),
            "cosine_std_drift": abs(base.cosine_std - rotated_report.cosine_std),
            "effective_rank_ratio": rotated_report.effective_rank / base.effective_rank,
        },
        "scaling": {
            "cosine_mean_scaled": describe_geometry(
                2.5 * spaces["iso"], sampling=sampling, with_twonn=False
            ).cosine_mean,
            "cosine_mean_base": base.cosine_mean,
            "norm_ratio": float(
                np.linalg.norm(2.5 * spaces["iso"], axis=1).mean() / base.norm_mean
            ),
        },
        "translation": {
            "cosine_mean_shifted": describe_geometry(
                spaces["aniso"] + 5.0, sampling=sampling, with_twonn=False
            ).cosine_mean,
            "cosine_mean_base": reports["aniso"].cosine_mean,
        },
    }
    if not (
        invariants["rotation"]["cosine_mean_drift"] < 1e-9
        and abs(invariants["scaling"]["cosine_mean_scaled"] - base.cosine_mean) < 1e-9
        and abs(invariants["scaling"]["norm_ratio"] - 2.5) < 1e-9
        and abs(
            invariants["translation"]["cosine_mean_shifted"]
            - reports["aniso"].cosine_mean
        )
        > 0.05
    ):
        print("INVARIANT CHECK FAILED")
        return 1

    space_hashes = {
        name: SpaceIdentity(
            model=f"synthetic-geometry-{name}-v1",
            dimensions=D,
            revision=f"seed={SEED}",
            dtype="float64",
            normalize=True,
        ).space_hash
        for name in spaces
    }
    try:
        code_version = importlib.metadata.version("relate-search")
    except importlib.metadata.PackageNotFoundError:
        code_version = "unknown"
    provenance = {
        "benchmark_id": BENCHMARK_ID,
        "seed": SEED,
        "n": N,
        "dimensions": D,
        "sampling": asdict(sampling),
        "space_hashes": space_hashes,
        "code_version": f"relate-search {code_version}",
    }

    artifacts = {
        **{f"geometry-{n}.json": asdict(r) for n, r in reports.items()},
        "invariants.json": invariants,
        "provenance.json": provenance,
    }
    out_dir = HERE / "expected"
    if args.check:
        failures = sum(
            1
            for leaf, payload in artifacts.items()
            if not (out_dir / leaf).exists()
            or (out_dir / leaf).read_text() != _dump(payload)
        )
        print("MATCH" if not failures else f"{failures} differing artifact(s)")
        return 1 if failures else 0
    out_dir.mkdir(exist_ok=True)
    for leaf, payload in artifacts.items():
        (out_dir / leaf).write_text(_dump(payload))
    for name, report in reports.items():
        twonn = report.intrinsic_dimension.estimate
        print(
            f"  {name:8s} cos_mean={report.cosine_mean:+.3f} "
            f"erank={report.effective_rank:.1f} PR={report.participation_ratio:.1f} "
            f"twonn={twonn:.1f} (nominal {D})"
        )
    print(f"wrote {len(artifacts)} artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
