"""Neighborhood benchmark: recovery is not fidelity.

    python benchmarks/neighborhoods/run.py          # write expected/
    python benchmarks/neighborhoods/run.py --check  # verify byte-identical

Three conditions over the paired spaces from ``paired_spaces.py``:

- identity (X -> X): everything agrees -- the control;
- translated (X -> Y): shared skeleton, divergent fine signal;
- noise (X -> random): everything collapses -- the floor.

The translated condition must show the signature split in ONE report:
counterpart recovery near-perfect, CKA high, neighborhoods respectable,
yet the X-fitted relation readout loses fine ordering (3A evaluator
reused via ``evaluate_hard_negatives`` + ``compare_reports``). The final
``SpaceComparisonReport`` carries all four as evidence -- it never decides
compatibility.
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
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from paired_spaces import build_paired_spaces  # noqa: E402
from relate.evaluation import (  # noqa: E402
    SpaceComparisonReport,
    compare_geometry,
    compare_neighborhoods,
    compare_reports,
    cosine_scorer,
    counterpart_recovery,
    evaluate_hard_negatives,
    relation_scorer,
    scorer_id_of,
)
from relate.spaces import SpaceIdentity  # noqa: E402

BENCHMARK_ID = "neighborhoods-v1"
K = 10
TIE_TOL = 1e-9


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    paired = build_paired_spaces()
    vectors_x = {name: paired["x"][i] for i, name in enumerate(paired["ids"])}
    vectors_y = {name: paired["y"][i] for i, name in enumerate(paired["ids"])}
    rng = np.random.default_rng(99)
    noise = rng.normal(size=paired["x"].shape)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)

    projection = paired["projection"]
    cases = paired["cases"]
    scorers = {"relation": relation_scorer(projection), "cosine": cosine_scorer()}

    native = {
        name: evaluate_hard_negatives(cases, vectors_x, scorer, tie_tol=TIE_TOL)
        for name, scorer in scorers.items()
    }
    translated = {
        name: evaluate_hard_negatives(cases, vectors_y, scorer, tie_tol=TIE_TOL)
        for name, scorer in scorers.items()
    }
    deltas = {
        name: compare_reports(native[name], translated[name]) for name in scorers
    }

    conditions = {"identity": paired["x"], "translated": paired["y"], "noise": noise}
    neighborhood = {
        name: compare_neighborhoods(paired["x"], target, paired["ids"], k=K)
        for name, target in conditions.items()
    }
    counterpart = {
        name: counterpart_recovery(paired["x"], target, paired["ids"])
        for name, target in conditions.items()
    }
    geometry = compare_geometry(
        paired["x"],
        paired["y"],
        paired["ids"],
        neighborhood_k=K,
        reference_space_hash="x",
        candidate_space_hash="y",
    )

    space_x = SpaceIdentity(
        model="synthetic-paired-x-v1",
        dimensions=paired["x"].shape[1],
        revision=f"seed={paired['seed']}",
        dtype="float64",
        normalize=True,
    )
    space_y = SpaceIdentity(
        model="synthetic-paired-y-v1",
        dimensions=paired["y"].shape[1],
        revision=f"seed={paired['seed']};signal_alpha={paired['signal_alpha']}",
        dtype="float64",
        normalize=True,
    )
    comparison = SpaceComparisonReport(
        source_space_hash=space_x.space_hash,
        target_space_hash=space_y.space_hash,
        geometry=geometry,
        neighborhood=neighborhood["translated"],
        counterpart=counterpart["translated"],
        hard_negatives=deltas["relation"],
    )

    # The signature split, asserted so regressions fail loudly.
    translated_row = {
        "cka": geometry.cka,
        "counterpart_top1": counterpart["translated"].top1,
        "neighborhood_overlap": neighborhood["translated"].mean_overlap,
        "relation_native": native["relation"].accuracy,
        "relation_translated": translated["relation"].accuracy,
    }
    if not (
        translated_row["cka"] > 0.85
        and translated_row["counterpart_top1"] > 0.99
        and translated_row["neighborhood_overlap"] > 0.5
        and native["relation"].accuracy - translated["relation"].accuracy > 0.1
    ):
        print(f"SIGNATURE SPLIT NOT REPRODUCED: {translated_row}")
        return 1

    summary = {
        name: {
            "neighborhood_overlap": neighborhood[name].mean_overlap,
            "top1_agreement": neighborhood[name].top1_agreement,
            "counterpart_top1": counterpart[name].top1,
            "counterpart_mrr": counterpart[name].mrr,
        }
        for name in conditions
    }
    summary["translated"]["relation_accuracy_delta"] = deltas["relation"].accuracy_delta
    summary["translated"]["cka"] = geometry.cka
    summary["translated"]["cosine_matrix_correlation"] = (
        geometry.cosine_matrix_correlation
    )

    try:
        code_version = importlib.metadata.version("relate-search")
    except importlib.metadata.PackageNotFoundError:
        code_version = "unknown"
    provenance = {
        "benchmark_id": BENCHMARK_ID,
        "seed": paired["seed"],
        "signal_alpha": paired["signal_alpha"],
        "k": K,
        "tie_tol": TIE_TOL,
        "source_space": space_x.to_dict(),
        "target_space": space_y.to_dict(),
        "scorers": {name: scorer_id_of(s) for name, s in scorers.items()},
        "code_version": f"relate-search {code_version}",
    }

    artifacts = {
        "summary.json": summary,
        "neighborhood-xy.json": asdict(neighborhood["translated"]),
        "counterpart-xy.json": asdict(counterpart["translated"]),
        "geometry-xy.json": asdict(geometry),
        "hardneg-delta-xy.json": asdict(deltas["relation"]),
        "space-comparison-xy.json": asdict(comparison),
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
    for name, row in summary.items():
        print(
            f"  {name:10s} NN={row['neighborhood_overlap']:.3f} "
            f"cpTop1={row['counterpart_top1']:.3f} mrr={row['counterpart_mrr']:.3f}"
        )
    print(f"wrote {len(artifacts)} artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
