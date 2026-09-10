"""Bridge-producer benchmark: deliberately boring producers, 4A judgment.

    python benchmarks/bridges/run.py          # write expected/
    python benchmarks/bridges/run.py --check  # verify byte-identical

Six producers (identity, constant centroid, random map, Procrustes,
linear, ridge) fit on train anchors, transform held-out eval rows, and
are judged exclusively through ``compare_native_spaces`` -- the same
4A path as native comparison. Training reconstruction is reported as a
labeled FIT DIAGNOSTIC, never as preservation evidence. Every
candidate set receives a derived space identity that is asserted
different from the native target hash.
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

from bridge_cases import build_bridge_cases  # noqa: E402
from relate.bridges import (  # noqa: E402
    BridgeSpec,
    bridge_output_space,
    constant_centroid_bridge,
    fit_bridge,
    identity_bridge,
    random_map_bridge,
)
from relate.evaluation import (  # noqa: E402
    compare_native_spaces,
    cosine_scorer,
    scorer_id_of,
)
from relate.spaces import SpaceIdentity  # noqa: E402

BENCHMARK_ID = "bridges-v1"
K = 10
TIE_TOL = 1e-9
RANDOM_SEED = 0


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _mean_cosine(first: np.ndarray, second: np.ndarray) -> float:
    first_n = first / np.linalg.norm(first, axis=1, keepdims=True).clip(min=1e-12)
    second_n = second / np.linalg.norm(second, axis=1, keepdims=True).clip(min=1e-12)
    return float(np.mean(np.sum(first_n * second_n, axis=1)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    fixture = build_bridge_cases()
    vectors_x, vectors_y = fixture["x"], fixture["y"]
    train_corr, eval_corr = (
        fixture["train_correspondence"], fixture["eval_correspondence"])
    if train_corr.content_hash == eval_corr.content_hash:
        print("TRAIN AND EVAL CORRESPONDENCES COLLIDE")
        return 1
    cases = fixture["cases"]
    scorer = cosine_scorer()
    eval_rows = np.asarray(fixture["eval_rows"])
    train_rows = np.asarray(
        [r for r in range(vectors_x.shape[0]) if r not in set(fixture["eval_rows"])]
    )

    space_x = SpaceIdentity(model="synthetic-bridges-x-v1", dimensions=vectors_x.shape[1])
    space_y = SpaceIdentity(
        model="synthetic-bridges-y-v1", dimensions=vectors_y.shape[1],
        revision="fine-dims-weakened",
    )

    fitted = {}
    for method in ("procrustes", "linear", "ridge"):
        params = {"alpha": 1.0} if method == "ridge" else {}
        fitted[method] = fit_bridge(
            source_vectors=vectors_x, target_vectors=vectors_y,
            correspondence=train_corr,
            spec=BridgeSpec(
                source_space_hash=space_x.space_hash,
                target_space_hash=space_y.space_hash,
                method=method, params=params),
        )
    producers = {
        **fitted,
        "identity": identity_bridge(
            vectors_x.shape[1], source_space_hash=space_x.space_hash,
            target_space_hash=space_y.space_hash),
        "constant_target_centroid": constant_centroid_bridge(
            target_anchors=vectors_y[train_rows], source_dimensions=vectors_x.shape[1],
            source_space_hash=space_x.space_hash, target_space_hash=space_y.space_hash,
            anchor_set_hash=train_corr.content_hash),
        "random_map": random_map_bridge(
            source_dimensions=vectors_x.shape[1], target_dimensions=vectors_y.shape[1],
            seed=RANDOM_SEED, source_space_hash=space_x.space_hash,
            target_space_hash=space_y.space_hash),
    }

    procrustes_gram = producers["procrustes"].mapping.T @ producers["procrustes"].mapping
    if not np.allclose(procrustes_gram, np.eye(procrustes_gram.shape[0]), atol=1e-9):
        print("PROCRUSTES NOT ORTHOGONAL")
        return 1

    derived, fit_diagnostics, rows = {}, {}, {}
    for name, bridge in producers.items():
        candidate_full = bridge.transform(vectors_x)
        derived_space = bridge_output_space(space_x, bridge, space_y)
        if derived_space.space_hash == space_y.space_hash:
            print(f"DERIVED IDENTITY COLLAPSED FOR {name}")
            return 1
        derived[name] = {
            "space_hash": derived_space.space_hash,
            "derived_from": derived_space.derived_from,
            "derivation": derived_space.derivation,
            "dimensions": derived_space.dimensions,
        }
        candidate_eval = candidate_full[eval_rows]
        candidate_map = {
            f"item-{row}": candidate_eval[i] for i, row in enumerate(eval_rows)
        }
        target_map = {f"item-{row}": vectors_y[row] for row in eval_rows}
        report = compare_native_spaces(
            source_vectors=candidate_full, target_vectors=vectors_y,
            correspondence=eval_corr,
            source_space_hash=derived_space.space_hash,
            target_space_hash=space_y.space_hash,
            k=K, hard_negative_cases=cases,
            hard_negative_vectors=(candidate_map, target_map),
            scorer=scorer, scorer_id=scorer_id_of(scorer),
        )
        rows[name] = {
            "counterpart_top1": report.counterpart.top1,
            "neighborhood_overlap": report.neighborhood.mean_overlap,
            "cka": report.geometry.cka,
            "hard_negative_delta": report.hard_negatives.accuracy_delta,
            "relation_deltas": dict(report.hard_negatives.relation_deltas),
        }
        if name in fitted:
            fit_diagnostics[name] = {
                "kind": "FIT DIAGNOSTIC, not preservation evidence",
                "train_reconstruction_cosine": _mean_cosine(
                    bridge.transform(vectors_x[train_rows]), vectors_y[train_rows]),
            }

    if not all(
        rows[name]["counterpart_top1"] > rows["constant_target_centroid"]["counterpart_top1"]
        for name in fitted
    ):
        print("FITTED PRODUCERS DO NOT BEAT THE CENTROID FLOOR")
        return 1
    if not all(value["train_reconstruction_cosine"] > 0.9 for value in fit_diagnostics.values()):
        print("TRAIN RECONSTRUCTION SUSPICIOUSLY LOW")
        return 1

    try:
        code_version = importlib.metadata.version("relate-search")
    except importlib.metadata.PackageNotFoundError:
        code_version = "unknown"
    artifacts = {
        "summary.json": rows,
        "fit-diagnostics.json": fit_diagnostics,
        "derived-identities.json": derived,
        "provenance.json": {
            "benchmark_id": BENCHMARK_ID,
            "seed": fixture["seed"],
            "k": K,
            "tie_tol": TIE_TOL,
            "random_seed": RANDOM_SEED,
            "source_space": space_x.to_dict(),
            "target_space": space_y.to_dict(),
            "train_correspondence_hash": train_corr.content_hash,
            "eval_correspondence_hash": eval_corr.content_hash,
            "bridges": {
                name: {
                    "bridge_id": bridge.bridge_id,
                    "method": bridge.method,
                    "parameter_hash": bridge.parameter_hash,
                    "anchor_set_hash": bridge.anchor_set_hash,
                    "provenance": asdict(bridge.fit_provenance),
                }
                for name, bridge in producers.items()
            },
            "scorer": scorer_id_of(scorer),
            "code_version": f"relate-search {code_version}",
        },
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
    for name, row in rows.items():
        print(f"  {name:24s} cpTop1={row['counterpart_top1']:.3f} "
              f"NN={row['neighborhood_overlap']:.3f} hnDelta={row['hard_negative_delta']:+.3f}")
    print(f"wrote {len(artifacts)} artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
