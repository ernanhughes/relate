"""Native cross-space benchmark: coarse agreement, fine disagreement.

    python benchmarks/cross-space/run.py          # write expected/
    python benchmarks/cross-space/run.py --check  # verify byte-identical

Conditions over the typed native spaces from ``native_spaces.py``:

- identity (X vs X): the control -- everything agrees;
- native pair (X vs Y): the question -- direct same-dimension comparison
  with no fitted map (the identity/no-op control: decent coarse geometry
  is evidence, never authorization);
- permuted (X vs row-shuffled X): the chance floor.

The native-pair report must hold the thesis shape in one artifact:
CKA high, counterpart recovery near-perfect, neighborhoods moderate,
negation/temporal ordering degraded, topic ordering preserved, and a
transferred operating point that no longer holds its FAR. Calibration
transfer uses 3C machinery in this runner -- never inside the comparison
evaluator. Comparing is allowed throughout; mixing stays denied
(asserted via the guard).
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

from native_spaces import build_native_spaces  # noqa: E402
from relate import RelateError  # noqa: E402
from relate.evaluation import (  # noqa: E402
    calibrate,
    compare_native_spaces,
    cosine_scorer,
    identity_correspondence,
    permuted_copy,
    require_same_space_for_mixing,
    scorer_id_of,
)
from relate.spaces import SpaceIdentity  # noqa: E402

BENCHMARK_ID = "cross-space-v1"
K = 10
TIE_TOL = 1e-9
PERMUTATION_SEED = 5


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    fixture = build_native_spaces()
    vectors_x, vectors_y, ids = fixture["x"], fixture["y"], fixture["ids"]
    cases = fixture["cases"]
    scorer = cosine_scorer()
    correspondence = identity_correspondence(ids)
    map_x = {name: vectors_x[i] for i, name in enumerate(ids)}
    map_y = {name: vectors_y[i] for i, name in enumerate(ids)}

    space_x = SpaceIdentity(model="synthetic-native-x-v1", dimensions=vectors_x.shape[1])
    space_y = SpaceIdentity(
        model="synthetic-native-y-v1",
        dimensions=vectors_y.shape[1],
        revision="fine-dims-weakened",
    )
    try:
        require_same_space_for_mixing(space_x.space_hash, space_y.space_hash)
        print("MIXING GUARD DID NOT FIRE")
        return 1
    except RelateError:
        pass  # comparing is allowed; mixing is denied.

    identity = compare_native_spaces(
        source_vectors=vectors_x, target_vectors=vectors_x,
        correspondence=correspondence,
        source_space_hash=space_x.space_hash, target_space_hash=space_x.space_hash,
        k=K, hard_negative_cases=cases,
        hard_negative_vectors=(map_x, map_x), scorer=scorer,
        scorer_id=scorer_id_of(scorer),
    )
    native = compare_native_spaces(
        source_vectors=vectors_x, target_vectors=vectors_y,
        correspondence=correspondence,
        source_space_hash=space_x.space_hash, target_space_hash=space_y.space_hash,
        k=K, hard_negative_cases=cases,
        hard_negative_vectors=(map_x, map_y), scorer=scorer,
        scorer_id=scorer_id_of(scorer),
    )
    shuffled = compare_native_spaces(
        source_vectors=vectors_x,
        target_vectors=permuted_copy(vectors_x, PERMUTATION_SEED),
        correspondence=correspondence,
        source_space_hash=space_x.space_hash, target_space_hash=space_x.space_hash,
        k=K,
    )

    # Calibration transfer: 3C fit on X scores, applied to Y scores.
    pos_x = [scorer(map_x[c.anchor_id], map_x[c.positive_id]) for c in cases]
    neg_x = [scorer(map_x[c.anchor_id], map_x[c.negative_id]) for c in cases]
    pos_y = [scorer(map_y[c.anchor_id], map_y[c.positive_id]) for c in cases]
    neg_y = [scorer(map_y[c.anchor_id], map_y[c.negative_id]) for c in cases]
    fit = calibrate(pos_x, neg_x)
    point = fit.operating_point
    applied = {}
    for name, (pos, neg) in (("x", (pos_x, neg_x)), ("y", (pos_y, neg_y))):
        applied[name] = {
            "far_at_accept": float(np.mean(np.array(neg) >= point.threshold_high)),
            "frr_at_reject": float(np.mean(np.array(pos) < point.threshold_low)),
        }
    transfer = {
        "eer_threshold_x": fit.eer_threshold,
        "eer_rate_x": fit.eer_rate,
        "auc_x": fit.auc,
        "threshold_low": point.threshold_low,
        "threshold_high": point.threshold_high,
        "applied": applied,
        "far_ratio_y_over_x": applied["y"]["far_at_accept"] / applied["x"]["far_at_accept"],
    }

    # Regression gates: the thesis shape, not sacred constants.
    deltas = native.hard_negatives.relation_deltas
    checks = [
        ("cka", native.geometry.cka > 0.85),
        ("counterpart", native.counterpart.top1 > 0.99),
        ("neighborhood", 0.35 <= native.neighborhood.mean_overlap <= 0.70),
        ("negation-degraded", deltas["negation"] < -0.10),
        ("temporal-degraded", deltas["temporal-mismatch"] < -0.10),
        ("swap-degraded", deltas["relation-swap"] < -0.03),
        ("topic-preserved", abs(deltas["topic-related"]) < 0.05),
        ("transfer-fails", transfer["far_ratio_y_over_x"] > 1.5),
        ("identity-control",
         identity.neighborhood.mean_overlap == 1.0 and identity.counterpart.top1 == 1.0),
        ("permuted-control",
         shuffled.neighborhood.mean_overlap < 0.1 and shuffled.counterpart.top1 < 0.05),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        print(f"THESIS SHAPE NOT REPRODUCED: {failed}")
        return 1

    summary = {
        name: {
            "cka": report.geometry.cka,
            "neighborhood_overlap": report.neighborhood.mean_overlap,
            "top1_agreement": report.neighborhood.top1_agreement,
            "counterpart_top1": report.counterpart.top1,
            "hard_negative_delta": (
                report.hard_negatives.accuracy_delta
                if report.hard_negatives is not None else None
            ),
        }
        for name, report in (("identity", identity), ("native_xy", native), ("permuted", shuffled))
    }
    try:
        code_version = importlib.metadata.version("relate-search")
    except importlib.metadata.PackageNotFoundError:
        code_version = "unknown"
    artifacts = {
        "summary.json": summary,
        "space-comparison-xy.json": asdict(native),
        "hardneg-delta-xy.json": asdict(native.hard_negatives),
        "calibration-transfer.json": transfer,
        "provenance.json": {
            "benchmark_id": BENCHMARK_ID,
            "seed": fixture["seed"],
            "k": K,
            "tie_tol": TIE_TOL,
            "permutation_seed": PERMUTATION_SEED,
            "source_space": space_x.to_dict(),
            "target_space": space_y.to_dict(),
            "correspondence_hash": correspondence.content_hash,
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
    for name, row in summary.items():
        print(f"  {name:10s} CKA={row['cka']:.3f} NN={row['neighborhood_overlap']:.3f} "
              f"cpTop1={row['counterpart_top1']:.3f} hnDelta={row['hard_negative_delta']}")
    print(f"wrote {len(artifacts)} artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
