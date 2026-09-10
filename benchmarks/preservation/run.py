"""Preservation benchmark: one full profile per producer, mixed verdicts.

    python benchmarks/preservation/run.py          # write expected/
    python benchmarks/preservation/run.py --check  # verify byte-identical

Six producers fit on train anchors and are judged on held-out eval rows:
4A comparison against the native target, 3C calibration transfer of the
native operating point onto candidate scores, and round-trip evidence
from reverse fits (evidence only -- round-trip never gates target
claims). Declared policies turn the same evidence into scoped verdicts;
no producer passes every scope.
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

from preservation_cases import build_preservation_cases  # noqa: E402
from relate import RelateError  # noqa: E402
from relate.bridges import (  # noqa: E402
    BridgeSpec,
    bridge_output_space,
    constant_centroid_bridge,
    fit_bridge,
    identity_bridge,
    random_map_bridge,
)
from relate.evaluation import (  # noqa: E402
    PreservationPolicy,
    ReferenceFrame,
    Requirement,
    PreservationResult,
    build_preservation_profile,
    calibrate,
    calibration_transfer_results,
    compare_native_spaces,
    compare_neighborhoods,
    cosine_scorer,
    results_from_space_comparison,
    scorer_id_of,
)
from relate.spaces import SpaceIdentity  # noqa: E402

BENCHMARK_ID = "preservation-v1"
K = 10
TIE_TOL = 1e-9
RANDOM_SEED = 0

POLICIES = (
    PreservationPolicy(
        scope="retrieval",
        requirements=(
            Requirement("counterpart_recovery", "recall_at_1", "min_value", 0.90),
            Requirement("hard_negative_ordering", "accuracy_delta", "max_abs_delta", 0.15),
            Requirement("neighborhood_structure", "overlap_at_10", "min_value", 0.50,
                        advisory=True),
        ),
    ),
    PreservationPolicy(
        scope="threshold_transfer",
        requirements=(
            Requirement("calibration_transfer", "far_increase", "max_delta", 0.05),
            Requirement("calibration_transfer", "frr_increase", "max_delta", 0.05),
            Requirement("calibration_transfer", "threshold_shift", "max_abs_delta", 0.05),
        ),
    ),
    PreservationPolicy(
        scope="neighborhood_use",
        requirements=(
            Requirement("neighborhood_structure", "overlap_at_10", "min_value", 0.50),
        ),
    ),
)


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

    fixture = build_preservation_cases()
    vectors_x, vectors_y = fixture["x"], fixture["y"]
    train_corr, eval_corr = (
        fixture["train_correspondence"], fixture["eval_correspondence"])
    assert train_corr.content_hash != eval_corr.content_hash
    cases = fixture["cases"]
    scorer = cosine_scorer()
    eval_rows = np.asarray(fixture["eval_rows"])
    train_rows = np.asarray(fixture["train_rows"])

    space_coarse = SpaceIdentity(model="synthetic-preservation-coarse-v1",
                                 dimensions=vectors_y.shape[1],
                                 revision="fine-dims-weakened")
    space_fine = SpaceIdentity(model="synthetic-preservation-fine-v1",
                               dimensions=vectors_x.shape[1])

    # Direction under test: coarse -> fine. Candidates inherit the coarse
    # source's blindness, so fine distinctions fail while coarse retrieval
    # transfers -- the book's asymmetry, made executable.
    fitted = {}
    for method in ("procrustes", "linear", "ridge"):
        fitted[method] = fit_bridge(
            source_vectors=vectors_y, target_vectors=vectors_x,
            correspondence=train_corr,
            spec=BridgeSpec(
                source_space_hash=space_coarse.space_hash,
                target_space_hash=space_fine.space_hash,
                method=method, params={"alpha": 1.0} if method == "ridge" else {}),
        )
    reverse = {}
    for method in ("procrustes", "linear", "ridge"):
        reverse[method] = fit_bridge(
            source_vectors=vectors_x, target_vectors=vectors_y,
            correspondence=train_corr,
            spec=BridgeSpec(
                source_space_hash=space_fine.space_hash,
                target_space_hash=space_coarse.space_hash,
                method=method, params={"alpha": 1.0} if method == "ridge" else {}),
        )
    producers = {
        **fitted,
        "identity": identity_bridge(
            vectors_x.shape[1], source_space_hash=space_coarse.space_hash,
            target_space_hash=space_fine.space_hash),
        "constant_target_centroid": constant_centroid_bridge(
            target_anchors=vectors_x[train_rows], source_dimensions=vectors_y.shape[1],
            source_space_hash=space_coarse.space_hash,
            target_space_hash=space_fine.space_hash,
            anchor_set_hash=train_corr.content_hash),
        "random_map": random_map_bridge(
            source_dimensions=vectors_y.shape[1], target_dimensions=vectors_x.shape[1],
            seed=RANDOM_SEED, source_space_hash=space_coarse.space_hash,
            target_space_hash=space_fine.space_hash),
    }
    reverse["identity"] = identity_bridge(
        vectors_x.shape[1], source_space_hash=space_fine.space_hash,
        target_space_hash=space_coarse.space_hash)

    # Native-target calibration: the fine-target authority transfer is judged against.
    native_map = {f"item-{r}": vectors_x[r] for r in eval_rows}
    native_pos = [scorer(native_map[c.anchor_id], native_map[c.positive_id]) for c in cases]
    native_neg = [scorer(native_map[c.anchor_id], native_map[c.negative_id]) for c in cases]
    native_fit = calibrate(native_pos, native_neg)

    profiles, summaries = {}, {}
    for name, bridge in producers.items():
        candidate_full = bridge.transform(vectors_y)
        derived_space = bridge_output_space(space_coarse, bridge, space_fine)
        assert derived_space.space_hash != space_fine.space_hash
        candidate_eval = candidate_full[eval_rows]
        candidate_map = {f"item-{row}": candidate_eval[i]
                         for i, row in enumerate(eval_rows)}
        target_map = {f"item-{row}": vectors_x[row] for row in eval_rows}
        comparison = compare_native_spaces(
            source_vectors=candidate_full, target_vectors=vectors_x,
            correspondence=eval_corr,
            source_space_hash=derived_space.space_hash,
            target_space_hash=space_fine.space_hash,
            k=K, hard_negative_cases=cases,
            hard_negative_vectors=(candidate_map, target_map),
            scorer=scorer, scorer_id=scorer_id_of(scorer),
        )
        results = results_from_space_comparison(comparison)
        candidate_pos = [scorer(candidate_map[c.anchor_id], candidate_map[c.positive_id])
                         for c in cases]
        candidate_neg = [scorer(candidate_map[c.anchor_id], candidate_map[c.negative_id])
                         for c in cases]
        try:
            results.extend(calibration_transfer_results(
                native_fit=native_fit,
                candidate_positive_scores=candidate_pos,
                candidate_negative_scores=candidate_neg,
            ))
            transfer_measured = True
        except RelateError:  # degenerate candidates fail closed
            transfer_measured = False
        if name in reverse:
            # Round trip returns to the coarse frame: evidence about the
            # pair of maps, never authority for fine-target claims.
            roundtrip_eval = reverse[name].transform(candidate_eval)
            roundtrip_overlap = compare_neighborhoods(
                roundtrip_eval, vectors_y[eval_rows],
                [f"item-{row}" for row in eval_rows], k=K).mean_overlap
            results.append(PreservationResult(
                capability="round_trip", metric="mean_cosine",
                value=_mean_cosine(roundtrip_eval, vectors_y[eval_rows]),
                reference_value=1.0,
                reference_frame=ReferenceFrame.SOURCE_NATIVE,
                evidence_id="reverse-fit"))
            results.append(PreservationResult(
                capability="round_trip", metric="neighborhood_overlap",
                value=roundtrip_overlap, reference_value=1.0,
                reference_frame=ReferenceFrame.SOURCE_NATIVE,
                evidence_id="reverse-fit"))
        profile = build_preservation_profile(
            source_space_hash=space_coarse.space_hash,
            candidate_space_hash=derived_space.space_hash,
            target_space_hash=space_fine.space_hash,
            results=results,
            policies=list(POLICIES),
            bridge_id=bridge.bridge_id,
            evaluation_correspondence_hash=eval_corr.content_hash,
            scorer=scorer_id_of(scorer),
        )
        profiles[name] = profile
        negation = profile.result("hard_negative_ordering/negation", "accuracy_delta")
        summaries[name] = {
            verdict.scope: verdict.verdict.value for verdict in profile.verdicts
        }
        summaries[name]["negation_delta"] = negation.delta if negation else None
        summaries[name]["transfer_measured"] = transfer_measured

    # Regression gates: controls fail retrieval; fitted mix verdicts;
    # nobody passes every scope.
    if profiles["constant_target_centroid"].usable_for("retrieval"):
        print("CENTROID PASSES RETRIEVAL")
        return 1
    if profiles["random_map"].usable_for("retrieval"):
        print("RANDOM MAP PASSES RETRIEVAL")
        return 1
    for name in ("procrustes", "linear", "ridge"):
        if "retrieval" not in profiles[name].usable_scopes:
            print(f"{name} FAILS RETRIEVAL")
            return 1
        if profiles[name].usable_for("threshold_transfer"):
            print(f"{name} PASSES THRESHOLD TRANSFER")
            return 1
    if any(
        all(status == "PASS" for status in row.values() if isinstance(status, str))
        for row in summaries.values()
    ):
        print("A UNIVERSAL WINNER EXISTS")
        return 1

    try:
        code_version = importlib.metadata.version("relate-search")
    except importlib.metadata.PackageNotFoundError:
        code_version = "unknown"
    artifacts = {
        "summary.json": summaries,
        "policies.json": [
            {"scope": policy.scope, "policy_id": policy.policy_id,
             "requirements": [asdict(requirement) for requirement in policy.requirements]}
            for policy in POLICIES
        ],
        **{f"profiles/{name}.json": asdict(profile) for name, profile in profiles.items()},
        "provenance.json": {
            "benchmark_id": BENCHMARK_ID,
            "seed": fixture["seed"],
            "k": K,
            "tie_tol": TIE_TOL,
            "source_space": space_coarse.to_dict(),
            "target_space": space_fine.to_dict(),
            "train_correspondence_hash": train_corr.content_hash,
            "eval_correspondence_hash": eval_corr.content_hash,
            "scorer": scorer_id_of(scorer),
            "policy_ids": [policy.policy_id for policy in POLICIES],
            "code_version": f"relate-search {code_version}",
        },
    }
    out_dir = HERE / "expected"
    if args.check:
        failures = []
        for leaf, payload in artifacts.items():
            path = out_dir / leaf
            if not path.exists() or path.read_text() != _dump(payload):
                failures.append(leaf)
        if failures:
            print(f"{len(failures)} differing artifact(s): {failures}")
            return 1
        print("MATCH")
        return 0
    out_dir.mkdir(exist_ok=True)
    (out_dir / "profiles").mkdir(exist_ok=True)
    for leaf, payload in artifacts.items():
        (out_dir / leaf).write_text(_dump(payload))
    for name, row in summaries.items():
        verdicts = " ".join(f"{k}={v}" for k, v in row.items() if isinstance(v, str))
        print(f"  {name:24s} {verdicts}")
    print(f"wrote {len(artifacts)} artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
