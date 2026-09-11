"""Compression benchmark: dimensionality is not a permission boundary.

    python benchmarks/compression/run.py          # write expected/
    python benchmarks/compression/run.py --check  # verify byte-identical

PCA / random projection / prefix truncation across 64/32/16/8 dims,
plus the identity reference. Every producer flows through
``evaluate_transformation`` against the native source authority
(SOURCE_NATIVE frame); calibration transfer appends 3C reuse; declared
policies verdict per scope. Knees differ by scope -- that is the result.
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

from compression_cases import WIDTHS, build_compression_cases  # noqa: E402
from knee import find_smallest_passing_dimension  # noqa: E402
from relate.evaluation import (  # noqa: E402
    PreservationPolicy,
    ReferenceFrame,
    Requirement,
    build_preservation_profile,
    calibrate,
    calibration_transfer_results,
    cosine_scorer,
    scorer_id_of,
)
from relate.spaces import SpaceIdentity  # noqa: E402
from relate.transformations import (  # noqa: E402
    derived_transformation_space,
    evaluate_transformation,
    fit_pca,
    prefix_truncation,
    random_projection,
)

BENCHMARK_ID = "compression-v1"
K = 10
TIE_TOL = 1e-9
RANDOM_SEED = 0

POLICIES = (
    PreservationPolicy(
        scope="retrieval",
        requirements=(
            Requirement("neighborhood_structure", "overlap_at_10", "min_value", 0.78),
            Requirement("hard_negative_ordering", "accuracy_delta", "max_abs_delta", 0.10),
            Requirement("representation_similarity", "cka", "min_value", 0.95,
                        advisory=True),
        ),
    ),
    PreservationPolicy(
        scope="fine_ordering",
        requirements=(
            Requirement("hard_negative_ordering", "accuracy_delta", "max_abs_delta", 0.05),
            Requirement("hard_negative_ordering/negation", "accuracy_delta",
                        "max_abs_delta", 0.08),
        ),
    ),
    PreservationPolicy(
        scope="threshold_transfer",
        requirements=(
            Requirement("calibration_transfer", "far_increase", "max_delta", 0.05),
            Requirement("calibration_transfer", "frr_increase", "max_delta", 0.05),
        ),
    ),
)


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    fixture = build_compression_cases()
    vectors, ids = fixture["vectors"], fixture["ids"]
    fit_rows = np.asarray(fixture["fit_rows"])
    eval_rows = np.asarray(fixture["eval_rows"])
    cases = fixture["cases"]
    scorer = cosine_scorer()

    source_space = SpaceIdentity(model="synthetic-compression-128-v1",
                                 dimensions=vectors.shape[1])
    native_map = {f"item-{r}": vectors[r] for r in eval_rows}
    native_pos = [scorer(native_map[c.anchor_id], native_map[c.positive_id]) for c in cases]
    native_neg = [scorer(native_map[c.anchor_id], native_map[c.negative_id]) for c in cases]
    native_fit = calibrate(native_pos, native_neg)

    from relate.evaluation import CorrespondenceSet

    def correspondence(rows) -> CorrespondenceSet:
        return CorrespondenceSet(
            ids=tuple(ids[r] for r in rows),
            source_rows=tuple(int(r) for r in rows),
            target_rows=tuple(int(r) for r in rows),
        )

    fit_corr, eval_corr = correspondence(fit_rows), correspondence(eval_rows)
    assert fit_corr.content_hash != eval_corr.content_hash

    producers = {}
    for width in WIDTHS:
        producers[f"pca-{width}"] = fit_pca(
            vectors[fit_rows], output_dimensions=width,
            source_space_hash=source_space.space_hash,
            fit_corpus_hash=fit_corr.content_hash)
        producers[f"random-{width}"] = random_projection(
            vectors.shape[1], width, seed=RANDOM_SEED,
            source_space_hash=source_space.space_hash)
        producers[f"prefix-{width}"] = prefix_truncation(
            vectors.shape[1], width, source_space_hash=source_space.space_hash,
            training_support="unknown")
    assert producers["prefix-32"].artifact.spec.kind == "prefix_truncation"
    assert (producers["prefix-32"].artifact.provenance.detail["training_support"]
            == "unknown")
    assert (producers["pca-32"].artifact.provenance.detail["fit_corpus_hash"]
            == fit_corr.content_hash)

    profiles, summaries = {}, {}
    for name, producer in producers.items():
        derived = derived_transformation_space(
            source_space, producer.transformation_id,
            {"method": producer.artifact.spec.kind,
             "output_dimensions": producer.output_dimensions},
            dimensions=producer.output_dimensions)
        assert derived.space_hash != source_space.space_hash
        candidate_full = producer.transform(vectors)
        candidate_map = {f"item-{r}": candidate_full[r] for r in eval_rows}
        target_map = {f"item-{r}": vectors[r] for r in eval_rows}
        profile = evaluate_transformation(
            transformation=producer, source_vectors=vectors, reference_vectors=vectors,
            correspondence=eval_corr, reference_frame=ReferenceFrame.SOURCE_NATIVE,
            candidate_space_hash=derived.space_hash,
            reference_space_hash=source_space.space_hash,
            k=K, with_counterpart=False, hard_negative_cases=cases,
            hard_negative_vectors=(candidate_map, target_map),
            scorer=scorer, scorer_id=scorer_id_of(scorer),
            policies=list(POLICIES),
        )
        candidate_pos = [scorer(candidate_map[c.anchor_id], candidate_map[c.positive_id])
                         for c in cases]
        candidate_neg = [scorer(candidate_map[c.anchor_id], candidate_map[c.negative_id])
                         for c in cases]
        transfer = calibration_transfer_results(
            native_fit=native_fit, candidate_positive_scores=candidate_pos,
            candidate_negative_scores=candidate_neg, frame=ReferenceFrame.SOURCE_NATIVE)
        full = build_preservation_profile(
            source_space_hash=source_space.space_hash,
            candidate_space_hash=derived.space_hash,
            target_space_hash=source_space.space_hash,
            results=list(profile.results) + transfer,
            policies=list(POLICIES),
            bridge_id=producer.transformation_id,
            evaluation_correspondence_hash=eval_corr.content_hash,
            scorer=scorer_id_of(scorer),
        )
        profiles[name] = full
        summaries[name] = {
            verdict.scope: verdict.verdict.value for verdict in full.verdicts
        }

    knees = {
        scope: find_smallest_passing_dimension([
            (int(name.rsplit("-", 1)[1]), profiles[name].usable_for(scope))
            for name in profiles if name.startswith("pca-")
        ])
        for scope in ("retrieval", "fine_ordering", "threshold_transfer")
    }
    # Regression gates: the task-dependent shape, not sacred constants.
    if not (knees["retrieval"] == 64 and knees["fine_ordering"] == 16
            and knees["threshold_transfer"] is None):
        print(f"KNEES OUT OF SHAPE: {knees}")
        return 1
    pca8 = profiles["pca-8"]
    cka8 = pca8.result("representation_similarity", "cka")
    if not (cka8 is not None and cka8.value > 0.95
            and pca8.verdict_for("fine_ordering").verdict.value == "FAIL"):
        print("GEOMETRY/SEMANTICS SPLIT NOT VISIBLE")
        return 1
    pca64 = profiles["pca-64"]
    rand64 = profiles["random-64"]
    pca32 = profiles["pca-32"]
    rand32 = profiles["random-32"]
    pca_nn = pca32.result("neighborhood_structure", "overlap_at_10").value
    rand_nn = rand32.result("neighborhood_structure", "overlap_at_10").value
    if not (pca64.usable_for("retrieval")
            and not rand64.usable_for("retrieval")
            and pca_nn - rand_nn > 0.2):
        print("STRUCTURED VS RANDOM NOT SEPARATED")
        return 1

    try:
        code_version = importlib.metadata.version("relate-search")
    except importlib.metadata.PackageNotFoundError:
        code_version = "unknown"
    artifacts = {
        "summary.json": {"verdicts": summaries, "pca_knees": knees},
        "policies.json": [
            {"scope": policy.scope, "policy_id": policy.policy_id,
             "requirements": [asdict(requirement) for requirement in policy.requirements]}
            for policy in POLICIES
        ],
        **{f"profiles/{name}.json": asdict(profile) for name, profile in profiles.items()},
        "provenance.json": {
            "benchmark_id": BENCHMARK_ID,
            "seed": fixture["seed"],
            "widths": list(WIDTHS),
            "k": K,
            "tie_tol": TIE_TOL,
            "source_space": source_space.to_dict(),
            "fit_correspondence_hash": fit_corr.content_hash,
            "eval_correspondence_hash": eval_corr.content_hash,
            "reference_frame": ReferenceFrame.SOURCE_NATIVE.value,
            "scorer": scorer_id_of(scorer),
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
        print(f"  {name:12s} {verdicts}")
    print(f"pca knees: {knees}")
    print(f"wrote {len(artifacts)} artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
