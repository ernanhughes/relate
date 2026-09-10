"""CLI: the capstone story, exercised through production APIs only."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relate",
        description="RELATE -- relation-aware embedding runtime",
    )
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", help="run the capstone flow on synthetic spaces")
    sub.add_parser("spaces", help="print space-identity help")
    return parser


def run_demo() -> int:
    """Register, compare, bridge, derive, judge, explain -- one flow."""
    import numpy as np

    from relate import Observatory
    from relate.bridges import BridgeSpec, fit_bridge
    from relate.evaluation import (
        CorrespondenceSet,
        HardNegativeCase,
        PreservationPolicy,
        Requirement,
        build_preservation_profile,
        calibrate,
        calibration_transfer_results,
        compare_native_spaces,
        cosine_scorer,
        results_from_space_comparison,
        scorer_id_of,
    )

    rng = np.random.default_rng(0)
    n_anchors, dims = 120, 16
    content = rng.uniform(-3.0, 3.0, size=(n_anchors, 2))
    rest = np.column_stack([
        rng.choice([-1.0, 1.0], size=n_anchors),
        rng.uniform(-2.0, 2.0, size=n_anchors),
    ])
    base = np.hstack([content, rest])
    items, index = [], {}
    for i in range(n_anchors):
        row = base[i]
        head, polarity, tense = row[:2], row[2], row[3]
        for kind, latent in (
            ("base", row),
            ("paraphrase", row + rng.normal(0, 0.05, 4)),
            ("negation", np.hstack([head, [-polarity], [tense]])),
            ("topic", np.hstack([rng.uniform(-3.0, 3.0, 2), [polarity], [tense]])),
        ):
            index[(i, kind)] = len(items)
            items.append(latent)
    latent = np.array(items)
    mixing = rng.normal(size=(4, dims))
    coarse_mixing = mixing.copy()
    coarse_mixing[2:, :] *= 0.15
    nuisance = rng.normal(size=(len(items), dims))
    nuisance *= 2.0 / np.linalg.norm(nuisance, axis=1, keepdims=True)

    def embed(matrix: np.ndarray) -> np.ndarray:
        signal = latent @ matrix
        signal *= 2.5 / np.linalg.norm(signal, axis=1, keepdims=True)
        vectors = signal + nuisance
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    fine, coarse = embed(mixing), embed(coarse_mixing)
    ids = [f"item-{i}" for i in range(len(items))]

    def rows_for(anchors: range) -> list[int]:
        return [index[(a, k)] for a in anchors
                for k in ("base", "paraphrase", "negation", "topic")]

    def correspondence(rows: list[int]) -> CorrespondenceSet:
        return CorrespondenceSet(
            ids=tuple(ids[r] for r in rows),
            source_rows=tuple(rows),
            target_rows=tuple(rows),
        )

    train_corr = correspondence(rows_for(range(80)))
    eval_corr = correspondence(rows_for(range(80, 120)))
    eval_rows = [index[(a, k)] for a in range(80, 120)
                 for k in ("base", "paraphrase", "negation", "topic")]
    cases = []
    for anchor in range(80, 120):
        positive = index[(anchor, "paraphrase")]
        for negative, relation in ((index[(anchor, "negation")], "negation"),
                                   (index[(anchor, "topic")], "topic-related")):
            cases.append(HardNegativeCase(
                case_id=f"demo-{anchor}-{relation}", anchor_id=ids[index[(anchor, "base")]],
                positive_id=ids[positive], negative_id=ids[negative],
                relation=relation, group="demo"))

    runtime = Observatory()
    coarse_space = runtime.register_space(model="demo-coarse", dimensions=dims)
    fine_space = runtime.register_space(model="demo-fine", dimensions=dims)
    print(f"SOURCE   {coarse_space.short}")
    print(f"TARGET   {fine_space.short}")

    native = runtime.compare_spaces(
        coarse, fine, correspondence=eval_corr,
        source_space=coarse_space, target_space=fine_space,
    )
    print(f"NATIVE   CKA {native.geometry.cka:.2f}  "
          f"NN overlap {native.neighborhood.mean_overlap:.2f}  "
          f"counterpart top-1 {native.counterpart.top1:.2f}")

    bridge = runtime.fit_bridge(
        coarse, fine, source_space=coarse_space, target_space=fine_space,
        correspondence=train_corr, method="ridge", params={"alpha": 1.0},
    )
    derived = runtime.bridge_space(coarse_space, bridge, fine_space)
    print(f"BRIDGE   {bridge.method} {bridge.direction}  "
          f"anchors {bridge.fit_provenance.n_anchors}")
    print(f"DERIVED  {derived.short}  (native target hash never reused)")

    candidate = bridge.transform(coarse)
    scorer = cosine_scorer()
    candidate_map = {f"item-{r}": candidate[r] for r in eval_rows}
    target_map = {f"item-{r}": fine[r] for r in eval_rows}
    comparison = compare_native_spaces(
        source_vectors=candidate, target_vectors=fine,
        correspondence=eval_corr, k=5,
        hard_negative_cases=cases,
        hard_negative_vectors=(candidate_map, target_map),
        scorer=scorer, scorer_id=scorer_id_of(scorer),
    )
    native_pos = [scorer(target_map[c.anchor_id], target_map[c.positive_id]) for c in cases]
    native_neg = [scorer(target_map[c.anchor_id], target_map[c.negative_id]) for c in cases]
    native_fit = calibrate(native_pos, native_neg)
    candidate_pos = [scorer(candidate_map[c.anchor_id], candidate_map[c.positive_id])
                     for c in cases]
    candidate_neg = [scorer(candidate_map[c.anchor_id], candidate_map[c.negative_id])
                     for c in cases]
    results = results_from_space_comparison(comparison)
    results.extend(calibration_transfer_results(
        native_fit=native_fit, candidate_positive_scores=candidate_pos,
        candidate_negative_scores=candidate_neg))
    policies = (
        PreservationPolicy(scope="retrieval", requirements=(
            Requirement("counterpart_recovery", "recall_at_1", "min_value", 0.90),)),
        PreservationPolicy(scope="threshold_transfer", requirements=(
            Requirement("calibration_transfer", "far_increase", "max_delta", 0.05),)),
    )
    profile = build_preservation_profile(
        source_space_hash=coarse_space.space_hash,
        candidate_space_hash=derived.space_hash,
        target_space_hash=fine_space.space_hash,
        results=results, policies=list(policies), bridge_id=bridge.bridge_id,
        evaluation_correspondence_hash=eval_corr.content_hash,
        scorer=scorer_id_of(scorer),
    )
    for verdict in profile.verdicts:
        print(f"VERDICT  {verdict.scope:17s} {verdict.verdict.value}")
    print(f"WHY      {profile.explain('threshold_transfer').replace(chr(10), ' / ')}")
    # The point, in one line: a map proves coordinates can be
    # transformed; the profile determines what it is good for.
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from importlib.metadata import version

        try:
            print(version("relate-search"))
        except Exception:
            print("0.1.0")
        return 0
    if args.command == "demo":
        return run_demo()
    if args.command == "spaces":
        print("Every vector carries a space_hash. Compare spaces freely; "
              "mixing without a measured bridge is denied.")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
