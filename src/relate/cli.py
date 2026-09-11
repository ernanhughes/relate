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
        PreservationResult,
        ReferenceFrame,
        Requirement,
        build_preservation_profile,
        calibrate,
        calibration_transfer_results,
        compare_native_spaces,
        cosine_scorer,
        cosine_similarity,
        results_from_space_comparison,
        scorer_id_of,
    )
    from relate.transformations import (
        fit_constant_delta,
        fit_pca,
        identity_map,
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
    eval_rows = rows_for(range(80, 120))
    eval_corr = correspondence(eval_rows)
    eval_rows = rows_for(range(80, 120))
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

    print("COMPRESSION")
    cartridge = fit_pca(fine, output_dimensions=12,
                        source_space_hash=fine_space.space_hash)
    cartridge_space = runtime.derive_transformation_space(
        parent=fine_space, transformation_id=cartridge.transformation_id,
        parameters={"method": "pca", "output_dimensions": 12}, dimensions=12)
    cartridge_candidates = cartridge.transform(fine)
    compression = runtime.evaluate_compression(
        cartridge, fine, fine, correspondence=eval_corr,
        candidate_space=cartridge_space, reference_space=fine_space,
        k=5, hard_negative_cases=cases,
        hard_negative_vectors=(
            {f"item-{r}": cartridge_candidates[r] for r in eval_rows},
            {f"item-{r}": fine[r] for r in eval_rows},
        ),
        scorer=scorer, scorer_id=scorer_id_of(scorer),
        policies=[
            PreservationPolicy(scope="retrieval", requirements=(
                Requirement("neighborhood_structure", "overlap_at_5",
                            "min_value", 0.60),)),
            PreservationPolicy(scope="threshold_transfer", requirements=(
                Requirement("calibration_transfer", "far_increase",
                            "max_delta", 0.05),)),
        ],
    )
    for verdict in compression.profile.verdicts:
        print(f"VERDICT  compression/{verdict.scope:17s} {verdict.verdict.value}")

    print("OPERATOR")
    shift = np.zeros(4)
    shift[0] = 3.0
    op_source, op_target = [], []
    op_rng = np.random.default_rng(2)
    for _ in range(60):
        latent = op_rng.normal(size=(4,))
        op_source.append(latent)
        op_target.append(latent + shift)
    op_source = np.array(op_source)
    op_target = np.array(op_target)
    op_space = runtime.register_space(model="demo-operator", dimensions=4)
    delta = fit_constant_delta(op_source[:40], op_target[:40], relation="weakened",
                               source_space_hash=op_space.space_hash)
    plain = identity_map(relation="weakened", source_space_hash=op_space.space_hash)
    op_corr = CorrespondenceSet(ids=tuple(f"op-{i}" for i in range(20)),
                                source_rows=tuple(range(40, 60)),
                                target_rows=tuple(range(40, 60)))
    op_source_eval = op_source[40:]
    op_target_eval = op_target[40:]
    fidelity_policy = PreservationPolicy(scope="operator_fidelity", requirements=(
        Requirement("operator_fidelity", "mean_cosine_to_native", "min_value", 0.85),))
    for operator, label in ((delta, "constant_delta"), (plain, "identity_map")):
        judged = runtime.evaluate_operator(
            operator, op_source, op_target, correspondence=op_corr,
            reference_space=op_space,
            policies=[fidelity_policy],
        )
        candidate = operator.transform(op_source_eval)
        fidelity = float(np.mean([
            cosine_similarity(candidate[i], op_target_eval[i])
            for i in range(len(op_target_eval))]))
        rebuilt = build_preservation_profile(
            source_space_hash=op_space.space_hash,
            candidate_space_hash=f"derived:{operator.transformation_id}",
            target_space_hash=op_space.space_hash,
            results=list(judged.profile.results) + [PreservationResult(
                capability="operator_fidelity", metric="mean_cosine_to_native",
                value=fidelity, reference_value=1.0,
                reference_frame=ReferenceFrame.TARGET_NATIVE,
                evidence_id=operator.transformation_id)],
            policies=[fidelity_policy],
            bridge_id=operator.transformation_id,
            evaluation_correspondence_hash=op_corr.content_hash,
            scorer="cosine_similarity",
        )
        print(f"VERDICT  operator/{label:17s} "
              f"{rebuilt.verdict_for('operator_fidelity').verdict.value} ({fidelity:.3f})")
    print("DOCTRINE transform -> derive identity -> measure -> scoped verdict; "
          "lineage composes, permission does not.")
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
