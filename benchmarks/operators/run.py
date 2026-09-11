"""Operator bakeoff: which rung, if any, earns the transformation.

    python benchmarks/operators/run.py          # write expected/
    python benchmarks/operators/run.py --check  # verify byte-identical

Real RELATE-DOC pairs (frozen texts, hashed content, nine exact
transformation classes, deterministic train/eval splits) meet a
mirror encoder: deterministic per-pair latents whose class recipes
reproduce the Wave-5 taxonomy (five identity-like edits, one
consistent direction, three edits no ladder rung captures). The
embeddings are synthetic; the case identities, counts, splits, and
the selection structure are the real result shape.

Per class, four operator hypotheses fit on train pairs and face
held-out native targets through ``evaluate_transformation`` plus an
operator-fidelity result (mean cosine to native, 0.85 bar). Selection
reads verdicts only, simplest first; NONE_PASS is a healthy outcome.
"""

from __future__ import annotations

import argparse
import hashlib
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

from operator_cases import CORPUS_HASH, TRANSFORMATION_TYPES, load_doc_cases, split_train_eval  # noqa: E402
from relate.evaluation import (  # noqa: E402
    PreservationPolicy,
    ReferenceFrame,
    Requirement,
    PreservationResult,
    build_preservation_profile,
    cosine_scorer,
    scorer_id_of,
)
from relate.spaces import SpaceIdentity  # noqa: E402
from relate.transformations import (  # noqa: E402
    OPERATOR_COMPLEXITY,
    ContentTransformationCase,
    OperatorSelectionOutcome,
    evaluate_transformation,
    fit_affine_operator,
    fit_constant_delta,
    fit_linear_operator,
    hash_case_set,
    identity_map,
    select_simplest_passing,
)

BENCHMARK_ID = "operators-v1"
SEED = 0
N_DIMS = 64
N_LATENT = 8
FIDELITY_BAR = 0.85
SCOPE = "operator_fidelity"

IDENTITY_LIKE = {"active_to_passive", "present_to_past", "relation_swap",
                 "claim_strengthened", "temporal_shift"}
DELTA_LIKE = {"claim_weakened"}

POLICY = PreservationPolicy(
    scope=SCOPE,
    requirements=(
        Requirement("operator_fidelity", "mean_cosine_to_native", "min_value",
                    FIDELITY_BAR),
    ),
)


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _mirror_encoder():
    """Deterministic text-free encoder stand-in (seeded, documented)."""
    base_rng = np.random.default_rng(SEED)
    mixing = base_rng.normal(size=(N_LATENT, N_DIMS))
    weaken = np.zeros(N_LATENT)
    weaken[0] = 2.0

    def embed(case: ContentTransformationCase) -> tuple[np.ndarray, np.ndarray]:
        seed = int(hashlib.sha256(case.case_id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        latent = rng.normal(size=(N_LATENT,))
        shared = rng.normal(size=(N_DIMS,))
        shared *= 1.0 / np.linalg.norm(shared)

        def jitter() -> np.ndarray:
            vector = rng.normal(size=(N_DIMS,))
            return 0.15 * vector / np.linalg.norm(vector)

        first, second = jitter(), jitter()
        if case.relation in IDENTITY_LIKE:
            shifted = latent + rng.normal(0, 0.08, size=(N_LATENT,))
        elif case.relation in DELTA_LIKE:
            shifted = latent + weaken
        else:
            direction = rng.normal(size=(N_LATENT,))
            direction /= np.linalg.norm(direction)
            shifted = latent + 3.0 * direction

        def encode(current: np.ndarray, extra: np.ndarray) -> np.ndarray:
            signal = current @ mixing
            signal *= 2.5 / np.linalg.norm(signal)
            vector = signal + shared + extra
            return vector / np.linalg.norm(vector)

        return encode(latent, first), encode(shifted, second)

    return embed


def _mean_cosine(first: np.ndarray, second: np.ndarray) -> float:
    first_n = first / np.linalg.norm(first, axis=1, keepdims=True).clip(min=1e-12)
    second_n = second / np.linalg.norm(second, axis=1, keepdims=True).clip(min=1e-12)
    return float(np.mean(np.sum(first_n * second_n, axis=1)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    corpus_dir = ROOT / "corpus"
    frozen = (corpus_dir / "doc" / "relate-doc-0.1.0" / "corpus_hash.txt"
              ).read_text().strip()
    if frozen != CORPUS_HASH:
        print(f"FROZEN DOC CORPUS HASH CHANGED: {frozen}")
        return 1
    by_type = load_doc_cases(corpus_dir)
    embed = _mirror_encoder()
    scorer = cosine_scorer()
    space = SpaceIdentity(model="synthetic-operator-mirror-v1", dimensions=N_DIMS)

    selections, summaries, profiles = {}, {}, {}
    for transformation in TRANSFORMATION_TYPES:
        train_cases, eval_cases = split_train_eval(by_type[transformation])
        train_hash, eval_hash = hash_case_set(train_cases), hash_case_set(eval_cases)
        train_source = np.array([embed(c)[0] for c in train_cases])
        train_target = np.array([embed(c)[1] for c in train_cases])
        eval_source = np.array([embed(c)[0] for c in eval_cases])
        eval_target = np.array([embed(c)[1] for c in eval_cases])
        train_diag = _mean_cosine(
            fit_linear_operator(train_source, train_target, relation=transformation,
                                source_space_hash=space.space_hash,
                                train_case_set_hash=train_hash).transform(train_source),
            train_target)
        operators = {
            "identity_map": identity_map(
                relation=transformation, source_space_hash=space.space_hash,
                train_case_set_hash=train_hash),
            "constant_delta": fit_constant_delta(
                train_source, train_target, relation=transformation,
                source_space_hash=space.space_hash, train_case_set_hash=train_hash),
            "linear": fit_linear_operator(
                train_source, train_target, relation=transformation,
                source_space_hash=space.space_hash, train_case_set_hash=train_hash),
            "affine": fit_affine_operator(
                train_source, train_target, relation=transformation,
                source_space_hash=space.space_hash, train_case_set_hash=train_hash),
        }
        from relate.evaluation import CorrespondenceSet

        eval_ids = [c.case_id for c in eval_cases]
        correspondence = CorrespondenceSet(
            ids=tuple(eval_ids),
            source_rows=tuple(range(len(eval_cases))),
            target_rows=tuple(range(len(eval_cases))),
        )
        native_cosine = _mean_cosine(eval_source, eval_target)
        fidelities, built = {}, {}
        for name, operator in operators.items():
            candidate = operator.transform(eval_source)
            fidelity = _mean_cosine(candidate, eval_target)
            fidelities[name] = fidelity
            profile = evaluate_transformation(
                transformation=operator, source_vectors=eval_source,
                reference_vectors=eval_target, correspondence=correspondence,
                reference_frame=ReferenceFrame.TARGET_NATIVE,
                candidate_space_hash=f"derived:{operator.transformation_id}",
                reference_space_hash=space.space_hash,
                policies=[POLICY],
            )
            results = list(profile.results) + [PreservationResult(
                capability="operator_fidelity", metric="mean_cosine_to_native",
                value=fidelity, reference_value=1.0,
                reference_frame=ReferenceFrame.TARGET_NATIVE,
                evidence_id=operator.transformation_id)]
            built[name] = build_preservation_profile(
                source_space_hash=space.space_hash,
                candidate_space_hash=f"derived:{operator.transformation_id}",
                target_space_hash=space.space_hash,
                results=results, policies=[POLICY],
                bridge_id=operator.transformation_id,
                evaluation_correspondence_hash=eval_hash,
                scorer=scorer_id_of(scorer),
            )
            profiles[f"{transformation}/{name}"] = built[name]
        selection = select_simplest_passing(
            [(OPERATOR_COMPLEXITY[name], f"{transformation}/{name}", built[name])
             for name in operators],
            scope=SCOPE, policy_hash=POLICY.policy_id)
        selections[transformation] = {
            "outcome": selection.outcome.value,
            "selected": selection.selected_operator_id,
            "evaluated": list(selection.evaluated_operator_ids),
            "policy": selection.policy_hash,
        }
        summaries[transformation] = {
            "n_train": len(train_cases),
            "n_eval": len(eval_cases),
            "train_case_set_hash": train_hash,
            "eval_case_set_hash": eval_hash,
            "native_cosine": native_cosine,
            "fidelities": fidelities,
            "train_linear_diagnostic": train_diag,
            "selection": selections[transformation]["selected"],
            "outcome": selections[transformation]["outcome"],
        }

    # Regression gates: the earned taxonomy, not a full sweep.
    identity_wins = [t for t in IDENTITY_LIKE
                     if summaries[t]["outcome"] == "selected"
                     and summaries[t]["selection"].endswith("/identity_map")]
    if len(identity_wins) != 5:
        print(f"IDENTITY CLASSES NOT ALL IDENTITY: {identity_wins}")
        return 1
    if not (summaries["claim_weakened"]["outcome"] == "selected"
            and summaries["claim_weakened"]["selection"].endswith("/constant_delta")):
        print("WEAKENED CLAIM NOT CONSTANT_DELTA")
        return 1
    nones = [t for t in ("formal_to_informal", "verbose_to_concise", "statement_to_negation")
             if summaries[t]["outcome"] == OperatorSelectionOutcome.NONE_PASS.value]
    if len(nones) != 3:
        print(f"NONE CLASSES NOT ALL NONE_PASS: {nones}")
        return 1
    negation_native = summaries["statement_to_negation"]["native_cosine"]
    if not (negation_native > 0.6
            and summaries["statement_to_negation"]["outcome"] == "none_pass"):
        print("HIGH-COSINE NEGATION WITHOUT NONE_PASS NOT SHOWN")
        return 1

    try:
        code_version = importlib.metadata.version("relate-search")
    except importlib.metadata.PackageNotFoundError:
        code_version = "unknown"
    artifacts = {
        "summary.json": summaries,
        "selections.json": selections,
        **{f"profiles/{leaf}.json": asdict(profile) for leaf, profile in profiles.items()},
        "provenance.json": {
            "benchmark_id": BENCHMARK_ID,
            "seed": SEED,
            "dimensions": N_DIMS,
            "fidelity_bar": FIDELITY_BAR,
            "scope": SCOPE,
            "policy": asdict(POLICY),
            "corpus_release": "relate-doc-0.1.0",
            "corpus_hash": CORPUS_HASH,
            "space": space.to_dict(),
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
    for leaf, payload in artifacts.items():
        path = out_dir / leaf
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump(payload))
    for transformation in TRANSFORMATION_TYPES:
        row = summaries[transformation]
        print(f"  {transformation:22s} native={row['native_cosine']:.3f} "
              f"-> {row['selection'].split('/')[-1] if row['selection'] else 'NONE_PASS'}")
    print(f"wrote {len(artifacts)} artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
