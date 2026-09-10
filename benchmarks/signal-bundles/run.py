"""Signal-bundle ablation: score-only vs geometric bundle vs external.

    python benchmarks/signal-bundles/run.py          # write expected/
    python benchmarks/signal-bundles/run.py --check  # verify byte-identical

Every test decision flows through production APIs: 3A
``evaluate_hard_negatives`` for margins, 3C ``calibrate`` for the
calibration record, ``Observatory.inspect_result`` for bundle
composition. The ablation combiner (train-tuned logistic probe over
bundle fields) is benchmark methodology, documented here -- not runtime:
bundles never classify, fit, or decide.

Expected pattern (mirror, not reproduction): score-only struggles on the
hard mix, the geometric bundle materially improves, and a simulated
generic verifier -- right where geometry is confident, coin-flip where
it is hard -- adds no reliable improvement and is reported separately.
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

from signals_fixture import N_TRAIN_ANCHORS, build_fixture  # noqa: E402
from relate import Observatory  # noqa: E402
from relate.evaluation import (  # noqa: E402
    calibrate,
    cosine_scorer,
    evaluate_hard_negatives,
    scorer_id_of,
)
from relate.spaces import SpaceIdentity  # noqa: E402

BENCHMARK_ID = "signal-bundles-v1"
K = 10
TIE_TOL = 1e-9
FAR_TARGET = 0.10
FRR_TARGET = 0.10
FEATURES = ["score", "margin", "density", "hubness", "stability"]
CONFIGS = {
    "score": ["score"],
    "score+margin": ["score", "margin"],
    "+density": ["score", "margin", "density"],
    "+hubness": ["score", "margin", "density", "hubness"],
    "all-geometric": FEATURES,
}


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _balanced_accuracy(predicted: np.ndarray, truth: np.ndarray) -> float:
    predicted = np.asarray(predicted)
    truth = np.asarray(truth)
    hit_pos = float(np.mean(predicted[truth == 1] == 1)) if (truth == 1).any() else 0.0
    hit_neg = float(np.mean(predicted[truth == 0] == 0)) if (truth == 0).any() else 0.0
    return float((hit_pos + hit_neg) / 2.0)


def _fit_probe(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Train-tuned linear probe (benchmark methodology, not runtime)."""
    design = np.hstack([features, np.ones((len(features), 1))])
    weights = np.zeros(design.shape[1])
    signed = 2 * labels - 1
    for _ in range(3000):
        margin = signed * (design @ weights)
        gradient = -(signed[:, None] * design) * (1 / (1 + np.exp(margin)))[:, None]
        weights -= 0.5 * gradient.mean(axis=0)
    return weights


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    fixture = build_fixture()
    vectors, ids = fixture["vectors"], fixture["ids"]
    vector_map = {name: vectors[i] for i, name in enumerate(ids)}
    scorer = cosine_scorer()
    observations = {}
    for cases in (fixture["hard_cases"], fixture["easy_cases"]):
        for obs in evaluate_hard_negatives(cases, vector_map, scorer, tie_tol=TIE_TOL).observations:
            observations[obs.case_id] = obs

    runtime = Observatory()
    space = runtime.register_space(
        model="synthetic-signals-v1", dimensions=vectors.shape[1]
    )

    # Calibration comes from real 3C measurement on train scores only.
    train_scores_pos, train_scores_neg = [], []
    for block in fixture["blocks"][:N_TRAIN_ANCHORS]:
        for instance in block:
            score = float(
                vectors[instance["anchor"]] @ vectors[instance["candidate"]]
            )
            (train_scores_pos if instance["label"] == 1 else train_scores_neg).append(score)
    calibration = calibrate(train_scores_pos, train_scores_neg,
                            far_target=FAR_TARGET, frr_target=FRR_TARGET)

    train_rows, test_rows, test_bundles, completeness = [], [], [], {"full": 0, "partial": 0}
    for block_index, block in enumerate(fixture["blocks"]):
        for instance in block:
            anchor, candidate = instance["anchor"], instance["candidate"]
            observation = observations[instance["case_id"]]
            # Every 20th test bundle omits calibration: partial but valid.
            partial = block_index >= N_TRAIN_ANCHORS and block_index % 20 == 0
            bundle = runtime.inspect_result(
                query_vector=vectors[anchor],
                candidate_vector=vectors[candidate],
                context_vectors=vectors,
                candidate_index=candidate,
                k=K,
                scorer=scorer,
                scorer_id=scorer_id_of(scorer),
                space_hash=space.space_hash,
                observation=observation,
                calibration=None if partial else calibration.record,
                calibration_id="" if partial else "signals-train-calibration-v1",
            )
            assert bundle.margin == observation.margin
            # Presentation sign is benchmark framing, not bundle evidence:
            # the bundle carries the case margin; the decision task signs it.
            row = [bundle.score, instance["sign"] * bundle.margin, bundle.local_density,
                   bundle.hubness, bundle.neighborhood_stability]
            if block_index < N_TRAIN_ANCHORS:
                train_rows.append((row, instance["label"]))
            else:
                test_rows.append((row, instance["label"]))
                test_bundles.append(bundle)
                completeness["full" if not partial else "partial"] += 1

    train_features = np.array([row for row, _ in train_rows])
    train_labels = np.array([label for _, label in train_rows])
    test_features = np.array([row for row, _ in test_rows])
    test_labels = np.array([label for _, label in test_rows])
    means, stds = train_features.mean(axis=0), train_features.std(axis=0) + 1e-9
    train_scaled, test_scaled = (train_features - means) / stds, (test_features - means) / stds

    ablation = {}
    for name, fields in CONFIGS.items():
        columns = [FEATURES.index(field) for field in fields]
        weights = _fit_probe(train_scaled[:, columns], train_labels)
        predicted = (
            np.hstack([test_scaled[:, columns], np.ones((len(test_scaled), 1))])
            @ weights >= 0
        ).astype(int)
        ablation[name] = {
            "test_balanced_accuracy": _balanced_accuracy(predicted, test_labels),
            "train_balanced_accuracy": _balanced_accuracy(
                (np.hstack([train_scaled[:, columns], np.ones((len(train_scaled), 1))])
                 @ weights >= 0).astype(int), train_labels),
        }

    # External verifier: simulated generic model, right where geometry is
    # confident, coin-flip where it is hard. Stacked the book's way: the
    # verdict joins the probe as one more feature, so redundancy (not harm)
    # is what a no-improvement result looks like.
    median_margin = float(np.median(np.abs(train_features[:, 1])))
    verifier_rng = np.random.default_rng(2)

    def verdicts_for(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        correct = np.array([
            verifier_rng.random() < (0.95 if abs(row[1]) > median_margin else 0.55)
            for row in features
        ])
        return np.where(correct, labels, 1 - labels)

    train_verdict = verdicts_for(train_features, train_labels)
    test_verdict = verdicts_for(test_features, test_labels)
    external_accuracy = _balanced_accuracy(test_verdict, test_labels)
    stacked_train = np.hstack([train_scaled, train_verdict[:, None]])
    stacked_test = np.hstack([test_scaled, test_verdict[:, None]])
    means6, stds6 = stacked_train.mean(axis=0), stacked_train.std(axis=0) + 1e-9
    stacked_weights = _fit_probe((stacked_train - means6) / stds6, train_labels)
    stacked = (
        np.hstack([(stacked_test - means6) / stds6, np.ones((len(stacked_test), 1))])
        @ stacked_weights >= 0
    ).astype(int)
    stacked_accuracy = _balanced_accuracy(stacked, test_labels)

    score_only = ablation["score"]["test_balanced_accuracy"]
    geo = ablation["all-geometric"]["test_balanced_accuracy"]
    if not 0.60 <= score_only <= 0.80:
        print(f"SCORE-ONLY OUT OF PATTERN: {score_only}")
        return 1
    if not geo - score_only > 0.10:
        print("GEOMETRIC BUNDLE NOT MATERIALLY BETTER")
        return 1
    if not 0.85 <= geo <= 0.95:
        print(f"GEOMETRIC BUNDLE OUT OF PATTERN: {geo}")
        return 1
    if not abs(stacked_accuracy - geo) < 0.03:
        print("EXTERNAL VERIFIER CHANGED THE RESULT RELIABLY")
        return 1

    try:
        code_version = importlib.metadata.version("relate-search")
    except importlib.metadata.PackageNotFoundError:
        code_version = "unknown"
    exemplary = test_bundles[0]
    artifacts = {
        "summary.json": {
            "ablation": {name: row["test_balanced_accuracy"] for name, row in ablation.items()},
            "external_alone": external_accuracy,
            "stacked": stacked_accuracy,
            "completeness": completeness,
            "n_train": len(train_rows),
            "n_test": len(test_rows),
        },
        "ablation.json": ablation,
        "external.json": {
            "source": "simulated-generic-verifier-v1",
            "rule": "correct with p=0.95 where |margin| confident else p=0.55",
            "standalone_balanced_accuracy": external_accuracy,
            "stacked_balanced_accuracy": stacked_accuracy,
            "geometric_balanced_accuracy": geo,
        },
        "calibration.json": {
            "auc": calibration.auc,
            "eer_rate": calibration.eer_rate,
            "ambiguity_fraction": calibration.operating_point.ambiguity_fraction,
            "evaluation_id": "signals-train-calibration-v1",
        },
        "bundle-example.json": asdict(exemplary),
        "provenance.json": {
            "benchmark_id": BENCHMARK_ID,
            "seed": fixture["seed"],
            "k": K,
            "tie_tol": TIE_TOL,
            "space": space.to_dict(),
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
    for name, row in ablation.items():
        print(f"  {name:15s} test={row['test_balanced_accuracy']:.4f}")
    print(f"  external-alone  test={external_accuracy:.4f} stacked={stacked_accuracy:.4f}")
    print(f"wrote {len(artifacts)} artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
