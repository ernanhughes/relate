"""Calibration benchmark: respectable discriminator, operationally poor.

    python benchmarks/calibration/run.py          # write expected/
    python benchmarks/calibration/run.py --check  # verify byte-identical

Seeded Gaussian preference-score regimes (higher = more positive; no
vectors anywhere -- the scores could come from any scorer, which is the
point):

- ordinary: well-separated negatives (random-unrelated regime);
- hard: overlapping negatives (adversarial hard-negative regime);
- domain-a / domain-b: same positives, shifted negatives (scope matters).

The hard regime must show the Wave-1 pattern: decent AUC alongside a
large ambiguity region at bounded error rates. The domain pair must show
the operating point moving with the distribution, and staleness must name
the changed dimensions.
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

from relate.evaluation import (  # noqa: E402
    CalibrationScope,
    HardNegativeCase,
    NegativeSetDescriptor,
    calibrate,
    calibration_card,
    content_hash_of_scores,
    evaluate_hard_negatives,
    extract_margin_distributions,
    scorer_id_of,
    cosine_scorer,
)
from relate.spaces import SpaceIdentity  # noqa: E402

BENCHMARK_ID = "calibration-v1"
SEED = 0
N = 1500
FAR_TARGET = 0.10
FRR_TARGET = 0.10


def _draw(rng: np.random.Generator, mean: float, std: float) -> list[float]:
    return [float(v) for v in rng.normal(mean, std, N)]


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rng = np.random.default_rng(SEED)

    ordinary_pos = _draw(rng, 2.0, 1.0)
    ordinary_neg = _draw(rng, 0.0, 1.0)
    hard_pos = _draw(rng, 1.0, 0.9)
    hard_neg = _draw(rng, 0.0, 1.0)
    domain_pos = _draw(rng, 1.0, 1.0)
    domain_neg_a = _draw(rng, 0.0, 1.0)
    domain_neg_b = _draw(rng, 0.25, 1.0)

    def descriptor(name: str, kind: str, scores: list[float]) -> NegativeSetDescriptor:
        return NegativeSetDescriptor(
            name=name,
            kind=kind,
            n=len(scores),
            content_hash=content_hash_of_scores(scores),
            metadata={"generator": "gaussian", "seed": SEED, "n": N},
        )

    ordinary_fit = calibrate(
        ordinary_pos,
        ordinary_neg,
        far_target=FAR_TARGET,
        frr_target=FRR_TARGET,
        negative_set=descriptor("random-unrelated", "gaussian-separated", ordinary_neg),
        positive_set="grade-3-positives",
        scope=CalibrationScope(task="duplicate_filter", domain="general"),
        scorer="synthetic/gaussian:ordinary",
        seed=SEED,
    )
    hard_fit = calibrate(
        hard_pos,
        hard_neg,
        far_target=FAR_TARGET,
        frr_target=FRR_TARGET,
        negative_set=descriptor("adversarial-hard", "gaussian-overlap", hard_neg),
        positive_set="grade-3-positives",
        scope=CalibrationScope(task="duplicate_filter", domain="general"),
        scorer="synthetic/gaussian:hard",
        seed=SEED,
    )
    domain_fits = {
        name: calibrate(
            domain_pos,
            negatives,
            far_target=FAR_TARGET,
            frr_target=FRR_TARGET,
            negative_set=descriptor(f"negatives-{name}", "gaussian", negatives),
            positive_set="grade-3-positives",
            scope=CalibrationScope(task="duplicate_filter", domain=f"syn-{name}"),
            scorer=f"synthetic/gaussian:{name}",
            seed=SEED,
        )
        for name, negatives in (("a", domain_neg_a), ("b", domain_neg_b))
    }

    # The 3A seam: real hard-negative observations become score distributions.
    seam_vectors = {
        "a": np.array([1.0, 0.0]),
        "p": np.array([1.0, 0.1]),
        "n": np.array([0.0, 1.0]),
    }
    seam_cases = [
        HardNegativeCase(
            case_id=f"seam-{i}", anchor_id="a", positive_id="p", negative_id="n",
            relation="r",
        )
        for i in range(12)
    ]
    seam_report = evaluate_hard_negatives(seam_cases, seam_vectors, cosine_scorer())
    seam_paired = extract_margin_distributions(seam_report)
    seam_fit = calibrate(list(seam_paired.positive), list(seam_paired.negative))

    # Regression gates: the Wave-1 pattern, not sacred constants.
    hard_point = hard_fit.operating_point
    ordinary_point = ordinary_fit.operating_point
    if not (0.72 <= hard_fit.auc <= 0.82):
        print(f"HARD AUC OUT OF PATTERN: {hard_fit.auc}")
        return 1
    if not (0.24 <= hard_fit.eer_rate <= 0.34):
        print(f"HARD EER OUT OF PATTERN: {hard_fit.eer_rate}")
        return 1
    if not hard_point.ambiguity_fraction > 0.4:
        print("HARD AMBIGUITY NOT LARGE")
        return 1
    if not ordinary_fit.auc > 0.9 or not ordinary_point.ambiguity_fraction < 0.2:
        print("ORDINARY REGIME NOT CLEAN")
        return 1
    if not hard_point.ambiguity_fraction > 3 * ordinary_point.ambiguity_fraction:
        print("AMBIGUITY CONTRAST TOO SMALL")
        return 1
    eer_shift = (
        domain_fits["b"].eer_threshold - domain_fits["a"].eer_threshold
    )
    if not 0.05 <= eer_shift <= 0.18:
        print(f"DOMAIN SHIFT OUT OF PATTERN: {eer_shift}")
        return 1
    staleness = hard_fit.record.staleness_against(
        space_hash="other",
        scorer="other-scorer",
        domain="other-domain",
    )
    if staleness.changed != ("space_hash", "domain", "scorer"):
        print(f"STALENESS DIMENSIONS WRONG: {staleness.changed}")
        return 1

    space = SpaceIdentity(
        model="synthetic-score-distributions-v1",
        dimensions=1,
        revision=f"seed={SEED}",
        dtype="float64",
        normalize=False,
    )
    card = calibration_card(
        hard_fit,
        evaluation_id="duplicate-filter-hard-v1",
        corpus="synthetic-v1",
        corpus_hash="n/a-synthetic",
        space_hash=space.space_hash,
        scorer="synthetic/gaussian:hard",
        seed=SEED,
    )
    try:
        code_version = importlib.metadata.version("relate-search")
    except importlib.metadata.PackageNotFoundError:
        code_version = "unknown"
    provenance = {
        "benchmark_id": BENCHMARK_ID,
        "seed": SEED,
        "n_per_side": N,
        "far_target": FAR_TARGET,
        "frr_target": FRR_TARGET,
        "space": space.to_dict(),
        "scorers": ["synthetic/gaussian:ordinary", "synthetic/gaussian:hard"],
        "code_version": f"relate-search {code_version}",
    }
    summary = {
        "ordinary": {
            "auc": ordinary_fit.auc,
            "eer_rate": ordinary_fit.eer_rate,
            "ambiguity_fraction": ordinary_point.ambiguity_fraction,
        },
        "hard": {
            "auc": hard_fit.auc,
            "eer_rate": hard_fit.eer_rate,
            "eer_threshold": hard_fit.eer_threshold,
            "ambiguity_fraction": hard_point.ambiguity_fraction,
        },
        "domain_eer_shift": eer_shift,
        "seam_auc": seam_fit.auc,
        "staleness_changed": list(staleness.changed),
    }
    artifacts = {
        "summary.json": summary,
        "distributions-ordinary.json": {
            "positive": asdict(ordinary_fit.positive),
            "negative": asdict(ordinary_fit.negative),
        },
        "distributions-hard.json": {
            "positive": asdict(hard_fit.positive),
            "negative": asdict(hard_fit.negative),
        },
        "roc-hard.json": {
            "thresholds": list(hard_fit.thresholds),
            "fars": list(hard_fit.fars),
            "frrs": list(hard_fit.frrs),
        },
        "operating-points.json": {
            "ordinary": asdict(ordinary_point),
            "hard": asdict(hard_point),
            "domain_a": asdict(domain_fits["a"].operating_point),
            "domain_b": asdict(domain_fits["b"].operating_point),
        },
        "evaluation-card-hard.json": asdict(card),
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
    print(
        f"  ordinary AUC={ordinary_fit.auc:.3f} amb={ordinary_point.ambiguity_fraction:.3f} | "
        f"hard AUC={hard_fit.auc:.3f} EER={hard_fit.eer_rate:.3f} amb={hard_point.ambiguity_fraction:.3f} | "
        f"domain shift={eer_shift:+.3f}"
    )
    print(f"wrote {len(artifacts)} artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
