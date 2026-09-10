"""Hard-negative benchmark runner: one evaluator, every scorer.

    python benchmarks/hard-negatives/run.py          # write expected/
    python benchmarks/hard-negatives/run.py --check  # verify byte-identical

Two halves. First, the RELATE adapter half: load the frozen task view into
generic cases and record the summary (structure/provenance validation; no
vectors needed, since embedding generation stays external to RELATE).
Second, the scoring half: a deterministic synthetic fixture mirroring the
Python-structure experiment runs cosine, Euclidean, random and the relation
readout through the SAME evaluator.

Provenance (space_hash, corpus hash/version, scorer identity, seed, code
version) is written with every artifact.
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

from relate.evaluation import (  # noqa: E402
    compare_reports,
    cosine_scorer,
    euclidean_scorer,
    evaluate_hard_negatives,
    hard_negative_card,
    random_scorer,
    relation_scorer,
    scorer_id_of,
)
from relate.spaces import SpaceIdentity  # noqa: E402
from relate_cases import load_relate_cases, summarize  # noqa: E402
from synthetic_vectors import build_fixture  # noqa: E402

BENCHMARK_ID = "hard-negatives-v1"
CORPUS_RELEASE = "relate-0.1.0"
CORPUS_HASH = "8cad6816d90e06bc49e5b0b64cd460945e17ea4b1ec3c46054409669eda525b3"
EXPECTED_CASES = 937
EXPECTED_NEGATIVE_RELATIONS = {
    "entity-related": 329,
    "topic-related": 266,
    "negation": 257,
    "relation-swap": 61,
    "temporal-mismatch": 24,
}
TIE_TOL = 1e-9
RANDOM_SEED = 7


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    corpus_dir = ROOT / "corpus"
    frozen = (corpus_dir / CORPUS_RELEASE / "corpus_hash.txt").read_text().strip()
    if frozen != CORPUS_HASH:
        print(f"FROZEN CORPUS HASH CHANGED: {frozen}")
        return 1

    # -- adapter half: RELATE cases feed the generic evaluator's input type --
    relate_cases = load_relate_cases(corpus_dir, release=CORPUS_RELEASE)
    summary = summarize(relate_cases)
    if summary["total"] != EXPECTED_CASES:
        print(f"ADAPTER CASE COUNT CHANGED: {summary['total']}")
        return 1
    if summary["by_negative_relation"] != EXPECTED_NEGATIVE_RELATIONS:
        print("ADAPTER RELATION DISTRIBUTION CHANGED")
        return 1

    # -- scoring half: synthetic mirror of the Python-structure experiment --
    fixture = build_fixture()
    vectors = fixture["vectors"]
    cases = fixture["cases"]
    scorers = {
        "relation": relation_scorer(fixture["projection"]),
        "cosine": cosine_scorer(),
        "euclidean": euclidean_scorer(),
        "random": random_scorer(seed=RANDOM_SEED),
    }
    reports = {
        name: evaluate_hard_negatives(cases, vectors, scorer, tie_tol=TIE_TOL)
        for name, scorer in scorers.items()
    }
    delta = compare_reports(reports["cosine"], reports["relation"])

    space = SpaceIdentity(
        model="synthetic-python-structure-mirror-v1",
        dimensions=32,
        revision="seed=0",
        dtype="float64",
        normalize=True,
    )
    card = hard_negative_card(
        reports["relation"],
        evaluation_id="python-structure-mirror-v1",
        space_hash=space.space_hash,
        corpus="synthetic-v1",
        corpus_hash="n/a-synthetic",
        scorer=scorers["relation"],
        seed=fixture["seed"],
    )

    try:
        code_version = importlib.metadata.version("relate-search")
    except importlib.metadata.PackageNotFoundError:
        code_version = "unknown"
    provenance = {
        "benchmark_id": BENCHMARK_ID,
        "space_hash": space.space_hash,
        "space": space.to_dict(),
        "adapter": {
            "corpus_release": CORPUS_RELEASE,
            "corpus_hash": CORPUS_HASH,
            "view": "hard-negative-v0.1.json",
            "cases": summary["total"],
        },
        "vectors_source": "synthetic-v1",
        "fixture_seed": fixture["seed"],
        "scorers": {name: scorer_id_of(s) for name, s in scorers.items()},
        "tie_tol": TIE_TOL,
        "random_seed": RANDOM_SEED,
        "code_version": f"relate-search {code_version}",
    }

    artifacts = {
        "relate-cases-summary.json": summary,
        **{f"report-{n}.json": asdict(r) for n, r in reports.items()},
        "delta-relation-vs-cosine.json": asdict(delta),
        "evaluation-card-relation.json": asdict(card),
        "provenance.json": provenance,
    }

    out_dir = HERE / "expected"
    if args.check:
        failures = 0
        for leaf, payload in artifacts.items():
            existing = (out_dir / leaf).read_text() if (out_dir / leaf).exists() else None
            if existing != _dump(payload):
                print(f"DIFFERS: {leaf}")
                failures += 1
        print("MATCH" if not failures else f"{failures} differing artifact(s)")
        return 1 if failures else 0

    out_dir.mkdir(exist_ok=True)
    for leaf, payload in artifacts.items():
        (out_dir / leaf).write_text(_dump(payload))
    order = {n: (r.accuracy, r.mean_margin) for n, r in reports.items()}
    for name, (accuracy, margin) in order.items():
        print(f"  {name:10s} accuracy={accuracy:.4f} mean_margin={margin:+.4f}")
    print(f"wrote {len(artifacts)} artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
