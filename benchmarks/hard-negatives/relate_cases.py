"""RELATE corpus adapter: frozen task views feed the generic evaluator.

Stdlib only. Dependency direction: benchmarks -> src/relate/evaluation and
benchmarks -> corpus/*. Nothing corpus-specific enters ``src/relate``.

Each view triple ``(query, grade-3 positive, hard negative)`` becomes one
:class:`HardNegativeCase`: the query anchor should prefer its correct answer
over the deceptive negative. ``relation`` is the adapter's generic answer
label; the corpus's typed failure mechanism travels in
``negative_relation`` so grouped reports expose *which* distinction failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from relate.evaluation import HardNegativeCase  # noqa: E402

ANSWER_RELATION = "answers_query"


def load_relate_cases(
    corpus_dir: str | Path,
    release: str = "relate-0.1.0",
    view: str = "hard-negative-v0.1.json",
) -> list[HardNegativeCase]:
    """Load cases from a frozen RELATE task view plus item domains."""
    root = Path(corpus_dir) / release
    triples = json.loads((root / "views" / view).read_text())["triples"]
    domains: dict[str, str] = {}
    with open(root / "items.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                domains[record["id"]] = record["domain"]
    cases = []
    for triple in triples:
        query_id = triple["query_id"]
        positive_id = triple["positive_id"]
        negative_id = triple["hard_negative_id"]
        cases.append(
            HardNegativeCase(
                case_id=f"{query_id}:{positive_id}:{negative_id}",
                anchor_id=query_id,
                positive_id=positive_id,
                negative_id=negative_id,
                relation=ANSWER_RELATION,
                negative_relation=triple["underlying_relation"],
                group=domains.get(positive_id),
                metadata={
                    "method": triple["method"],
                    "corpus_release": release,
                    "view": view,
                    "query_id": query_id,
                },
            )
        )
    return cases


def summarize(cases: list[HardNegativeCase]) -> dict:
    """Counts the benchmark records for provenance (no vectors needed)."""
    from collections import Counter

    return {
        "total": len(cases),
        "by_negative_relation": dict(
            Counter(c.negative_relation for c in cases)
        ),
        "by_method": dict(Counter(c.metadata["method"] for c in cases)),
        "by_group": dict(Counter(c.group for c in cases)),
    }
