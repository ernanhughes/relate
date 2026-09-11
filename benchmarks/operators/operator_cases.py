"""RELATE-DOC transformation cases: real pairs, hashed content, split sets.

Loads the frozen ``transformation_pairs.jsonl`` (260 pairs, nine exact
transformation classes), binds every pair as a ``ContentTransformationCase``
with sha256 content hashes of the real source/target texts, and splits
each class into deterministic train/eval case sets with distinct hashes.
No text is rewritten and nothing is embedded here -- geometry enters
downstream through the mirror encoder in ``run.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from relate.transformations import (
    ContentTransformationCase,
    hash_case_set,
    hash_content,
)

TRANSFORMATION_TYPES = (
    "active_to_passive",
    "claim_strengthened",
    "claim_weakened",
    "formal_to_informal",
    "present_to_past",
    "relation_swap",
    "statement_to_negation",
    "temporal_shift",
    "verbose_to_concise",
)

CORPUS_HASH = "111cc2bb9008557d632061f98a0e4f847b91293e31bcdb2d78285a35de7f7914"


def load_doc_cases(corpus_dir: str | Path) -> dict[str, list[ContentTransformationCase]]:
    """Map transformation type -> content cases in frozen file order."""
    path = (Path(corpus_dir) / "doc" / "relate-doc-0.1.0"
            / "transformation_pairs.jsonl")
    by_type: dict[str, list[ContentTransformationCase]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            transformation = record["transformation"]
            case = ContentTransformationCase(
                case_id=record["id"],
                source_content_hash=hash_content(record["source"]),
                target_content_hash=hash_content(record["target"]),
                relation=transformation,
                source_id=record.get("base_id"),
                target_id=record["id"],
                group=record.get("domain"),
                metadata={
                    "template_family": record.get("template_family"),
                    "lexical_realisation": record.get("lexical_realisation"),
                    "corpus_release": "relate-doc-0.1.0",
                    "corpus_hash": CORPUS_HASH,
                },
            )
            by_type.setdefault(transformation, []).append(case)
    missing = [name for name in TRANSFORMATION_TYPES if name not in by_type]
    if missing:
        raise ValueError(f"frozen corpus lost transformation types: {missing}")
    return by_type


def split_train_eval(
    cases: list[ContentTransformationCase], train_fraction: float = 2 / 3,
) -> tuple[list[ContentTransformationCase], list[ContentTransformationCase]]:
    """Deterministic file-order split with separately hashed halves."""
    cut = max(2, int(len(cases) * train_fraction))
    train, eval_cases = cases[:cut], cases[cut:]
    if hash_case_set(train) == hash_case_set(eval_cases):
        raise ValueError("train and eval case sets collide")
    return train, eval_cases
