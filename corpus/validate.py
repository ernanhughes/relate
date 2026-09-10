"""RELATE v0.1 validators — run BEFORE any generator writes to the corpus.

A record cannot enter RELATE unless it passes here. Two layers:

  1. structural  (schema.py dataclass .check())
  2. semantic    (this module): relation constraints, directionality,
     referential integrity, adjudication sanity, corpus-level coverage.

Usage:
    python validate.py <dir>          # dir contains items.jsonl / pairs.jsonl / queries.jsonl
    python validate.py --report <dir> # also print the coverage/counts report

Exit code 0 iff every check passes.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from schema import (
    ONTOLOGY,
    Item,
    Pair,
    Query,
    SchemaError,
    load_jsonl,
)

_REL = {r["id"]: r for r in ONTOLOGY["relations"]}
_RANK = {r["id"]: r["adjudication_rank"] for r in ONTOLOGY["relations"]}
_BANDS = ONTOLOGY["lexical_overlap_bands"]
_OVERLAP_BALANCED = set(ONTOLOGY["overlap_balanced_relations"])
_MIN_BAND_FRAC = ONTOLOGY["overlap_band_min_fraction_per_band"]


class ValidationReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


def _band(x: float) -> str:
    for name, b in _BANDS.items():
        if b["min"] <= x < b["max"]:
            return name
    return "high"


# --------------------------------------------------------------------------- #
# semantic checks
# --------------------------------------------------------------------------- #
def check_pair_semantics(p: Pair, items: dict[str, Item], rep: ValidationReport) -> None:
    rel = _REL[p.relation]
    cons = rel.get("constraints", {})
    a = items.get(p.a_id)
    b = items.get(p.b_id)

    if a is None or b is None:
        rep.err(f"{p.id}: pair references unknown item(s) {p.a_id!r}/{p.b_id!r}")
        return

    # directionality
    if rel["directional"] and p.direction != "a->b":
        rep.err(f"{p.id}: relation {p.relation!r} is directional; direction must be 'a->b' (got {p.direction!r})")
    if not rel["directional"] and p.direction is not None:
        rep.err(f"{p.id}: relation {p.relation!r} is symmetric; direction must be null (got {p.direction!r})")

    # per-relation numeric constraints
    if "lexical_overlap_min" in cons and p.lexical_overlap < cons["lexical_overlap_min"]:
        rep.err(
            f"{p.id}: relation {p.relation!r} requires lexical_overlap >= {cons['lexical_overlap_min']}, "
            f"got {p.lexical_overlap:.3f} ({cons.get('lexical_overlap_min_reason', '')})"
        )
    if "lexical_overlap_max" in cons and p.lexical_overlap > cons["lexical_overlap_max"]:
        rep.err(f"{p.id}: relation {p.relation!r} requires lexical_overlap <= {cons['lexical_overlap_max']}, got {p.lexical_overlap:.3f}")
    if "entity_overlap_min" in cons and p.entity_overlap < cons["entity_overlap_min"]:
        rep.err(f"{p.id}: relation {p.relation!r} requires entity_overlap >= {cons['entity_overlap_min']}, got {p.entity_overlap}")
    if "entity_overlap_max" in cons and p.entity_overlap > cons["entity_overlap_max"]:
        rep.err(f"{p.id}: relation {p.relation!r} requires entity_overlap <= {cons['entity_overlap_max']}, got {p.entity_overlap}")

    # temporal-mismatch: both items need a time index
    if cons.get("both_items_require_time_index"):
        if a.time_index is None or b.time_index is None:
            rep.err(f"{p.id}: temporal-mismatch requires time_index on BOTH items ({p.a_id}={a.time_index}, {p.b_id}={b.time_index})")
        elif a.time_index == b.time_index:
            rep.err(f"{p.id}: temporal-mismatch has identical time_index {a.time_index!r} on both items")

    # partial-support: a must carry >=2 atomic claims
    if cons.get("a_requires_atomic_claims"):
        need = cons.get("a_min_atomic_claims", 2)
        if len(a.atomic_claims) < need:
            rep.err(f"{p.id}: partial-support requires a ({p.a_id}) to have >= {need} atomic_claims, has {len(a.atomic_claims)}")

    # relation-swap must be between items sharing entities
    if cons.get("requires_asymmetric_relation") and p.entity_overlap < 1:
        rep.err(f"{p.id}: relation-swap requires shared entities (entity_overlap >= 1), got {p.entity_overlap}")

    # adjudication sanity: runner_up recorded for everything except 'unrelated'
    if p.relation != "unrelated" and p.runner_up is None:
        rep.warn(f"{p.id}: no runner_up recorded for relation {p.relation!r} (spec 3.9: record the next-best relation)")

    # cheap sanity: 'equivalent' pairs should not be perfectly lexically identical unless truly reworded
    if p.relation == "equivalent" and a.text.strip().lower() == b.text.strip().lower():
        rep.err(f"{p.id}: 'equivalent' pair has byte-identical text on both sides (that is a duplicate, not a relation)")

    # entities declared on items should back the claimed entity_overlap
    shared = set(map(str.lower, a.entities)) & set(map(str.lower, b.entities))
    if p.entity_overlap != len(shared):
        rep.warn(
            f"{p.id}: entity_overlap={p.entity_overlap} but items declare {len(shared)} shared entities "
            f"({sorted(shared)}); recompute or fix item.entities"
        )


def check_query_semantics(q: Query, items: dict[str, Item], rep: ValidationReport) -> None:
    if q.target_item_id not in items:
        rep.err(f"{q.id}: target_item_id {q.target_item_id!r} not an item")
    for p in q.positives:
        if p.item_id not in items:
            rep.err(f"{q.id}: positive {p.item_id!r} not an item")
    for hn in q.hard_negatives:
        if hn.item_id not in items:
            rep.err(f"{q.id}: hard-negative {hn.item_id!r} not an item")
        if hn.item_id in {p.item_id for p in q.positives}:
            rep.err(f"{q.id}: {hn.item_id!r} is listed as both a positive and a hard negative")


# --------------------------------------------------------------------------- #
# corpus-level
# --------------------------------------------------------------------------- #
def coverage_report(items: dict[str, Item], pairs: list[Pair], queries: list[Query], rep: ValidationReport) -> dict:
    rel_counts = Counter(p.relation for p in pairs)
    domain_counts = Counter(i.domain for i in items.values())
    prov_counts = Counter(i.provenance.kind for i in items.values())
    tf_counts = Counter(i.template_family for i in items.values())
    ef_counts = Counter(i.entity_family for i in items.values())

    # overlap-band coverage for the balanced relations
    band_by_rel: dict[str, Counter] = defaultdict(Counter)
    for p in pairs:
        if p.relation in _OVERLAP_BALANCED:
            band_by_rel[p.relation][_band(p.lexical_overlap)] += 1
    for rel, bands in band_by_rel.items():
        total = sum(bands.values())
        for band in _BANDS:
            frac = bands.get(band, 0) / total if total else 0.0
            if frac < _MIN_BAND_FRAC:
                rep.warn(
                    f"overlap strata: relation {rel!r} has {frac:.0%} in the {band!r} band "
                    f"(spec 5 target >= {_MIN_BAND_FRAC:.0%}); DEV-stage warning, must hold at freeze"
                )

    # every relation should appear at least once at DEV stage
    for rel in _REL:
        if rel_counts.get(rel, 0) == 0:
            rep.warn(f"relation {rel!r} has zero pairs (DEV stage: add at least a few)")

    return {
        "items": len(items),
        "pairs": len(pairs),
        "queries": len(queries),
        "relations": dict(sorted(rel_counts.items())),
        "domains": dict(sorted(domain_counts.items())),
        "provenance": dict(sorted(prov_counts.items())),
        "template_families": len(tf_counts),
        "entity_families": len(ef_counts),
        "overlap_bands_balanced_relations": {r: dict(b) for r, b in band_by_rel.items()},
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def validate_dir(path: str | Path, do_report: bool = False) -> tuple[ValidationReport, dict]:
    d = Path(path)
    rep = ValidationReport()
    items: dict[str, Item] = {}
    pairs: list[Pair] = []
    queries: list[Query] = []

    # --- items
    for raw in load_jsonl(d / "items.jsonl"):
        try:
            it = Item.from_dict(raw)
            it.check()
        except SchemaError as e:
            rep.err(f"item: {e}")
            continue
        if it.id in items:
            rep.err(f"duplicate item id {it.id!r}")
        items[it.id] = it

    # --- pairs
    seen_pair_keys: set[tuple[str, str, str]] = set()
    for raw in load_jsonl(d / "pairs.jsonl"):
        try:
            pr = Pair.from_dict(raw)
            pr.check()
        except SchemaError as e:
            rep.err(f"pair: {e}")
            continue
        key = (pr.a_id, pr.b_id, pr.relation)
        if key in seen_pair_keys:
            rep.err(f"{pr.id}: duplicate (a_id, b_id, relation) triple")
        seen_pair_keys.add(key)
        check_pair_semantics(pr, items, rep)
        pairs.append(pr)

    # --- queries
    qfile = d / "queries.jsonl"
    if qfile.exists():
        for raw in load_jsonl(qfile):
            try:
                q = Query.from_dict(raw)
                q.check()
            except SchemaError as e:
                rep.err(f"query: {e}")
                continue
            check_query_semantics(q, items, rep)
            queries.append(q)

    report = coverage_report(items, pairs, queries, rep) if (do_report or True) else {}
    return rep, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a RELATE corpus directory.")
    ap.add_argument("dir", help="directory with items.jsonl / pairs.jsonl / queries.jsonl")
    ap.add_argument("--report", action="store_true", help="print the coverage/counts report")
    args = ap.parse_args(argv)

    rep, report = validate_dir(args.dir, do_report=args.report)

    for w in rep.warnings:
        print(f"WARN  {w}")
    for e in rep.errors:
        print(f"ERROR {e}")

    if args.report:
        import json as _j
        print("\n--- coverage report ---")
        print(_j.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
