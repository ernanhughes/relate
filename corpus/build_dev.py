"""Build RELATE-DEV from dev/dev_source.py, then validate it.

  python build_dev.py            # build + validate + report
  python build_dev.py --no-validate

Emits (deterministic, content-addressed pair/query ids):
  dev/corpus/items.jsonl
  dev/corpus/pairs.jsonl
  dev/corpus/queries.jsonl
  dev/corpus/build_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "dev"))

import dev_source  # noqa: E402  (dev/ is on sys.path)
import lexical  # noqa: E402
from schema import Item, Pair, Query, dump_jsonl  # noqa: E402
from validate import validate_dir  # noqa: E402

OUT = HERE / "dev" / "corpus"


def _pid(a: str, b: str, rel: str) -> str:
    return "p-" + hashlib.sha1(f"{a}|{b}|{rel}".encode()).hexdigest()[:8]


def _qid(text: str, target: str) -> str:
    return "q-" + hashlib.sha1(f"{text}|{target}".encode()).hexdigest()[:8]


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    items_by_id: dict[str, dict] = {}

    # --- items: structural check + passthrough
    item_objs = []
    for raw in dev_source.ITEMS:
        raw = dict(raw)
        raw.setdefault("time_index", None)
        raw.setdefault("atomic_claims", [])
        raw.setdefault("entities", [])
        it = Item.from_dict(raw)
        it.check()
        items_by_id[it.id] = raw
        item_objs.append(raw)
    dump_jsonl(item_objs, OUT / "items.jsonl")

    # --- pairs: compute overlap features, assign ids
    pair_objs = []
    for intent in dev_source.PAIRS:
        a, b = items_by_id.get(intent["a"]), items_by_id.get(intent["b"])
        if a is None or b is None:
            raise SystemExit(f"pair intent references unknown item: {intent}")
        rec = dict(
            id=_pid(intent["a"], intent["b"], intent["relation"]),
            a_id=intent["a"],
            b_id=intent["b"],
            relation=intent["relation"],
            rule_fired=intent["relation"],
            runner_up=intent.get("runner_up"),
            direction=intent.get("direction"),
            lexical_overlap=round(lexical.jaccard(a["text"], b["text"]), 4),
            char_3gram_overlap=round(lexical.dice_3gram(a["text"], b["text"]), 4),
            entity_overlap=lexical.entity_overlap(a.get("entities", []), b.get("entities", [])),
            notes=intent.get("notes", ""),
        )
        Pair.from_dict(rec).check()  # structural
        pair_objs.append(rec)
    dump_jsonl(pair_objs, OUT / "pairs.jsonl")

    # --- queries
    query_objs = []
    for q in dev_source.QUERIES:
        rec = dict(
            id=_qid(q["text"], q["target"]),
            text=q["text"],
            query_style=q["query_style"],
            target_item_id=q["target"],
            positives=[dict(item_id=i, grade=g) for i, g in q["positives"]],
            hard_negatives=[
                dict(item_id=i, underlying_relation=r, method=m) for i, r, m in q.get("hard_negatives", [])
            ],
        )
        Query.from_dict(rec).check()
        query_objs.append(rec)
    dump_jsonl(query_objs, OUT / "queries.jsonl")

    return {"items": len(item_objs), "pairs": len(pair_objs), "queries": len(query_objs)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)

    counts = build()
    print(f"built: {counts}")

    report = {"counts": counts}
    rc = 0
    if not args.no_validate:
        rep, coverage = validate_dir(OUT, do_report=True)
        for w in rep.warnings:
            print(f"WARN  {w}")
        for e in rep.errors:
            print(f"ERROR {e}")
        print(f"\n{len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
        report["validation"] = {
            "errors": rep.errors,
            "warnings": rep.warnings,
            "coverage": coverage,
        }
        rc = 0 if rep.ok else 1

    (OUT / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
