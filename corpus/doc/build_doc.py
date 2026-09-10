"""RELATE-DOC v0.1 full build — validate, split, datasheet, freeze.

  python build_doc.py            # build relate-doc-0.1.0/
  python build_doc.py --check    # rebuild to a temp dir, assert the hash

Output (parallels relate-0.1.0/):
  documents.jsonl                structured source docs + atomic-claim ledgers + queries
  compressions.jsonl            Family A: faithfulness corruptions + method compressions
  transformation_pairs.jsonl    Family B: 9 typed (source,target) transformations
  splits/{entity,template,domain,lexical}.json
  datasheet.md   corpus_hash.txt   manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import generate

HERE = Path(__file__).parent
OUT = HERE / "relate-doc-0.1.0"
VERSION = "0.1.0"
_WORD = re.compile(r"[A-Za-z0-9']+")

CORRUPTION_KINDS = {
    "faithful", "number_dropped", "number_changed", "relation_reversed",
    "negation_inserted", "minority_entity_dropped", "temporal_value_shifted", "conclusion_changed",
}
TRANSFORM_TYPES = {
    "verbose_to_concise", "formal_to_informal", "active_to_passive", "present_to_past",
    "statement_to_negation", "relation_swap", "temporal_shift", "claim_strengthened", "claim_weakened",
}


def _bucket(key: str, salt: str) -> str:
    x = int(hashlib.sha1(f"{salt}|{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if x < 0.6 else ("dev" if x < 0.8 else "test")


def validate(g) -> list[str]:
    errs = []
    doc_ids = {d["id"] for d in g["documents"]}
    for d in g["documents"]:
        cids = {c["id"] for c in d["atomic_claims"]}
        if len(cids) != len(d["atomic_claims"]):
            errs.append(f"{d['id']}: duplicate claim ids")
        if not d["primary_claims"]:
            errs.append(f"{d['id']}: no primary claims")
        for q in d["queries"]:
            if q["answer_claim"] not in cids:
                errs.append(f"{d['id']}: query answer_claim {q['answer_claim']} not in ledger")
        if d["key_relation"] and d["key_relation"]["claim"] not in cids:
            errs.append(f"{d['id']}: key_relation claim not in ledger")
        if d["conclusion_claim"] not in cids:
            errs.append(f"{d['id']}: conclusion_claim not in ledger")
        if d["word_count"] < 15:
            errs.append(f"{d['id']}: document too short ({d['word_count']} words)")
    seen_ctrl = defaultdict(int)
    for c in g["compressions"]:
        if c["document_id"] not in doc_ids:
            errs.append(f"{c['id']}: compression references unknown document")
        if c["is_control"]:
            seen_ctrl[c["document_id"]] += 1
    for did, n in seen_ctrl.items():
        if n != 1:
            errs.append(f"{did}: {n} control (faithful) compressions, expected 1")
    for did in doc_ids:
        if seen_ctrl.get(did, 0) == 0:
            errs.append(f"{did}: no faithful control compression")
    for t in g["transformation_pairs"]:
        if t["transformation"] not in TRANSFORM_TYPES:
            errs.append(f"{t['id']}: unknown transformation {t['transformation']}")
        if t["source"].strip() == t["target"].strip():
            errs.append(f"{t['id']}: source == target")
    return errs


def _splits(g):
    docs = g["documents"]
    out = {}
    for axis, keyfn in [
        ("entity", lambda d: d["entity_family"]),
        ("template", lambda d: d["template_family"]),
        ("domain", lambda d: d["domain"]),
    ]:
        keys = sorted({keyfn(d) for d in docs})
        assign = {k: _bucket(k, axis) for k in keys}
        out[axis] = dict(by_key={k: assign[k] for k in keys},
                         by_document={d["id"]: assign[keyfn(d)] for d in docs})
    # lexical split for transformation pairs
    lex = sorted({t["lexical_realisation"] for t in g["transformation_pairs"]})
    lassign = {k: _bucket(k, "lexical") for k in lex}
    out["lexical"] = dict(by_key=lassign,
                          by_transformation_pair={t["id"]: lassign[t["lexical_realisation"]] for t in g["transformation_pairs"]},
                          by_base={t["base_id"]: _bucket(t["base_id"], "trbase") for t in g["transformation_pairs"]})
    return out


def _corpus_hash(recs) -> str:
    h = hashlib.sha256()
    for group in recs:
        for r in sorted(group, key=lambda x: x["id"]):
            h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()); h.update(b"\n")
    return h.hexdigest()


def _write_jsonl(path, recs):
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in sorted(recs, key=lambda x: x["id"]):
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def build(out: Path) -> dict:
    g = generate.generate()
    errs = validate(g)
    if errs:
        raise SystemExit("RELATE-DOC VALIDATION FAILED:\n" + "\n".join(errs[:40]))

    out.mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(exist_ok=True)
    _write_jsonl(out / "documents.jsonl", g["documents"])
    _write_jsonl(out / "compressions.jsonl", g["compressions"])
    _write_jsonl(out / "transformation_pairs.jsonl", g["transformation_pairs"])
    splits = _splits(g)
    for axis, tbl in splits.items():
        (out / "splits" / f"{axis}.json").write_text(json.dumps(tbl, indent=2, sort_keys=True), encoding="utf-8")

    ch = _corpus_hash([g["documents"], g["compressions"], g["transformation_pairs"]])
    (out / "corpus_hash.txt").write_text(ch + "\n", encoding="utf-8")

    wc = [d["word_count"] for d in g["documents"]]
    stats = dict(
        documents=len(g["documents"]),
        by_domain=dict(Counter(d["domain"] for d in g["documents"])),
        word_count=dict(min=min(wc), mean=round(sum(wc) / len(wc)), max=max(wc)),
        compressions=len(g["compressions"]),
        compressions_by_kind=dict(Counter(c["kind"] for c in g["compressions"])),
        faithfulness_family=sum(1 for c in g["compressions"] if c["family"] == "A-faithfulness"),
        method_family=sum(1 for c in g["compressions"] if c["family"] == "A-method"),
        transformation_pairs=len(g["transformation_pairs"]),
        transformation_by_type=dict(Counter(t["transformation"] for t in g["transformation_pairs"])),
        reversible_transformation_pairs=sum(1 for t in g["transformation_pairs"] if t["is_reversible"]),
        entity_families=len({d["entity_family"] for d in g["documents"]}),
        template_families=len({d["template_family"] for d in g["documents"]}),
        total_atomic_claims=sum(len(d["atomic_claims"]) for d in g["documents"]),
        rare_entity_claims=sum(len(d["rare_claims"]) for d in g["documents"]),
    )
    (out / "datasheet.md").write_text(_datasheet(stats, ch), encoding="utf-8")
    manifest = dict(version=VERSION, corpus_hash=ch, stats=stats,
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return dict(stats=stats, corpus_hash=ch)


def _datasheet(s, ch) -> str:
    return f"""# RELATE-DOC v0.1 — Datasheet

Long-document extension of RELATE, for the compression / faithfulness chapter
(Ch 22) and the operator / transformation chapter (Ch 23). `corpus_hash`
(SHA-256 of the sorted document + compression + transformation records):
`{ch}`.

## Motivation

RELATE v0.1 is a *pair-level* probe. RELATE-DOC adds **structured source
documents** so the book can ask two document-level questions:

1. **Family A — faithfulness / compression (Ch 22).** For each document, a
   `faithful` control summary plus seven controlled corruptions, each targeting a
   *named atomic claim*: `number_dropped`, `number_changed`, `relation_reversed`,
   `negation_inserted`, `minority_entity_dropped`, `temporal_value_shifted`,
   `conclusion_changed`. Plus method compressions (`truncation` / `extractive` /
   `abstractive` at 50/25/10%). The Wave 4 question: **which corruption classes
   are invisible to whole-document embedding drift, neighbourhood preservation,
   query-conditioned preservation, and claim-conditioned preservation** — the
   blind-spot matrix. The adversarial cases (`relation_reversed`,
   `negation_inserted`, `number_changed`) deliberately keep the topic fixed so
   the coarse geometry *should* survive while one load-bearing fact is reversed.

2. **Family B — transformation (Ch 23).** 35 base sentences × 9 typed
   transformations (`verbose_to_concise`, `formal_to_informal`,
   `active_to_passive`, `present_to_past`, `statement_to_negation`,
   `relation_swap`, `temporal_shift`, `claim_strengthened`, `claim_weakened`),
   with content varied and the transformation type held fixed, split by entity /
   template / domain / lexical realisation so the operator bake-off cannot
   memorise a surface pattern.

## Composition

| Quantity | Count |
|---|---|
| Source documents | {s['documents']} |
| — by domain | {json.dumps(s['by_domain'])} |
| — word count (min / mean / max) | {s['word_count']['min']} / {s['word_count']['mean']} / {s['word_count']['max']} |
| Atomic claims (total) | {s['total_atomic_claims']} |
| — rare / minority-entity claims | {s['rare_entity_claims']} |
| Compressions (total) | {s['compressions']} |
| — Family A faithfulness corruptions | {s['faithfulness_family']} |
| — Family A method compressions | {s['method_family']} |
| Transformation pairs | {s['transformation_pairs']} |
| — reversible (for inverse-consistency) | {s['reversible_transformation_pairs']} |
| Entity families / template families | {s['entity_families']} / {s['template_families']} |

Compressions by kind: {json.dumps(s['compressions_by_kind'])}

Transformation pairs by type: {json.dumps(s['transformation_by_type'])}

## Collection process

Every document is assembled by `generate.py` from the structured scenarios in
`sources.py` (invented, or public fact restated in original wording). **No text
is drawn from an existing summarisation / faithfulness corpus** (CNN/DM, XSum,
FActScore, SummEval, …) — the templated pipeline is the contamination defence.
Each corruption is applied by a documented rule to a *named* target claim, so the
gold label ("which claim was corrupted, and how") is exact.

## Splits

`splits/{{entity,template,domain}}.json` for documents (60/20/80 by deterministic
hash), `splits/lexical.json` for transformation pairs (by verb realisation and by
base sentence). Operator fits in the Transformation Wave train on
`entity:train`/`template:train`/`domain:train` and report held-out transfer on
each.

## Known limitations

1. **Short documents.** v0.1 documents average {s['word_count']['mean']} words
   (spec §11 target 300–3,000). They are long enough to carry 4–10 atomic claims
   and a conclusion, which is what the blind-spot matrix needs, but a corruption
   has less room to hide than in a full-length article. A v0.2 would lengthen them.
2. **Synthetic.** Templated assembly gives exact corruption labels and zero
   public-corpus contamination, at the cost of surface regularity (the
   `template` split is the guard).
3. **No human faithfulness annotation** — the corruption target is known by
   construction; there is no human "is this summary faithful?" rating. Consistent
   with RELATE v0.1's labelling model.
4. **English only; Western-centric invented entities.**
5. **`minority_entity_dropped`** exists only for the {s['rare_entity_claims']}
   full scenarios that carry a rare entity; the short documents do not.

## License

Target CC BY 4.0. Content is expression-original.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            r = build(Path(td) / "relate-doc-0.1.0")
        frozen = (OUT / "corpus_hash.txt").read_text().strip() if (OUT / "corpus_hash.txt").exists() else "<none>"
        print(f"rebuilt {r['corpus_hash']}\nfrozen  {frozen}\n{'MATCH' if r['corpus_hash'] == frozen else 'DIFFERS'}")
        return 0 if r["corpus_hash"] == frozen else 1
    r = build(OUT)
    s = r["stats"]
    print(f"RELATE-DOC {VERSION} -> {OUT}")
    print(f"  {s['documents']} docs ({s['word_count']['mean']} avg words), {s['compressions']} compressions, {s['transformation_pairs']} transformation pairs")
    print(f"  corpus_hash {r['corpus_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
