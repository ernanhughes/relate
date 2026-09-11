# RELATE corpus — canonical source

> **Canonical owner: the RELATE project** (`github.com/ernanhughes/relate`,
> this directory). The historical copy under
> `next-books/experiments/embeddings-from-first-principles/relate/` was used
> for *Embeddings From First Principles* Waves 1–6 and is now a consumer of
> tagged RELATE releases. Migration rule: this tree reproduces the frozen
> corpora byte-identically; it never regenerates a replacement. Provenance:
> `PROVENANCE.md`.
>
> Corpus generation lives here, **not** in `src/relate` — the runtime stays a
> lightweight NumPy-only library; this directory is the diagnostic instrument;
> `benchmarks/` are the experimental consumers.

# RELATE — the continuous experimental artifact for *Embeddings From First Principles*

Spec: `spec/RELATE-v0.1.md` (in this directory — the executable experiment owns it)
Book/evidence map: `../evidence/BOOK-EVIDENCE-MAP.md`
Public book: https://programmer.ie/books/embeddings-from-first-principles/

RELATE is a **diagnostic instrument**, not a general benchmark: ~1,200 short
text items with typed pairwise relations, built so that a failure localizes to a
capability (polarity, role, time, quantity, claim strength). Every book
experiment runs against a frozen, content-addressed version of it.

## Build sequence (spec §15 + staged plan) — `relate-0.1.0` FROZEN 2026-09-08

| Stage | Status | Artifact |
|---|---|---|
| 1. Freeze ontology + schema as machine-readable constants | done | `ontology.json`, `schema.py` |
| 2. Validators before generators | done | `validate.py`, `tests/test_validators.py` (16 checks) |
| 3. Tiny RELATE-DEV, every pair inspected, ontology defects found | done | `dev/dev_source.py` → `dev/corpus/`, `dev/DEV_FINDINGS.md` |
| 4. Template / entity families | done | `sources.py` (250 entity families, 63 template families) |
| 5. Overgenerate candidate items + pairs | done | `generate.py` |
| 6. Balance by difficulty (overlap strata, query style, hard-neg subtype) | done | `build_full.py::_balance_bands` (every balanced relation ≥25% per overlap band) |
| 7. Adjudicate + leakage / dedup / contamination checks | done | `build_full.py` (rule adjudication vs `ontology.json`; shingle-Jaccard dedup; template-generation = contamination defense) |
| 8. Build deterministic task views | done | `relate-0.1.0/views/{retrieval,relation,hard-negative,calibration,dimensionality,alignment}-v0.1.json` |
| 9. Freeze `relate-0.1.0` (content hash + manifest + datasheet + validation report) | done | `relate-0.1.0/` (`corpus_hash.txt`, `manifest.json`, `datasheet.md`) |

`corpus_hash` of `relate-0.1.0`: see `relate-0.1.0/corpus_hash.txt`. Every
experiment records it (parallel to the book's `space_hash`).

**RELATE-DOC v0.1** (spec §11 — structured documents + controlled compressions +
typed transformations for Ch 22–23) is **BUILT AND FROZEN**: `doc/`,
`doc/relate-doc-0.1.0/`, `doc/relate-doc-0.1.0/corpus_hash.txt`. 45 documents, 595
compressions (Family A: 8-way faithfulness corruptions + method compressions),
260 transformation pairs (Family B: 9 typed transformations). It gates Wave 4
(`../wave4/`) and the Transformation Wave (`../wave5/`).

## Layout

```
ontology.json          frozen relation defs, adjudication order, domains, grades, overlap bands, id patterns
schema.py              Item / Pair / Query record dataclasses + structural .check()
lexical.py             lexical / entity overlap features (light-stemmer approximation)
validate.py            semantic validators + corpus coverage report
sources.py             entity families + structured facts (stage 4)
generate.py            templated item + pair + query-seed generator (stages 5-6)
build_full.py          orchestrator: validate -> balance -> dedup/leakage -> splits -> BM25 -> views -> datasheet -> hash
build_dev.py           the tiny DEV build (unchanged)
dev/                   RELATE-DEV + its findings
spec/                  RELATE-v0.1.md, RELATE-CHANGELOG.md  (moved out of Hugo content: the experiment owns its spec)
relate-0.1.0/          THE FROZEN RELEASE — items/pairs/queries JSONL, splits/, views/, bm25_scores.jsonl, datasheet.md, corpus_hash.txt, manifest.json
```

## Running

```bash
cd experiments/embeddings-from-first-principles/relate
python build_full.py           # (re)build the frozen release
python build_full.py --check   # rebuild to a temp dir and assert the corpus hash is unchanged
python validate.py relate-0.1.0 --report
python tests/test_validators.py
python build_dev.py            # the small DEV fixture the tests pin
```

Stdlib only. No embedding model is loaded in the build — that begins in
`../wave1/`.

## Invariants enforced at freeze

- Every item, pair, and query passes `schema.check()` (structure) **and**
  `validate.py` (relation constraints, directionality, referential integrity,
  adjudication sanity). The build aborts on any error; `relate-0.1.0` was frozen
  with **0 errors, 0 warnings**.
- A `negation` pair below 0.7 stemmed lexical overlap is auto-re-adjudicated to
  `contradiction` (spec §3.6); a `relation-swap` with no shared entity likewise
  (spec §3.11). Count of auto-adjustments is in the datasheet.
- For `paraphrase`, `entailment`, `contradiction`, `topic-related`, each of the
  low / mid / high lexical-overlap bands holds ≥25% of that relation's pairs
  (spec §5) — enforced by the stage-6 balancer.
- Pair, query, and item ids are content-addressed and deterministic; a rebuild
  reproduces `corpus_hash` exactly.
- Changing `ontology.json` is a MINOR version bump recorded in
  `spec/RELATE-CHANGELOG.md`.
