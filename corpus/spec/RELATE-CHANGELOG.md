# RELATE — changelog

Versioning: `relate-MAJOR.MINOR.PATCH`. The **built artifact** is content-addressed
(`corpus_hash`). Changing the relation set, split rules, or removing items is a
MINOR bump; additive changes with stable splits are PATCH.

The machine-readable ontology freeze lives at
`experiments/embeddings-from-first-principles/relate/ontology.json` and carries
its own `spec_version`.

---

## `relate-0.2.0` — 2026-09-08

**Frozen.** `corpus_hash` in `../relate-0.2.0/corpus_hash.txt`. **Additive, splits
stable → PATCH/MINOR by spec §12; released as v0.2 per the operator brief.**

- `items.jsonl` and `pairs.jsonl` are **byte-identical to `relate-0.1.0`**
  (verified `diff`). Only the query set changes.
- Adds **142 hard queries** (411 total): indirect phrasing with a real lexical
  gap to the answer, 5 competing distractors each — role disambiguation,
  claim-strength, temporal qualification, indirect reference. Built by
  `build_full.py --v02` (a `hard=True` flag on the qseed calls in `generate.py`;
  the hard qseeds are a separate list, so they cannot perturb the v0.1
  items/pairs/balancer — v0.1's hash is unchanged).
- **Motivation:** Wave 1 showed v0.1's near-restatement queries saturate the
  system-level rows. v0.2's mission is narrow — unsaturate retrieval-policy /
  reranker / model-comparison experiments — not to grow a general benchmark.
- Re-run of the saturated Wave-1 rows: `../../wave1/artifacts/v02/`.

## `relate-doc-0.1.0` — 2026-09-08

**Frozen.** `corpus_hash` in `../doc/relate-doc-0.1.0/corpus_hash.txt`. Long-document
extension for Ch 22 (compression / faithfulness) and Ch 23 (transformation / operators).

- 45 structured source documents (geo-civics place profiles, biomed study
  abstracts, corporate deal writeups), each with an atomic-claim ledger
  (primary / secondary / rare, typed fact / numeric / temporal / relational), a
  key relation with argument order, a conclusion claim, and graded queries.
- **Family A — faithfulness / compression** (595 records): per document, a
  `faithful` control + 7 rule-based corruptions each targeting a *named* claim
  (`number_dropped`, `number_changed`, `relation_reversed`, `negation_inserted`,
  `minority_entity_dropped`, `temporal_value_shifted`, `conclusion_changed`) +
  method compressions (`truncation` / `extractive` / `abstractive` at 50/25/10%).
- **Family B — transformation** (260 pairs): 35 base sentences + 10 two-entity
  bases × 9 typed transformations (`verbose_to_concise`, `formal_to_informal`,
  `active_to_passive`, `present_to_past`, `statement_to_negation`,
  `relation_swap`, `temporal_shift`, `claim_strengthened`, `claim_weakened`),
  content varied / transform fixed, split by entity / template / domain / lexical.
- Validates clean; deterministic rebuild reproduces the hash (`build_doc.py --check`).
- **Known deviation from spec §11:** documents average ~78 words (target
  300–3,000). Long enough for 4–10 claims + a conclusion, which is what the
  blind-spot matrix needs; a v0.2 lengthens them. Datasheet records it.
- **Location:** `experiments/embeddings-from-first-principles/relate/doc/` — the
  experiment owns it, not Hugo `content/`.

## `relate-0.1.0` — 2026-09-08

**Frozen.** `corpus_hash` in `../relate-0.1.0/corpus_hash.txt`
(`8cad6816d90e06bc49e5b0b64cd460945e17ea4b1ec3c46054409669eda525b3`).

- Stages 4–9 of the build sequence complete (`sources.py`, `generate.py`,
  `build_full.py`): 1,173 base items, 1,181 typed pairs, 269 queries, all 11
  relations, 5 domains, 63 template families, 250 entity families. Validates
  with 0 errors, 0 warnings; deterministic rebuild reproduces the hash
  (`build_full.py --check`).
- **Overlap strata** (spec §5): every balanced relation (`paraphrase`,
  `entailment`, `contradiction`, `topic-related`) holds ≥25% of its pairs in
  each low/mid/high band, enforced by a stage-6 balancer that trims over-full
  bands after ~70 hand-authored low-overlap paraphrase/entailment and
  high-overlap topic-related and mid-overlap contradiction pairs were added.
- **Adjudication** is rule-based against `ontology.json` (no human
  double-annotation in v0.1; no α — the datasheet reports a rule-consistency
  audit and the per-relation overlap distribution instead; α is the top v0.2
  task). `negation`→`contradiction` and `relation-swap`→`contradiction`
  auto-re-adjudication per spec §3.6 / §3.11.
- **Leakage / contamination** (spec §8): exact + shingle-Jaccard≥0.9 dedup;
  template generation from `sources.py` is the contamination defense — no text
  is drawn from SNLI/MNLI/STS-B/PAWS/BEIR.
- **Task views** (spec §13): `retrieval`, `relation`, `hard-negative`,
  `calibration`, `dimensionality`, `alignment`.
- **No ontology change** — `spec_version` stays 0.1.0.
- **Relocation (no content change):** `spec/RELATE-v0.1.md`,
  `spec/RELATE-CHANGELOG.md`, and the claim ledger (`claim-to-experiment-registry.md`
  → `../claims.md`) moved out of `content/books/embeddings-from-first-principles/`
  into the experiment tree. The executable experiment owns its specification;
  Hugo `content/` is a publication boundary. `ontology.json` `spec_source`
  updated.

  Repository note: the historical claim ledger is not vendored in this RELATE
  repository. The active book-to-benchmark map lives at
  `../../evidence/BOOK-EVIDENCE-MAP.md`.

Known gaps:
- **RELATE-DOC** (spec §11) not built — gates Wave 4 + Transformation Wave.
- **Query difficulty** — Wave 1 (2026-09-08) found the query set saturates
  system-level comparisons (rows 1.7, 1.9, 1.13, 1.14): the grade-3 positives are
  near-restatements of their queries, so a competent encoder + plain dense
  retrieval already wins. Pair-level probes are unaffected. **v0.2's top task**
  (ahead of Krippendorff's α): a hard query set — written questions with a real
  lexical/semantic gap to the answer, grade-3 positives that are not restatements.

## Stages 1–3 — 2026-09-08

- **2026-09-08** — Stages 1–3 of the build sequence complete:
  - `ontology.json` frozen at `spec_version` 0.1.0 (11 relations, adjudication order, domains, relevance grades, overlap bands, hard-negative methods, id patterns).
  - Validators implemented before generators (`validate.py`, 16 unit checks).
  - Tiny RELATE-DEV built and inspected (`dev/`): 54 items / 36 pairs / 3 queries across all 11 relations and 5 domains; validates clean.
  - **Spec clarifications** added to `RELATE-v0.1.md` §17 (no ontology change):
    (1) `negation` overlap is post-lemmatization and multi-claim negation must
    negate the top-level assertion; (2) in `product-support` the product is the
    shared named entity; (3) `partial-support` α is a priority datasheet metric.
