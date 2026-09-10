# RELATE v0.1 — Corpus Specification

**Status:** BUILT AND FROZEN as `relate-0.1.0` (2026-09-08). This document is the
specification; the frozen artifact is in `../relate-0.1.0/` with its
`corpus_hash`, `manifest.json`, and `datasheet.md`. Deviations of the built v0.1
from this spec's targets are recorded in the datasheet's *Known limitations*
(pair volume at the low end of the §1 envelope; synthetic template generation in
place of human annotation + α; `geo-civics` domain skew; RELATE-DOC §11 not yet
built).
**Owner:** Ernan Hughes
**Spec date:** 2026-09-08
**Location note:** this spec lives in the experiment tree
(`experiments/embeddings-from-first-principles/relate/spec/`), not in Hugo
`content/` — the executable experiment owns its specification.
**Corpus version at build time:** `relate-0.1.0` (semantic version; the built artifact carries a content hash — see §12).

RELATE = **Rel**ations **E**xplicitly **L**abeled **A**nd **T**yped for **E**mbeddings.

---

## 0. Why RELATE exists and what it must not be

The book's argument is empirical: it repeatedly claims that a given embedding geometry captures *aboutness* well and *polarity / role / time / claim-strength* poorly, and it measures where retrieval, calibration, and cross-space translation break. Every one of those measurements needs a dataset where **the relationship between two texts is known, typed, and defended against the obvious objections**.

RELATE must survive three reviewer objections:

1. **"Your categories overlap arbitrarily."** → §3 gives each relation a decision procedure and §3.9 gives the adjudication order for ambiguous pairs.
2. **"Your negatives are trivially easy / your positives are trivially lexical."** → §5 lexical-overlap controls; §6 hard-negative construction; every pair carries a lexical-overlap score.
3. **"You memorized generation patterns / there is train–test leakage."** → §7 splits by entity, template, and domain; §8 leakage checks; §9 provenance.

RELATE is **not** a general benchmark of embedding model quality (that is MTEB's job). It is a **diagnostic instrument** with known structure, designed so that a failure on RELATE localizes to a capability.

---

## 1. Size and shape

| Quantity | v0.1 target | Rationale |
|---|---|---|
| Base items (unique short texts) | 1,000–1,400 | Enough for stable Recall@10 / nDCG@10 CIs on ~300 queries; small enough to embed under many models cheaply and to brute-force exactly (Ch 9). |
| Labeled ordered pairs | 4,000–6,000 | ~4–6 typed relations anchored per base item on average. |
| Query set (for retrieval labs) | 300–400 | Drawn from base items or written as questions (§4). |
| Relations (types) | 11 (§3) | Frozen for v0.1. |
| Domains | 5 (§2) | For split-by-domain generalization tests. |
| Template families | ≥ 40 | For split-by-template tests (Ch 18–21 anchor generalization). |
| Entity families | ≥ 120 | For split-by-entity tests. |

Items are **short**: 1 sentence to ~3 sentences, 5–60 tokens. A separate **RELATE-DOC** extension (§11) adds ~200 longer source documents (300–3,000 words) for the compression chapters (22–23); it is versioned alongside but built second.

---

## 2. Domains

Five domains, chosen to vary register, entity density, and the availability of ground truth:

| Domain | Register | Why included |
|---|---|---|
| `geo-civics` | encyclopedic | clean factual relations (capitals, borders, populations); easy to construct verifiable contradictions and temporal mismatches. |
| `biomed-claims` | scientific-abstract | claim-strength and partial-support distinctions are natural ("reduced symptoms" vs "reduced symptoms in a subgroup"). |
| `product-support` | informal / user-written | keyword queries, typos, entity-and-version traps; matches real retrieval query style (Ch 13). |
| `corporate-events` | news-wire | relation-swap pairs (A acquired B / B acquired A), temporal ordering, entity overlap. |
| `everyday-statements` | conversational | negation and paraphrase pairs with high lexical overlap; the polarity stress cases. |

Each base item is tagged with exactly one domain. Cross-domain pairs are allowed only for the `unrelated` relation (and are marked).

---

## 3. Relation definitions

Labels attach to **ordered pairs** `(a, b)` unless noted symmetric. Direction matters for `entailment`, `partial-support`, `temporal-mismatch`, and `relation-swap`. Each relation has: a one-line definition, a **decision procedure**, a positive example, and a **near-miss** that belongs to a *different* relation.

Notation: `a ⊨ b` means "a entails b" (any world where a is true, b is true).

### 3.1 `equivalent` (symmetric)
**Definition.** `a` and `b` express the same proposition; `a ⊨ b` and `b ⊨ a`; truth-conditionally identical.
**Decision procedure.** Could a careful reader be shown one, then the other, and say "that's the same statement, reworded"? Swapping them in any factual context changes nothing.
**Example.** `Dublin is the capital of Ireland.` / `The capital of Ireland is Dublin.`
**Near-miss (→ paraphrase).** `Dublin is Ireland's capital city.` — adds "city", a near-truism here but a lexical/semantic addition; goes to `paraphrase`, not `equivalent`, if any nuance differs.

### 3.2 `paraphrase` (symmetric)
**Definition.** Same core meaning, possibly differing in emphasis, register, or non-truth-conditional detail; mutual entailment holds *for the salient claim* but one side may carry minor extra or softened content.
**Decision procedure.** Same event/claim, different words; a reader would accept "these say basically the same thing" but not necessarily "identical". If mutual entailment is exact and total → `equivalent` instead.
**Example.** `The committee rejected the proposal.` / `The proposal was turned down by the committee.`
**Near-miss (→ entailment).** `The committee rejected the proposal unanimously.` vs `The committee rejected the proposal.` — the second does not entail the first; ordered pair `(unanimous, plain)` is `entailment`, `(plain, unanimous)` is `partial-support` or `unrelated-strengthening` (we use `partial-support`).

### 3.3 `entailment` (directional: `a ⊨ b`, and *not* `b ⊨ a`)
**Definition.** Every situation making `a` true makes `b` true; the reverse fails (strict entailment, one-directional).
**Decision procedure.** Assume `a`. Is `b` now guaranteed? Then assume `b`. Is `a` still possibly false? Both yes → `entailment (a→b)`.
**Example.** `a = A golden retriever is asleep on the porch.` `b = A dog is on the porch.`
**Near-miss (→ paraphrase).** If the reverse also holds, it is `paraphrase`/`equivalent`. If `b` adds information not in `a`, the pair is not `entailment` in this direction.

### 3.4 `partial-support` (directional: `b` partially supports `a`)
**Definition.** `b` provides evidence for *some but not all* of `a`; `b` neither entails nor contradicts `a`; a reader using `b` to check `a` would say "supports part of this, silent on the rest."
**Decision procedure.** Decompose `a` into atomic claims. Does `b` entail at least one and leave at least one unaddressed (not contradicted)? → `partial-support`.
**Example.** `a = The drug lowered blood pressure and improved sleep in elderly patients.` `b = In a trial of elderly patients, the drug lowered blood pressure.`
**Near-miss (→ contradiction).** If `b` denies any atomic claim of `a`, the pair is `contradiction`, not `partial-support`.

### 3.5 `contradiction` (symmetric in *incompatibility*, but stored directional for provenance)
**Definition.** `a` and `b` cannot both be true in the same world; at least one atomic claim is directly negated or made incompatible (mutually exclusive values, incompatible relations).
**Decision procedure.** Is there *no* consistent world containing both? Distinguish from `negation` (§3.6): contradiction may be lexically diverse ("X is landlocked" / "X has a long coastline"); negation is the explicit "not" form.
**Example.** `The bridge opened in 1998.` / `The bridge did not open until 2004.`
**Near-miss (→ temporal-mismatch).** `The population was 1.4M.` (2005) / `The population was 1.9M.` (2024) — both true at different times → `temporal-mismatch`, not `contradiction`, *if* both carry or imply a time index. Only an atemporal incompatibility is `contradiction`.

### 3.6 `negation` (directional: `b` is the explicit negation of `a`)
**Definition.** `b` is `a` with the main predicate explicitly negated (surface "not", "no", "never", "fails to"), minimal other change; high lexical overlap by construction.
**Decision procedure.** Can `b` be produced from `a` by inserting/removing a negation on the main verb or a core quantifier, changing little else? → `negation`. (A subset of `contradiction`, separated because it is the *lexically closest* incompatibility and the sharpest polarity probe.)
**Example.** `Dublin is the capital of Ireland.` / `Dublin is not the capital of Ireland.`
**Constraint.** Lexical-overlap score (§5) for `negation` pairs must be ≥ 0.7 (Jaccard over content lemmas). If a "negation" pair has low overlap it is mislabeled → `contradiction`.

### 3.7 `temporal-mismatch` (directional: `b` is `a` at a different time)
**Definition.** `a` and `b` state the same relation about the same entity but with different, time-dependent values; each is (or was) true at its own time; the mismatch is *only* temporal.
**Decision procedure.** Same subject, same relation, values differ, and the difference is explained by "measured at different times" rather than "one is wrong." Requires both items to carry or plausibly imply a time.
**Example.** `As of 2004, the company had 1,200 employees.` / `The company employs 8,500 people (2023).`
**Near-miss (→ contradiction).** Remove the time indices and it becomes `contradiction`. The label depends on whether time is expressed/implied.

### 3.8 `topic-related` (symmetric) and `entity-related` (symmetric)
**`topic-related`.** `a` and `b` concern the same topic or event but make *different, logically independent* claims (neither entails, contradicts, nor partially supports the other). Example: `The treaty proposed a customs union.` / `The treaty was ratified by seven of nine states.`
**`entity-related`.** `a` and `b` share ≥ 1 named entity but are about different topics/claims and are not topic-related in a strong sense. Example: `Apple released a new laptop.` / `Apple's headquarters is in Cupertino.` Used mainly to build the "entity trap" hard negatives (Ch 10).
**Decision procedure / order.** If any of {equivalent, paraphrase, entailment, partial-support, contradiction, negation, temporal-mismatch} applies, use that. `topic-related` and `entity-related` are the residual "close but logically independent" bucket; `topic-related` requires shared topic, `entity-related` only shared entity.

### 3.9 `unrelated` (symmetric)
**Definition.** No shared topic, no shared salient entity, no logical relation. Often cross-domain.
**Decision procedure.** Would a reader say "these have nothing to do with each other"? Content-lemma Jaccard < 0.1 and no entity overlap.

### 3.10 `hard-negative` (a *derived* label, not a base relation)
`hard-negative` is not one of the 11 mutually-exclusive relations; it is a **role tag** applied to a `(query, candidate)` pair meaning "candidate is highly similar to the query under lexical/topical/entity signals but is *not* a correct answer to the query." Every hard negative also carries its underlying relation (`negation`, `relation-swap`, `temporal-mismatch`, `topic-related`, `entity-related`, `partial-support`). Construction: §6.

### 3.11 `relation-swap` (directional; a specialization used for hard negatives)
**Definition.** `b` contains the same entities and the same relation type as `a` but with the argument roles reversed (`A acquired B` → `B acquired A`; `X is north of Y` → `Y is north of X`).
Stored as its own relation for the Ch 10–11 / Ch 21 hard-negative-agreement probes. A `relation-swap` pair is always also a `contradiction` when the relation is asymmetric.

### Adjudication order (applied top to bottom; first match wins)
```
1. equivalent
2. negation            (explicit-not form of a contradiction)
3. relation-swap       (role-reversed form)
4. contradiction       (other atemporal incompatibility)
5. temporal-mismatch   (same relation, time-indexed differing values)
6. entailment          (strict one-directional)
7. partial-support     (supports some atomic claims, silent on others)
8. paraphrase          (same salient claim, minor differences, mutual entailment of the claim)
9. topic-related       (same topic, independent claims)
10. entity-related     (shared entity only)
11. unrelated
```
Every stored pair records **which rule fired** and **which competing relation was the runner-up** (for the "categories overlap" defense and for inter-annotator analysis).

---

## 4. Items vs pairs; the query set

- **Base items** are stored once, with: `id`, `text`, `domain`, `template_family`, `entity_family`, `time_index` (nullable), `atomic_claims` (list, for `partial-support` scoring), `provenance` (§9).
- **Pairs** are stored as `(a_id, b_id, relation, direction, rule_fired, runner_up, lexical_overlap, entity_overlap, annotator_ids, agreement, notes)`.
- **Queries** are a labeled subset: each query has one or more `positive` items (graded relevance 3/2/1) and, for the hard-negative labs, a set of `hard_negative` items each with its underlying relation. A query may be a base item ("more like this") or a written question targeting a base item ("Is Dublin the capital of Ireland?"). The `query_style` field is one of `{restatement, question, keyword, long-nl}` for the Ch 13 per-style breakdown.

Graded relevance definition (written out, per Ch 13's "define relevance"):
```
3  answers the query and is correct           (equivalent / paraphrase / forward entailment to the query claim, and true)
2  substantially supports the query            (partial-support covering the main claim)
1  on-topic, does not answer                    (topic-related)
0  everything else                              (entity-related, contradiction, negation, temporal-mismatch, unrelated)
```
Note that `contradiction`/`negation` get relevance **0** even though they are "about" the query — this is the design choice that makes RELATE a polarity probe.

---

## 5. Lexical-overlap controls

Every pair carries `lexical_overlap` = Jaccard over **content lemmas** (stopwords removed, lemmatized) and `char_3gram_overlap` (Sørensen–Dice over character trigrams, to catch typo/morphology cases).

Design constraints:
- For each relation, the **distribution** of `lexical_overlap` is reported in the datasheet, and the build targets **overlap-balanced strata**: within `paraphrase`, `entailment`, `contradiction`, and `topic-related`, at least 25% of pairs must fall in each of low (`<0.2`), mid (`0.2–0.5`), and high (`>0.5`) overlap bands. This prevents "the model just did lexical matching" from explaining a result.
- `negation` is deliberately high-overlap (§3.6, ≥ 0.7).
- `equivalent`/`paraphrase` must include a **low-overlap subset** (different vocabulary, same meaning) — these are the cases that separate a semantic model from a lexical one.
- A **BM25 score** for every `(query, item)` pair is precomputed and stored, so hybrid-retrieval labs (Ch 12) and lexical-baseline comparisons need no recomputation.

---

## 6. Hard-negative construction

Hard negatives are built **per query**, by four methods, each tagged so labs can stratify (Ch 11):

1. **Structured perturbation of a positive** (primary, highest value): take a relevance-3 item and apply one transformation —
   - `negation` (insert "not" on the main predicate),
   - `relation-swap` (reverse argument roles),
   - `temporal-shift` (change the year / tense to a different valid time),
   - `quantifier-weaken` (`all` → `some`, `always` → `often`),
   - `entity-substitute` (swap one named entity for a same-type sibling from the entity family).
   Each perturbation is applied by template + human check, and the resulting item is added to the base pool with its own relation label.
2. **Lexical-overlap-matched non-answers** (BM25 top-k that are not relevance ≥ 2).
3. **In-model mined** (top cosine non-answers under a *reference* model — see §10 — with a top-2 skip margin to reduce false negatives).
4. **Entity-matched non-answers** (share the query's entity, different claim).

**False-negative control.** Every mined hard negative (methods 2–4) is human-checked for "is this actually relevant?" A per-method false-negative rate is reported in the datasheet. Structured perturbations (method 1) have near-zero false-negative rate by construction and are the ones used for the headline hard-negative-agreement numbers (Ch 16, 18–21).

---

## 7. Splits

Three orthogonal split axes, each producing train / dev / test at 60 / 20 / 20:

| Split axis | Purpose | Rule |
|---|---|---|
| `split_random` | default IID evaluation | pair-level random, but **no base item appears in two splits** (prevents trivial leakage). |
| `split_entity` | generalization to unseen entities | entity families partitioned; test entities never seen in train. For Ch 18–21: does a bridge fitted on train-entity anchors translate test-entity items? |
| `split_template` | generalization to unseen surface patterns | template families partitioned; test templates never seen in train. Detects "the model / the bridge learned the generation pattern." |
| `split_domain` | generalization to unseen domain | leave-one-domain-out (5 folds). For Ch 13 (domain shift) and Ch 16 (do models agree less out of domain?). |

Anchor sets for the bridge chapters (18–21) are drawn **only from train** under the relevant split; held-out anchor evaluation uses dev; final bridge numbers use test. This is stated in each lab.

---

## 8. Leakage and duplication checks (run at build time, recorded in the datasheet)

- **Exact and near-duplicate detection** across the whole item pool (MinHash LSH, Jaccard ≥ 0.9 on shingles): near-dups collapsed or moved to the same split.
- **Cross-split base-item check**: assert no `base_item.id` spans splits on any axis.
- **Query–positive triviality check**: flag any `(query, relevance-3)` pair with `lexical_overlap = 1.0` unless the relation is `equivalent`; cap the fraction of such pairs.
- **Public-corpus contamination scan**: check base items against a set of known public NLI/STS/retrieval datasets (SNLI, MNLI, STS-B, BEIR subsets, PAWS) by near-duplicate match; report overlap. Aim: RELATE items are newly written or heavily transformed, not lifted.
- **Template-leakage between anchor and eval** for the bridge labs: assert anchors and eval items do not share a template family under `split_template`.

---

## 9. Annotation and provenance

- **Item provenance** (`provenance` field): one of `authored` (written for RELATE), `adapted` (rewritten from a public fact, with source noted), `perturbed` (derived by a §6 transformation from another RELATE item, with parent id and transformation).
- **Labeling.** Each pair labeled independently by ≥ 2 annotators using the §3 decision procedures and the §3.9 adjudication order; a third adjudicates disagreements. Store per-annotator labels, the final label, and **Krippendorff's α per relation** (report in datasheet; relations with α < 0.67 are flagged as "definition needs work for v0.2").
- **Guidelines document** versioned with the corpus (`spec/RELATE-annotation-guidelines-v0.1.md`, to be written before labeling).
- **Adapted-fact sourcing**: `geo-civics` and `corporate-events` facts cite a source and an as-of date so `temporal-mismatch` and `contradiction` pairs are defensible.
- **License**: target CC BY 4.0 for the released artifact; ensure adapted facts are expression-original.

---

## 10. Reference model (for mining only, not for claims)

Hard-negative mining (§6 method 3) and the "in-model" negative set (Ch 11) need *a* model. To keep the corpus model-independent, RELATE fixes a **declared reference model** for construction: `<to be pinned at build: an open, widely available sentence encoder + exact version>`. The corpus datasheet records it. **No book claim uses the reference model as ground truth**; it is disclosed so that in-model negative difficulty is reproducible.

---

## 11. RELATE-DOC extension (for Ch 22–23)

- ~200 source documents, 300–3,000 words, in `geo-civics`, `biomed-claims`, `corporate-events`.
- Each document has: `atomic_claims` (list with a `salience` rank and a `is_rare_entity` flag), a set of queries it answers (relevance-graded), and **six controlled compressions** (Ch 22's adversarial set):
  1. `faithful` — human summary, all salient claims kept;
  2. `topic-drift` — same topic, one salient claim silently changed;
  3. `number-dropped` — one critical numeric claim removed;
  4. `relation-reversed` — one relation's arguments swapped;
  5. `minority-entity-dropped` — a low-salience but load-bearing entity removed;
  6. `conclusion-lost` — supporting detail kept, the document's conclusion omitted.
- Plus method-based compressions (`truncation`, `extractive`, `abstractive`) at ratios {50, 25, 10, 5}% for the retention curve.
- **Transformation pairs for Ch 23**: ≥ 150 `(source, target)` sentence pairs per transformation type for `verbose→concise`, `formal→informal`, `active→passive`, `statement→negation`, `plain→hedged`, `present→past`, each with a template family for the `split_template` transfer test.

---

## 12. Versioning and the corpus hash

- Semantic version `relate-MAJOR.MINOR.PATCH`. v0.1 is `0.1.0`.
- The **built artifact** (JSONL files + datasheet) is content-addressed: `corpus_hash = SHA256` of the canonicalized, sorted concatenation of all item and pair records plus the datasheet. Every experiment records the `corpus_hash` it ran against (parallel to the book's `space_hash`).
- Breaking changes (relation set, split rules, item removal) → MINOR bump. Additive (more pairs, RELATE-DOC) → PATCH if splits are stable, MINOR if not. Relabeling from α feedback → MINOR.
- Changelog in `spec/RELATE-CHANGELOG.md`.

---

## 13. Released files (build output)

```
relate-0.1.0/
  items.jsonl                 base items (§4 schema)
  pairs.jsonl                 typed pairs (§4 schema)
  queries.jsonl              query set with graded positives + hard negatives
  splits/
    random.json  entity.json  template.json  domain.json
  doc/
    documents.jsonl  compressions.jsonl  transformation_pairs.jsonl
  bm25_scores.parquet        precomputed (query|item) x item
  datasheet.md               §14
  corpus_hash.txt
```

---

## 14. Datasheet contents (Gebru et al. "Datasheets for Datasets" structure)

Motivation; composition (counts per relation × domain × split × overlap band); collection process (authoring, adaptation sources, perturbation templates); preprocessing (lemmatization, dedup, LSH params); **inter-annotator α per relation**; **per-relation lexical-overlap distributions**; **hard-negative false-negative rates per method**; **public-corpus contamination scan results**; reference model; known limitations (English-only v0.1; Western-centric entities in `geo-civics`/`corporate-events`; `temporal-mismatch` depends on annotator world knowledge); recommended uses and misuses; license; corpus hash.

---

## 15. Build plan (order of operations)

1. Freeze relation defs (§3) + write annotation guidelines.
2. Author `geo-civics` and `everyday-statements` base items (highest-value, cleanest relations) → ~400 items.
3. Perturbation templates (§6.1) → generate structured hard negatives → human check.
4. Label pairs (2+ annotators, adjudication) → compute α → revise definitions if needed.
5. Add `biomed-claims`, `corporate-events`, `product-support`.
6. Build queries + graded relevance + hard-negative sets.
7. Dedup + leakage + contamination scans (§8).
8. Splits (§7).
9. Precompute BM25.
10. Datasheet + hash + release `0.1.0`.
11. RELATE-DOC as `0.1.1` / `0.2.0`.

**Wave-1 experiments (Ch 1, 4, 9, 10, 11, 13, 14) need only steps 1–9.** RELATE-DOC (step 11) gates Wave 4.

---

## 16. Open questions for v0.2

- Should `paraphrase` split into `paraphrase-exact` (mutual entailment) and `paraphrase-loose` (same gist)? α results decide.
- Multilingual RELATE (cross-lingual alignment is a natural Ch 18–19 extension).
- A `numeric-approx` relation ("about 1.4 million" / "1,398,412") — currently folded into `paraphrase`/`equivalent`.
- Human ceiling: have annotators *do* the retrieval task on a sample to establish a human Recall@1 for comparison.

---

## 17. Clarifications from RELATE-DEV (2026-09-08)

Building the tiny RELATE-DEV set (`experiments/embeddings-from-first-principles/relate/dev/`, findings in `DEV_FINDINGS.md`) surfaced three points that refine, but do not change, the ontology. They apply at the full build:

1. **`negation` overlap is measured after lemmatization**, and a structured `negation` of a sentence with ≥ 2 atomic claims must negate the *top-level assertion* (De Morgan the conjunction), keeping all content tokens — negating a single conjunct is `contradiction`, not `negation`. (Refines §3.6, §6.)
2. **In `product-support`, the product under discussion (`the app`, `the printer`) counts as the shared named entity** for `entity-related` (§3.8) and for entity-matched hard negatives (§6). The encyclopedic/news/scientific domains keep the strict proper-noun reading.
3. **`partial-support` is the relation most dependent on human adjudication** — a validator can check the `a` side has ≥ 2 atomic claims but not that `b` entails one of them. Its inter-annotator α (§9) is a priority metric and must be reported and flagged separately in the datasheet.

Machine-readable ontology freeze: `experiments/embeddings-from-first-principles/relate/ontology.json`. Any change there is a MINOR version bump recorded in `spec/RELATE-CHANGELOG.md`.
