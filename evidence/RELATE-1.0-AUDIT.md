# RELATE-1.0-AUDIT.md — reconciliation verdict (2026-09-11)

Overarching question: does RELATE 1.0 faithfully implement the system
*Embeddings From First Principles* derives, and can every production
capability be traced to evidence that earned it?

Verdict: **YES, with the book now lagging the implementation.**
No UNJUSTIFIED PRODUCT SURFACE was found. The largest remaining gap
is inverted from the usual direction: the book still describes as
conceptual or future what RELATE already does. The work ahead is book
edits (listed in `CHAPTER-26-RECONCILIATION.md`), not code — with two
narrow `DOCUMENT FUTURE WORK` items below. No `IMPLEMENT RELATE GAP`
was issued.

## Pass 2 — capability → evidence (reverse map, condensed)

Identity: `SpaceIdentity`/`SpaceRegistry`/`derive*` → Ch 17,
hash-exactness doctrine. Comparison: `CorrespondenceSet`,
`SpaceComparisonReport`, mixing guard → Waves 3/6 alignment
discipline. Hard negatives: cases/observations/reports/deltas,
four scorers → original RELATE + Wave 1 (generic geometry misses
relation structure). Geometry: reports, sampling spec, erank/PR,
TwoNN-with-provenance, CKA → Wave 2 (spectrum honesty, rotation
invariance). Neighborhoods: comparison, counterpart (separate),
density/hubness/stability primitives → Wave 1 hubness + Waves 3/6
(recovery ≠ fidelity). Calibration: distributions, ROC/AUC/EER,
three-way decisions, scopes, descriptors, staleness-why →
Wave 1 (0.75/24%/86%/0.10). Bundles: evidence-only composition,
external separation, availability → Wave 1 (0.76→0.90→0.897).
Bridges: directional producers, correspondence fitting, controls,
registry without ranking, derived identity → Waves 3/6 (map ≠
preservation; direction matters). Preservation: results, frames,
policies, verdicts, explain → Waves 3/6 (no universal winner).
Compression: PCA/random/prefix + knees → Waves 2/4 (task knees;
drift ≠ faithfulness). Operators: content cases, four rungs,
selection, NONE_PASS → Wave 5 (identity/delta/none taxonomy).
Lineage: derivation records, walk, no transitive permission →
Ch 17 versions + bridge-composition lesson. Observatory/CLI:
composition + replay hash → Ch 26 capstone contract.

UNJUSTIFIED PRODUCT SURFACE: none found. Two notes, neither a
violation: legacy `CompressionRecord`/`TransformationRecord` DTOs
are superseded for decisions by profiles/selection but retained
for Wave artifact compat (marked, not removed); `Observatory.inspect`
is a trivial helper (architectural necessity, not a claim).

DOCUMENT FUTURE WORK (not gaps): standalone `evaluate_round_trip`
(the vocabulary + benchmark practice exist; no dedicated function);
task-restricted effective rank (still a book hypothesis 2.5, correctly
unimplemented); claim-conditioned (L4) measurement (in Wave-4
artifacts, correctly unpromoted); cost/migration estimator (book
figures schematic by its own statement); ANN instrumentation (index
infrastructure, correctly out of scope).

## Pass 3 — claims audit

| Claim | Evidence | Runtime enforcement | Benchmark | Verdict |
|---|---|---|---|---|
| A vector is not meaning | Ch 1, identity doctrine | `attach` refuses spaceless vectors; registry lookup | every runner's provenance | SUPPORTED (boundary noted: bare arrays still flow between calls) |
| Similarity is not equivalence | Ch 4, relation readout | relation scorers; preference convention; calibration bands | hard-negatives mirror | SUPPORTED |
| Retrieval is not verification | Ch 12/15 | `search` vs `inspect_result`/external separation | signal-bundles | SUPPORTED |
| Equal dims ≠ compatible | Ch 16/17 | mixing guard; identity control measured, never authorizing | cross-space permuted control | SUPPORTED |
| A bridge is not compatibility until measured | Ch 20/21 | producer carries zero verdict fields (source-scanned) | bridges/preservation | SUPPORTED |
| Counterpart recovery ≠ preservation | Wave 6 | separate DTOs; split demonstrated | neighborhoods, preservation | SUPPORTED |
| Preservation requires explicit authority | Ch 21 | required `ReferenceFrame` (stringly rejected) | all judgment benchmarks | SUPPORTED |
| Derived spaces receive new identity | Ch 17 | derive functions; native-hash reuse asserted absent | bridges/compression/operators | SUPPORTED |
| Global geometry ≠ faithfulness | Wave 4 | `check_compression` fail-closed; transfer FAIL coexists with retrieval PASS | compression, preservation | SUPPORTED (runtime cannot verify faithfulness at all — correct scope) |
| Every transformation obligates measurement | thesis | universal evaluate path per family; wrappers delegate | CLI three cartridges | QUALIFY (conventional, not compiler-enforced; the path exists for all families) |
| Lineage composes; permission does not | Ch 17/21 | records carry no verdicts; no transitive API; negative test | lineage tests | SUPPORTED |

## Pass 7 — canonical vocabulary

space (identified vectors + hash) · representation (vectors in a
space) · derived/candidate space (new hash, parent + transformation
lineage; never native) · map (function) · bridge (directional fitted
artifact A→B) · transformation (any producer under the 5A contract) ·
operator (relation-bound vector hypothesis; `identity_map` =
operator identity, never semantic identity) · preservation
(measured results vs authority) · compatibility (empirical,
task-dependent; never a scalar) · usability (policy verdict per
scope, fail-closed) · reference/authority/frame (whose behavior
judges: target-/source-native, task-gold) · evaluation (measuring) ·
profile (results + verdicts) · verdict/policy (PASS structural,
WARN advisory-miss, FAIL hard-or-unmeasured) · hard negative
(anchor prefers positive over deceptive negative; WIN/TIE/LOSS) ·
negative set (named distribution with content hash; calibration
belongs to it, not the model) · scope (task/domain/query-type data,
not branching).

Layers, unavoidable: IDENTITY exact provenance;
COMPATIBILITY measured preservation; USABILITY policy-scoped
permission.

## Gates 1–30 checklist

1–2 mapped (matrix + reverse map above; no unjustified surface).
3 no capability justified by mention alone (every row cites
experiment/invariant/necessity). 4 all README claims traced
(Pass 3). 5 book runtime-behavior claims checked (Pass 4 notes).
6 hypothetical language identified (8 passages). 7 RELATE-only
concepts listed (12 items). 8–11 terminology enumerated,
three layers + authority + bridge vocabularies fixed (Pass 7).
12–13 benchmarks mapped; HISTORIC/MIRROR/EXTERNAL kept distinct
(BOOK-EVIDENCE-MAP.md + labeling audit). 14–18 negative
results visible (`NONE_PASS` intact; recovery≠structure;
geometry≠faithfulness; scoped calibration). 19–20 identity
rules match; lineage without transitive permission. 21 no
product feature added (three Markdown files only — verified by
diff stat). 22–24 rewrite plan, surgical list, README items
(above + reconciliation doc). 25–27 three artifacts produced
here. 28–30 benchmarks/corpus/runtime untouched (no source,
benchmark, or corpus paths in this diff).

## README updates identified

None blocking. Optional post-audit: link `evidence/` docs from
the root README runtime section; consider a verdict-table
excerpt from `relate demo` output. Applied only after this tag.

## Final verdict

RELATE 1.0 implements the Chapter 26 architecture with
measurement/policy separations the book argues for and, in
several places (correspondence discipline, reference frames,
`NONE_PASS`, lineage without transitive permission), with
stricter machinery than the prose currently describes.
Recommend: tag `relate-1.0-reconciliation-audit`, then work is
book edits per `CHAPTER-26-RECONCILIATION.md`, then release
hardening. No new architecture is needed for 1.0.
