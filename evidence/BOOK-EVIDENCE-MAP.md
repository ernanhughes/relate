# BOOK-EVIDENCE-MAP.md — benchmark families back to book claims

Authoritative table mapping RELATE's ten benchmark families to the book
they execute. Status vocabulary (enforced per benchmark):

- `HISTORIC` — preserved prose evidence, not recomputed here (frozen
  assets absent); lives in `expected/historic-reference.json` as
  `EXTERNAL` with provenance and an explicit non-recomputation note.
- `REPRODUCED` — the preserved result recomputed through the generic
  evaluator within tolerance. (None yet: every family needing
  embedding-model assets is a mirror until those assets are vendored.)
- `MIRROR` — the observation structure reproduced on synthetic vectors
  through production APIs; digits differ honestly and the README says
  why.
- `EXTERNAL` — out-of-tree supporting evidence, cited, never promoted
  to a book `MEASURED` claim (same rule as the book's claims ledger).

Labeling audit (2026-09-11): full category discipline in
`hard-negatives`, `operators`, `signal-bundles` (historic-ref file +
README categories). `cross-space` ships the historic file but its
README does not yet state the categories — recommended edit.
`calibration`, `geometry`, `neighborhoods` cite waves and mirrors in
prose but pin no historic-reference artifact — recommended edit.
`bridges`, `preservation`, `compression` carry no historic file and no
wave pointer in-README; their numbers are mirror-only by construction
and should say so — recommended edit. No measurement artifact is
affected by any of these; they are documentation findings.

## 1. hard-negatives

- Benchmark path: `benchmarks/hard-negatives/` (`relate_cases.py`,
  `synthetic_vectors.py`, `run.py`).
- Canonical input: frozen `relate-0.1.0/views/hard-negative-v0.1.json`
  (937 triples) for the adapter half; seeded synthetic vectors for the
  scoring half.
- Spaces: synthetic Python-structure mirror (unit-norm, 32-d).
- Historical wave: original RELATE CodeBERT result + Wave 1 rows
  1.8/1.12 (margin collapse, signal ablation context).
- Status: MIRROR + HISTORIC(EXTERNAL). Mirror: relation 0.725 vs
  cosine/euclidean 0.100 through one evaluator. Historic: 0.733 /
  0.532 / 0.533, not recomputed (CodeBERT assets absent).
- Book chapters using result: 10, 11, 15.
- Headline: supervised readout exposes what raw geometry misses.
- Limitations: mirror negatives adversarially mined, so raw-geometry
  figures are harsher than the fixed historic pairs.

## 2. geometry

- Benchmark path: `benchmarks/geometry/` (`run.py`).
- Canonical input: none (seeded ISO/ANISO/LOWRANK spaces, n=1200,
  nominal d=256).
- Spaces: synthetic.
- Historical wave: Wave 2 rows 2.1/2.2/2.8 (rotation invariance,
  dimensionality, shape comparison).
- Status: MIRROR (no historic-ref file; recommended edit).
- Book chapters using result: 5, 7, 8.
- Headline: random-pair cosine 0.00 vs 0.49 across spaces; nominal 256
  → effective rank 6.0 / TwoNN 5.3; rotation drift < 1e-9.
- Limitations: synthetic; TwoNN is estimator-dependent by construction.

## 3. neighborhoods

- Benchmark path: `benchmarks/neighborhoods/` (`paired_spaces.py`,
  `run.py`).
- Canonical input: none (seeded paired spaces, shared skeleton).
- Spaces: synthetic X/Y pair.
- Historical wave: Wave 3 rows 3.4/3.9 (ladder, round-trip) and the
  Wave 6 recovery/structure split in shape.
- Status: MIRROR (no historic-ref file; recommended edit).
- Book chapters using result: 16, 21, 22.
- Headline: CKA 0.927, counterpart top-1 1.000, NN overlap 0.626,
  relation readout 0.725 → 0.585 in one report.
- Limitations: shared-nuisance pair is easier than cross-family
  encoders; centroid-split expressiveness proof lives in tests.

## 4. calibration

- Benchmark path: `benchmarks/calibration/` (`run.py`).
- Canonical input: none (seeded Gaussian preference scores).
- Spaces: synthetic score distributions (no vectors by design —
  scorer-agnosticism is the point).
- Historical wave: Wave 1 rows 1.10/1.11 (calibration, threshold drift).
- Status: MIRROR (no historic-ref file; recommended edit).
- Book chapters using result: 14.
- Headline: hard regime AUC 0.774 / EER 0.296 / ambiguity 0.478 vs
  ordinary 0.925 / 0.159 / 0.126; domain EER shift +0.109.
- Limitations: Gaussian draws, not measured encoder distributions.

## 5. signal-bundles

- Benchmark path: `benchmarks/signal-bundles/` (`signals_fixture.py`,
  `run.py`).
- Canonical input: none (seeded clustered fixture + simulated
  verifier).
- Spaces: synthetic.
- Historical wave: Wave 1 row 1.12 (signal ablation).
- Status: MIRROR + HISTORIC(EXTERNAL). Mirror: 0.677 → 0.906,
  stacked 0.906. Historic: 0.76 → 0.90 → 0.897.
- Book chapters using result: 15.
- Headline: geometric bundle rescues hard cases; generic verifier
  adds nothing reliable.
- Limitations: simulated verifier with correlated mistakes; the
  logistic probe is benchmark methodology, not runtime.

## 6. cross-space

- Benchmark path: `benchmarks/cross-space/` (`native_spaces.py`,
  `run.py`).
- Canonical input: none (seeded typed natives, shared backbone).
- Spaces: synthetic X (full) / Y (fine-weakened).
- Historical wave: Wave 3 rows 3.1/3.4/3.7 + Wave 6 shape.
- Status: MIRROR + HISTORIC(EXTERNAL, file present; README
  categorization recommended).
- Book chapters using result: 16–23.
- Headline: CKA 0.948, counterpart 1.000, NN 0.513, negation
  Δ −0.21, transfer FAR 0.10 → ~0.19.
- Limitations: shared backbone understates cross-family divergence.

## 7. bridges

- Benchmark path: `benchmarks/bridges/` (`bridge_cases.py`, `run.py`).
- Canonical input: none (seeded typed spaces, train/eval anchor
  split).
- Spaces: synthetic; six producers fit on train anchors.
- Historical wave: Wave 3 ladder (paired).
- Status: MIRROR-ONLY by construction (no historic file, no wave
  pointer in-README; both recommended). Deltas follow the 4A
  target-minus-candidate convention, documented in-README.
- Book chapters using result: 18–21.
- Headline: linear/ridge NN ~0.79 beat Procrustes-at-identity 0.60;
  centroid floor collapses; train reconstruction labeled diagnostic.
- Limitations: same fixture family as cross-space; rigid-map
  coincidence is fixture-specific and documented as such.

## 8. preservation

- Benchmark path: `benchmarks/preservation/` (`preservation_cases.py`,
  `run.py`).
- Canonical input: none (seeded typed spaces, coarse→fine
  direction, reverse fits for round-trips).
- Spaces: synthetic derived + native.
- Historical wave: Wave 3 rows 3.6/3.7 + Wave 6 trade-offs.
- Status: MIRROR-ONLY by construction (same documentation
  recommendation as bridges).
- Book chapters using result: 21.
- Headline: fitted maps PASS retrieval/neighborhood_use, FAIL
  threshold_transfer; controls fail everything; no universal winner.
- Limitations: transfer verdicts inherit the synthetic score
  distributions; round-trips need reverse fits that real upgrades
  may not have.

## 9. compression

- Benchmark path: `benchmarks/compression/` (`compression_cases.py`,
  `knee.py`, `run.py`).
- Canonical input: none (seeded typed 128-d natives).
- Spaces: synthetic native + derived cartridges.
- Historical wave: Wave 2 rows 2.3/2.6/2.7 (retention knees, method
  ordering).
- Status: MIRROR-ONLY by construction (same documentation
  recommendation). SOURCE_NATIVE authority throughout.
- Book chapters using result: 7.
- Headline: task knees retrieval 64 / fine_ordering 16 / transfer
  None; PCA-8 keeps CKA 0.99 while failing fine ordering.
- Limitations: knee bounds are declared policies, not discovered
  constants; synthetic spectrum, not a measured encoder's.

## 10. operators

- Benchmark path: `benchmarks/operators/` (`operator_cases.py`,
  `run.py`).
- Canonical input: REAL frozen `relate-doc-0.1.0`
  `transformation_pairs.jsonl` — 260 pairs, nine exact classes,
  real content hashes, deterministic train/eval splits. Strongest
  case provenance in the repo.
- Spaces: deterministic mirror encoder over the real pairs
  (geometry synthetic, case structure real).
- Historical wave: Wave 5 rows T.1–T.4.
- Status: MIRROR-ON-REAL-CASES + HISTORIC(EXTERNAL).
- Book chapters using result: 25.
- Headline: five `identity_map`, weakened claim `constant_delta`,
  three `NONE_PASS`; linear/affine train ≈1.0 against held-out
  failure.
- Limitations: mirror geometry; small per-type eval sets
  (relation_swap: 10 pairs total).
