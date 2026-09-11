# CHAPTER-26-RECONCILIATION.md — capstone components vs RELATE 1.0

Status vocabulary: `IMPLEMENTED` (real object, real behavior) /
`PARTIAL` (capability exists, wiring or enforcement incomplete) /
`BOOK-ONLY` (correctly out of runtime scope, or a hypothesis) /
`RELATE-ONLY` (runtime concept the book does not yet name) /
`RENAMED` (concept survives under another name/shape) / `OBSOLETE`
(superseded; retained only for artifact compat where noted).

## Pass 1 matrix — every Chapter 26 component

### space_record / space_registry (Ch 1, 17)

- Implementation status: IMPLEMENTED.
- Public object: `SpaceIdentity`, `SpaceRegistry`
  (`src/relate/spaces/`).
- Observatory surface: `register_space`, `attach` (refuses spaceless
  vectors), `spaces.require`.
- Evidence source: Ch 17 versioning work; identity ≠ compatibility ≠
  usability.
- Benchmark: identity mechanics exercised in every benchmark's
  provenance; cross-space denial tested in `test_cross_space.py`.
- Known limits: none structural. Bare `ndarray`s still flow between
  calls; identity attaches at `attach()` time (documented boundary).
- Book wording still accurate? YES.

### geometry_probe / shape_profile (Ch 2, 8)

- Implementation status: IMPLEMENTED.
- Public object: `GeometryReport`, `describe_geometry`,
  `PairSamplingSpec` (`src/relate/evaluation/geometry.py`).
- Observatory surface: `inspect` (lightweight); full reports via
  evaluation functions; CLI demo prints shape lines.
- Evidence source: Wave 2 shape comparison; anisotropy findings.
- Benchmark: `benchmarks/geometry/`.
- Known limits: vectors-only summaries; no plots (deliberate).
- Book wording still accurate? YES.

### dimensionality_report (Ch 7)

- Implementation status: IMPLEMENTED, one honest exception.
- Public object: `effective_rank`, `participation_ratio`,
  `twonn_estimate` → `IntrinsicDimensionEstimate(method, estimate,
  n_samples, parameters)`.
- Observatory surface: via `GeometryReport.intrinsic_dimension`.
- Evidence source: Wave 2 dimensionality report.
- Benchmark: `benchmarks/geometry/` (nominal 256 → 6.0/5.9/5.3).
- Known limits: task-restricted effective rank (the book's open
  hypothesis 2.5) is NOT implemented — correctly, since it is still
  a hypothesis, not a result. Flagged future work.
- Book wording still accurate? YES, with the hypothesis boundary
  already stated in the book.

### neighborhood_report (Ch 6)

- Implementation status: IMPLEMENTED with a shape note.
- Public object: `NeighborhoodReport`,
  `compare_neighborhoods` (+ `LocalDensity`, `Hubness`,
  `NeighborhoodStability`, `hubness_counts`).
- Observatory surface: `compare_spaces().neighborhood`;
  `inspect_result` bundles.
- Evidence source: Wave 1 hubness; Wave 3/6 neighborhood agreement.
- Benchmark: `benchmarks/neighborhoods/`.
- Known limits: no persisted "hub list" report object — hubness is a
  function output (`hubness_counts`), not a stored artifact. Same
  information, different shape; documented here.
- Book wording still accurate? YES (with that note).

### similarity_spec (Ch 4) / calibration_record (Ch 14)

- Implementation status: RENAMED (spec) / IMPLEMENTED (record).
- Public object: scorer identity (`scorer_id_of`), universal
  preference direction, `SpaceIdentity.normalize` (the distributed
  similarity spec); `CalibrationRecord` + `calibrate`,
  `CalibrationScope`, `NegativeSetDescriptor`,
  `CalibrationDecision`, `OperatingPoint`.
- Observatory surface: `record_calibration`; `inspect_result`
  carries calibration outcomes; policy routes on them.
- Evidence source: Wave 1 rows 1.10/1.11 (AUC 0.75, EER 24%, 86%
  escalate band, 0.10 domain shift).
- Benchmark: `benchmarks/calibration/`.
- Known limits: none — the 3C milestone deliberately refused
  `is_match()`.
- Book wording still accurate? YES; the book should name the
  preference-direction convention where Ch 4 discusses metrics.

### retrieval_policy (Ch 12)

- Implementation status: IMPLEMENTED with a boundary note.
- Public object: `RetrievalPolicy` (accept/rerank/verify/reject
  routing over bundles + calibration).
- Observatory surface: policy is a consumer object, not an
  Observatory method.
- Evidence source: Wave 1 policy-ablation discipline (components
  are hypotheses; ablate, don't assume).
- Benchmark: routing exercised in `test_observatory.py`;
  policy-ablation numbers live in Wave 1 artifacts.
- Known limits: RELATE routes retrieval; it does not implement
  hybrid/BM25 composition or reranking — application space, by
  design. Search execution (`RelationProjection.search`) and
  routing are separate objects.
- Book wording still accurate? YES.

### distractor_probe / negative_set_descriptor (Ch 10, 11)

- Implementation status: IMPLEMENTED.
- Public object: `HardNegativeCase`, `evaluate_hard_negatives`,
  `HardNegativeReport` (WIN/TIE/LOSS, per-relation/group evidence),
  `NegativeSetDescriptor`.
- Observatory surface: feeds `search` evaluation, bundles
  (margins), bridge/candidate deltas.
- Evidence source: original RELATE result + Wave 1 margin
  collapse/distractor work.
- Benchmark: `benchmarks/hard-negatives/` (937 frozen RELATE cases
  + mirror).
- Known limits: in-model mining ranks 3–7 caveat lives in Wave 1
  artifacts, correctly outside the generic evaluator.
- Book wording still accurate? YES.

### evaluation_card (Ch 13)

- Implementation status: IMPLEMENTED.
- Public object: `EvaluationCard` + per-capability card builders
  (`hard_negative_card`, `calibration_card`); no parallel DTOs.
- Observatory surface: `record_evaluation`.
- Evidence source: Wave 1 relevance-definition sweeps (per-task,
  per-definition measurement).
- Benchmark: cards emitted by hard-negative, calibration, and
  signal-bundle runners.
- Known limits: none structural.
- Book wording still accurate? YES.

### signal_bundle (Ch 15)

- Implementation status: IMPLEMENTED.
- Public object: `SignalBundle` (+ `build_signal_bundle`,
  `SignalProvenance`, `ExternalSignals`); geometric, calibration,
  and external evidence in separate fields; `available_signals`
  distinguishes missing from bad; no `query_difficulty` scalar by
  decision (individual signals preserved).
- Observatory surface: `inspect_result` (the Ch 15 line 243
  behavior — geometric block always, external only on
  escalation — is implemented; see Pass 4).
- Evidence source: Wave 1 row 1.12 (0.76 → 0.90 → 0.897).
- Benchmark: `benchmarks/signal-bundles/`.
- Known limits: perturbation-stability percentage (the book's 42%
  figure) has no dedicated evaluator; shared-neighborhood
  stability is the implemented proxy. Noted, not hidden.
- Book wording still accurate? YES, modulo pointing Ch 15 at the
  real `inspect_result` API (Pass 4 edit).

### space_comparison (Ch 16)

- Implementation status: IMPLEMENTED.
- Public object: `SpaceComparisonReport`, `GeometryComparison`,
  `CorrespondenceSet`, `compare_native_spaces`,
  `require_same_space_for_mixing`.
- Observatory surface: `compare_spaces` (evidence only, by
  construction).
- Evidence source: Wave 3 row 3.1 + Wave 6 (overlap 0.70–0.88,
  CKA 0.81–0.99, decision agreement high on clean queries).
- Benchmark: `benchmarks/cross-space/` (+ identity/permuted
  controls).
- Known limits: none structural.
- Book wording still accurate? YES.

### bridge_registry (Ch 20) / preservation_profile (Ch 21)

- Implementation status: IMPLEMENTED.
- Public object: `Bridge` (+Spec/provenance/controls/registry),
  `PreservationProfile` (+Result/Requirement/Policy/Verdict/
  ReferenceFrame), `usable_for`, `explain`.
- Observatory surface: `fit_bridge` (registered producers),
  `bridge_space`, `evaluate_bridge[_full]`,
  `evaluate_transformation`, lineage.
- Evidence source: Waves 3/6 (map ≠ preservation; no bridge wins
  every criterion; direction matters; VSP trade-offs).
- Benchmark: `benchmarks/bridges/`, `benchmarks/preservation/`.
- Known limits: MLP producers deliberately absent (negative
  result, documented); operating-point-parameterized
  `usable_for(scope, operating_point)` call form does not exist —
  operating points live in policies/requirements instead
  (RENAMED, see below).
- Book wording still accurate? MOSTLY — Ch 20 line 54's
  illustrative allowlist should now cite measured verdicts
  (Pass 4 edit). The `usable_for(scope, operating_point)`
  signature in Ch 20/26 prose should be reconciled to
  policy-carried operating points.

### compression_record (Ch 24)

- Implementation status: IMPLEMENTED via artifact + profile
  (legacy DTO retained).
- Public object: `TransformationArtifact` (PCA/random/prefix
  producers), `PreservationProfile`, `TransformationEvaluation`;
  legacy `CompressionRecord`/`check_compression` retained with
  fail-closed faithfulness semantics.
- Observatory surface: `evaluate_compression`,
  `derive_transformation_space`.
- Evidence source: Wave 2 retention knees (task-specific) +
  Wave 4 blind-spot matrix (drift 0% corruption detection; L1≠L5).
- Benchmark: `benchmarks/compression/` (knees 64/16/None).
- Known limits: claim-conditioned (L4) measurement exists in
  Wave-4 artifacts, not as a RELATE evaluator — correctly
  unpromoted; whitening tested, not shipped.
- Book wording still accurate? YES.

### transformation_record (Ch 25)

- Implementation status: IMPLEMENTED via selection framework
  (legacy DTO retained).
- Public object: `ContentTransformationCase`,
  `OperatorSelection`/`NONE_PASS`, complexity ranks, four
  producers; fidelity judged through `evaluate_transformation`
  under TARGET_NATIVE authority.
- Observatory surface: `evaluate_operator`.
- Evidence source: Wave 5 bakeoff (5 identity / 1 delta / 3 none;
  no mid-complexity passer).
- Benchmark: `benchmarks/operators/` (real DOC pairs, mirror
  encoder).
- Known limits: rank-1/low-rank/affine-general ladders stay
  benchmark-side; T.5 operator-keyed retrieval premise unmet
  (book states this).
- Book wording still accurate? YES; Ch 25 should adopt the
  `identity_map`-vs-semantic-identity and `NONE_PASS` vocabulary
  (Pass 4/5 edits).

### Observatory (Ch 26)

- Implementation status: IMPLEMENTED.
- Public object: `Observatory` (register/attach/inspect/compare/
  fit/derive/lineage/evaluate/inspect_result + CLI demo).
- Evidence source: composition of all of the above; replay hash
  test pins the lifecycle.
- Benchmark: `relate demo` (three cartridges, mixed verdicts).
- Known limits: no cost/migration estimator (book flags its
  figures schematic — correctly absent); no ANN instrumentation
  (index infrastructure, correctly absent); no plot outputs.
- Book wording still accurate? NO — this is the rewrite target.
  The chapter's reference-spec block, "reference design"
  disclaimers, and hypothetical registry/bridge/profile language
  describe as future what `9ac0dd1` implements. See rewrite plan.

## Fictional-architecture findings (Pass 4)

Concrete passages that must change from hypothetical to actual
(line references to `26-chapter.md`):

1. Line 204 + `26-concepts.txt` line 29 ("reference design",
   "not a production system"): RELATE 1.0 is a runnable,
   tested runtime with pinned replay hashes — not production
   infrastructure, but no longer a paper design. Rewrite to
   distinguish "runnable runtime" from "production deployment"
   (scale, latency, model coverage), which is the honest
   remaining gap.
2. The companion-component spec block ("reference specification
   ... not a claim that such an Observatory is already
   implemented"): now false. Replace with the real object map
   (this matrix) and real imports.
3. Ch 20 line 54 (`usable_for` illustrative allowlist "not a
   measured rung"): RELATE emits measured verdicts; cite
   `benchmarks/preservation/expected/summary.json` shape and
   `profile.explain()`.
4. Ch 15 line 243 ("The Observatory computes the geometric
   block..."): accurate behavior — attach the real API
   (`inspect_result`, `SignalBundle`, `available_signals`).
5. Ch 1 line 103 (RELATE repo = readout mechanism + demo):
   outdated. The repository is now the runtime; the readout is
   one capability. Rewrite the paragraph, keep the corpus-naming
   convention it establishes.
6. Ch 25 line 204 (operator-similarity precondition unmet):
   keep the logic, adopt `NONE_PASS` + `select_simplest_passing`
   vocabulary.
7. The eight illustrative Q&A outputs (Ch 26 lines ~75–106):
   keep the *form*, rebind each answer to its real call
   (`compare_spaces`, `evaluate_bridge_full`,
   `profile.usable_for/explain`, `lineage`).
8. "No generic second-model verifier would succeed" passages
   (Ch 26 line 198, Ch 15): consistent with `ExternalSignals`
   separation — no change needed; audit confirms alignment.

## RELATE-only concepts needing book homes (Pass 5)

- `CorrespondenceSet` → Ch 16 (comparison) + Ch 20 (anchors).
- Derived transformation identity + `bridge_output_space` →
  Ch 17 (versions) + Ch 20.
- `ReferenceFrame` (target/source/task-gold) → Ch 21 (the
  authority rule deserves its own callout).
- `ScopeVerdict`, structural WARN, fail-closed unknown scopes →
  Ch 21.
- `NONE_PASS`, `select_simplest_passing`, explicit complexity
  ranks → Ch 25.
- Transformation lineage + no-transitive-permission → Ch 17/26.
- Operator identity vs semantic identity → Ch 25.
- Preference-direction convention → Ch 4 (metrics) + Ch 15.
- Fit-vs-eval correspondence discipline → Ch 21.
- `available_signals` (missing vs bad) → Ch 15.
- Per-relation deltas that no overall number may hide → Ch 21.

## Chapter 26 rewrite plan (Pass 8)

Fourteen sections, each ending in runnable RELATE 1.0:

1. Why an embedding runtime is necessary (keep Ch 26 opening;
   RAG-objection stands).
2. The RELATE architecture (mermaid kept; boxes rebound to
   this matrix; "no new embedding method" stays).
3. Registering exact spaces (`SpaceIdentity`, registry,
   attach-refuses-spaceless).
4. Measuring native geometry (`describe_geometry`, sampling
   spec, erank/PR/TwoNN with estimator honesty).
5. Evaluating retrieval and hard negatives (generic evaluator,
   WIN/TIE/LOSS, RELATE adapter).
6. Calibrating operating points (three-way decisions,
   ambiguity, scope-bound thresholds, staleness-why).
7. Crossing spaces with bridges (directional producers,
   correspondence anchors, no-judgment contract).
8. Measuring preservation (results → frames → verdicts;
   `explain()` transcript as the section climax).
9. Compressing representations (cartridges, SOURCE_NATIVE
   authority, knees 64/16/None).
10. Testing semantic operators (content cases, ladder,
    `NONE_PASS` table).
11. Transformation lineage (derive/register/walk; permission
    never propagates — with the negative test quoted).
12. Scoped usability (`usable_for`/`explain`, fail-closed
    unknowns).
13. The complete Observatory run (`relate demo` transcript
    verbatim: bridge PASS/FAIL, compression PASS/FAIL,
    operator PASS/FAIL).
14. What RELATE still cannot establish (promote the current
    "does not establish" list, updated: no cost model, no ANN
    instrumentation, no production scale claims, mirror-not-
    reproduction status per BOOK-EVIDENCE-MAP.md, open
    hypotheses 2.5/T.5/3.2/3.8).

Demonstration: replace the composed Waves 1–4 session with the
`relate demo` transcript plus the preservation verdict table;
retain the invariants table with the measurement/policy split
marked per row (this audit's Pass 3).

## Surgical chapter list (Pass 8)

- Ch 1 §~103: RELATE project paragraph (repo is the runtime).
- Ch 4: preference-direction convention callout.
- Ch 15 line ~188: Relation Projection external-evidence note
  (repo grown; result status unchanged); line ~243: bind
  Observatory behavior to `inspect_result` API.
- Ch 16: `CorrespondenceSet` discipline note.
- Ch 17: derived identity + lineage + three-layer restatement
  (already aligned; extend, don't rewrite).
- Ch 20 line ~54: measured `usable_for` verdicts replace the
  illustrative allowlist.
- Ch 21: `ReferenceFrame` callout; `explain()` transcript;
  round-trip-evidence rule; per-relation visibility rule.
- Ch 24: cartridge/profile verdict language; L4 future-work
  pointer.
- Ch 25 line ~204: `NONE_PASS`/selection vocabulary; operator
  vs semantic identity callout.
- Concepts files: `26-concepts.txt` (reference-design lines),
  `01-concepts.txt` if it repeats the repo description,
  `20/21/25-concepts.txt` for the new vocabulary items.
- No chapter needs restructuring; all edits are paragraph/
  callout/transcript scope.
