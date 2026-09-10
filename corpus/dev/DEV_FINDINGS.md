# RELATE-DEV findings — ontology & authoring defects surfaced

**Build:** `dev/dev_source.py` → `dev/build/` (54 items, 36 pairs, 3 queries, all 11 relations, all 5 domains).
**Date:** 2026-09-08. **Result:** clean after the fixes below (0 errors, 2 DEV-stage strata warnings).

The point of RELATE-DEV (spec build step 3) is to hit ontology problems while
corrections are cheap. Six were found.

---

## 1. The negation ≥ 0.7 lexical-overlap constraint needs a lemmatizer

**What happened.** Valid, minimal negation pairs were rejected:
- `The committee rejected the proposal.` / `The committee did not reject the proposal.` — English do-support forces `reject` (not `rejected`), so lemmatizer-free content-token Jaccard sees `rejected` ≠ `reject` and drops to 0.5.
- `Acme Corp acquired Beta Systems…` / `…did not acquire…` — `acquired` ≠ `acquire`.

**Root cause.** Spec §5 defines `lexical_overlap` over *lemmatized* content tokens; the DEV approximation had no lemmatizer, so it *under*-estimates overlap and *false-rejects* clean negations.

**Fix.** Added a light suffix-stripper (`lexical._stem`) as a lemmatizer stand-in (`reject`/`rejected` → `reject`, `acquire`/`acquired` → `acquir`, trailing-`e` collapse). Documented that the frozen build uses a real lemmatizer.

**Spec consequence.** None to the ontology. Note for the frozen build: the `negation` overlap check must run *after* lemmatization, and the generator should verify each negation is a **minimal edit** (insert/remove a negation marker on the main predicate; do not also drop clauses).

---

## 2. Negation of a multi-claim sentence must negate the whole assertion

**What happened.** `The drug lowered blood pressure and improved sleep quality in elderly patients.` was "negated" to `The drug did not lower blood pressure in elderly patients.` — which only negates the first conjunct and silently drops the second. Overlap fell to 0.67 and, more importantly, it is not a clean negation.

**Fix.** Rewrote to `The drug did not lower blood pressure or improve sleep quality in elderly patients.` (negates the conjunction, keeps all content).

**Spec consequence.** Add to §3.6 / §6: when the source has ≥ 2 atomic claims, a structured-perturbation `negation` must negate the top-level assertion (De Morgan the conjunction), not a single conjunct. A single-conjunct edit is `contradiction`, not `negation`.

---

## 3. `entity-related` needs a *named* entity — "the app" / "the printer" in product-support

**What happened.** `the app crashes on launch` / `the app's reports screen has a CSV export button` was labeled `entity-related`, but neither item declared a named entity, so `entity_overlap = 0` and the constraint (`entity_overlap ≥ 1`) failed.

**Fix.** In `product-support`, the product itself (`the printer`, `the app`) is treated as the domain's named entity and put in each item's `entities` list.

**Spec consequence.** Add to §2 / §3.8: in `product-support`, the product under discussion counts as the shared named entity for `entity-related` and for hard-negative entity matching. `geo-civics` / `corporate-events` / `biomed-claims` keep the strict "proper noun" reading.

---

## 4. Adjudication precedence is subtle — "lower chamber" does not entail "two chambers"

**What happened.** `The lower chamber of the parliament has 300 seats.` was paired as `entailment → The national parliament has two chambers.` It entails there is *a* chamber, not *two*. Wrong relation.

**Fix.** Removed the entailment pair; kept only the `topic-related` pairing (independent claims about the same parliament).

**Spec consequence.** None — the adjudication order already handles it (entailment fails its decision procedure, so it falls through to `topic-related`). This is a reminder that the *annotator* must actually run the decision procedure, and that `runner_up` should record the relation that was *almost* right (here: entailment).

---

## 5. The validator cannot catch a semantically bogus `partial-support` pair

**What happened.** `partial-support`: a = "The drug lowered blood pressure and improved sleep quality in elderly patients" (2 atomic claims, so the structural check passes), b = "The compound was evaluated in a mouse model of cancer" (different entity, different everything). Structurally valid, semantically nonsense.

**Fix.** Removed the pair; added a real one (`The 1957 treaty created a customs union and abolished internal tariffs` / `Under the 1957 treaty, tariffs … were removed`).

**Spec consequence.** `partial-support` is the relation most dependent on human adjudication: the validator can only check `a` has ≥ 2 atomic claims, not that `b` actually entails one of them. The freeze-time datasheet must report `partial-support` inter-annotator α separately and flag it if low (spec §9 already asks for per-relation α; this makes `partial-support` a priority to watch).

---

## 6. Overlap-strata coverage is thin at DEV scale (expected)

**What happened.** DEV warnings: `entailment` has no low-overlap example; `topic-related` has no high-overlap example. Spec §5 wants ≥ 25% of each balanced relation's pairs in each of the low/mid/high bands.

**Fix.** None at DEV scale — the warning is informational and says "must hold at freeze." The generator (step 6, "balance by difficulty") must actively fill each band; a low-overlap `entailment` (very different wording, still strict entailment) and a high-overlap `topic-related` (near-identical wording, logically independent claims) are the harder cells to author and should be planned for.

---

## Net effect on the ontology

`ontology.json` was **not** changed by these findings. Three spec clarifications
are queued for `spec/RELATE-CHANGELOG.md` when the full build starts:

1. §3.6 / §6 — `negation` overlap is computed post-lemmatization; a structured
   `negation` of a multi-claim sentence must negate the top-level assertion.
2. §2 / §3.8 — in `product-support` the product is the shared named entity.
3. §9 — `partial-support` inter-annotator α is a priority metric; the validator
   cannot substitute for adjudication there.
