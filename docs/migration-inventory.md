# Migration Inventory

The new `ernanhughes/relate` repository is not a fork or continuation of either source repository.

Migration means extracting small, understandable scientific components with explicit provenance. It does not mean copying either tree wholesale.

## Source repositories

### Original failed project

```text
ernanhughes/similarity_is_relative
reference head: cb78933cde8eed2dc2490ed44919dd6b8b6b10de
```

### Second failed attempt

```text
ernanhughes/relate_attempt
reference head: fef0c2eeab45b369a26345759eafffc7c6747825
```

## Information already migrated

- the central research question;
- the exact bounded Option B claim;
- Option B primary measurements and limitations;
- the stronger supervised-comparison question from the second attempt;
- RELATE-E01's terminal `EXPERIMENT_INVALID` result;
- the lesson that the single-seed shuffled-control gate should not be reused.

See [`scientific-state.md`](scientific-state.md).

## Code candidates from `similarity_is_relative`

These files contain potentially reusable scientific logic. They are candidates for extraction, not verbatim import.

| Source file | Useful content | Migration action |
|---|---|---|
| `src/relate/experiments/option_b_real_code.py` | AST parsing and the three objective structural coordinates | Extract into a small `primitives.py` module with focused tests |
| `src/relate/experiments/option_b_selection_v2.py` | deterministic row selection and split discipline | Extract only the deterministic selection rules needed by the next experiment |
| `src/relate/experiments/option_b_embeddings.py` | frozen CodeBERT encoding and mean pooling | Extract a minimal encoder adapter; do not import cache/evidence machinery |
| `src/relate/experiments/option_b_probe_runner.py` | independent Ridge probes and validation selection | Extract the fitting logic and tie rule |
| `src/relate/experiments/option_b_predicted_executor.py` | predicted-coordinate executor and out-of-fold candidate predictions | Extract the mathematical core |
| `src/relate/experiments/option_b_hard_negative_manifest.py` | deterministic hard-negative construction | Retain only if the next experiment uses the same evaluation design |
| `src/relate/experiments/option_b_method_evaluation.py` | cosine, Euclidean, executor metrics and query aggregation | Extract reusable metric functions, not the monolithic runner |
| `src/relate/experiments/option_b_method_evaluation_independent.py` | separate primary-metric recomputation | Preserve as a design pattern; rewrite after the next experiment is fixed |

## Code candidates from `relate_attempt`

| Source file | Useful content | Migration action |
|---|---|---|
| `src/relate/experiments/models.py` | shared triplet scoring and supervised baseline definitions | Extract model-independent scoring first; review model implementations separately |
| `src/relate/experiments/development_training.py` | fair training access, validation selection, M3/M4/M5 comparison | Extract only reusable training loops and registered tie behaviour |
| `src/relate/experiments/first_experiment.py` | stronger baseline framing and decision structure | Convert into a short design note, not executable inherited protocol |
| `src/relate/experiments/shuffled_control.py` | implementation of the failed control | Do not reuse as a gate; retain only for understanding the failure |

## Documents worth keeping as references

From `similarity_is_relative`:

- `docs/experiments/08-option-b-real-code-premise-test.md`
- `docs/results/option-b-real-code-premise-checkpoint-v1.md`
- `docs/research/cosine-failure-and-relational-recovery.md` as an idea source only
- the compact Option B evaluation and independent-recomputation records

From `relate_attempt`:

- `docs/research-question.md`
- `docs/first-experiment.md`
- `experiments/RELATE-E01/README.md`
- the compact E01 terminal result and development model-selection summary

These should normally be linked to or summarized. They should not all be copied into the active repository.

## Material deliberately left behind

The following categories are not part of the new repository's starting point:

- capability systems;
- authorization phrases and one-shot marker machinery;
- artifact access ledgers;
- generalized cache provenance frameworks;
- static enforcement scanners;
- publication authorization systems;
- claim-management frameworks;
- review packages and hostile-review transcripts;
- agent workflows;
- architecture migration records;
- Option C and Option E programme machinery;
- large canonical embedding and pair artifacts;
- historical experiment numbering;
- historical tests that do not test extracted active modules.

Leaving this material behind is intentional. It remains available in the source archives.

## Extraction standard

A source module moves into the new repository only when all of the following are true:

1. it directly supports the next finite experiment;
2. its purpose can be explained in one paragraph;
3. its dependencies are explicit;
4. it can be covered by focused unit tests;
5. it does not pull in abandoned infrastructure;
6. its origin is recorded in the module docstring or migration notes;
7. it is understandable without reading either source repository.

## Initial target structure

```text
README.md
docs/
  scientific-state.md
  migration-inventory.md
  next-experiment.md
src/relate/
  primitives.py
  embeddings.py
  probes.py
  scoring.py
  baselines.py
tests/
  test_primitives.py
  test_scoring.py
  test_probes.py
```

This structure is provisional and may become smaller. It may not become larger before the next scientific experiment is stated clearly.

## Migration stopping rule

The migration ends when the new repository contains enough code and documentation to run one finite comparison.

It does not require reproducing either old repository, preserving all capabilities, or resolving every historical concern.
