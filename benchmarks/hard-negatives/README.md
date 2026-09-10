# Hard-negative benchmark

One reusable way of asking: *when two items are deceptively close in
embedding space, does this method preserve the distinction that matters?*

Dependency direction (kept clean on purpose):

```text
src/relate/evaluation/       generic science (cases, observations, reports,
                             scorer injection, baselines, deltas, cards)
benchmarks/hard-negatives/   RELATE-specific experiment (this directory)
corpus/                      diagnostic instrument (frozen views + releases)
```

No corpus loader lives in `src/relate`. No cosine/RELATE special case lives
in the evaluator.

## Running

```text
python benchmarks/hard-negatives/run.py          # write expected/
python benchmarks/hard-negatives/run.py --check  # verify byte-identical
```

## Halves

1. **Adapter half** (`relate_cases.py`): the frozen
   `relate-0.1.0/views/hard-negative-v0.1.json` view becomes 937 generic
   cases (query anchor should prefer its grade-3 answer over the deceptive
   negative). Records `relate-cases-summary.json`; fails if the frozen
   corpus hash, case count, or relation distribution ever changes.
2. **Scoring half** (`synthetic_vectors.py` + `run.py`): a deterministic
   synthetic mirror of the Python-structure experiment runs cosine,
   Euclidean, random, and the ridge relation readout through the SAME
   evaluator. Writes `report-*.json`, `delta-relation-vs-cosine.json`,
   `evaluation-card-relation.json`, and `provenance.json` (space_hash,
   corpus hash/version, scorer identity, seeds, code version).

## On the historic numbers

`expected/historic-reference.json` holds the preserved 0.733 / 0.532 / 0.533
figures as EXTERNAL evidence with their provenance. They are not recomputed
here: the CodeBERT assets are absent from this machine. What IS reproduced
through the new evaluator is the observation structure — supervised readout
far above raw geometry, cosine coinciding with Euclidean:

```text
mirror     relation 0.725   cosine 0.100   euclidean 0.100   random ~0.485
historic   relation 0.733   cosine 0.532   euclidean 0.533
```

The mirror's raw-geometry figures are harsher because its negatives are
adversarially mined (max cosine among latent-far items) — the hard-negative
regime RELATE targets — while the historic pairs were fixed. The shared,
load-bearing facts survive intact.

## What is deliberately absent

Calibration logic, bridge-specific baselines/evaluators, sklearn. Those
arrive in later slices; `compare_reports` is already the primitive that
native-vs-translated comparisons will reuse.
