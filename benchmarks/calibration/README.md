# Calibration benchmark

A globally respectable discriminator can still be operationally poor
because the safe accept/reject regions are small.

## Regimes (seeded Gaussian preference scores, n=1500/side)

| regime | AUC | EER | ambiguity @ FAR/FRR ≤ 0.10 |
|---|---|---|---|
| ordinary (separated negatives) | 0.925 | 0.159 | 0.126 |
| hard (overlapping negatives) | 0.774 | 0.296 | 0.478 |

Same scorer family, different negative distributions, different operating
points -- calibration belongs to a distribution, not to a model name.
The hard regime mirrors the Wave-1 pattern (AUC ≈ 0.75, EER ≈ 24-30%,
large escalate band) without pretending the constants are universal.

## Domain shift

Same positives, negatives shifted +0.25: the EER threshold moves +0.109
(cf. the book's ~0.10 cross-domain spread). Domain is carried as
`CalibrationScope`, so finance/legal/general calibrations are distinct
records -- scope is data, not branching. `staleness_against` names the
changed dimensions (`space_hash`, `domain`, `scorer`) instead of merely
flagging stale.

## The 3A seam

`run.py` also feeds real `HardNegativeReport` observations into
`extract_margin_distributions` → `calibrate`: one definition of margin,
no recomputation. Geometry/neighborhood code is never imported here;
their signals could only enter as explicitly supplied scores.

## Running

```text
python benchmarks/calibration/run.py          # write expected/
python benchmarks/calibration/run.py --check  # verify byte-identical
```

Higher score always means more positive (3A preference convention);
distances enter negated. `CalibrationDecision` has no `is_match`
shortcut. `evaluation-card-hard.json` carries the fit in an
`EvaluationCard` -- no parallel evidence model.
