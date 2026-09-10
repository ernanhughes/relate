# Geometry benchmark

Shape differs across spaces; invariants hold exactly. No semantic claim is
made about any space -- geometry diagnoses geometry first.

## Spaces (seeded, n=1200, nominal d=256)

| space | random-pair cosine | eff. rank | part. ratio | TwoNN ID |
|---|---|---|---|---|
| ISO (isotropic) | ~0.00 | ~230 | ~211 | ~90 |
| ANISO (concentrated) | ~0.49 | ~96 | ~34 | ~42 |
| LOWRANK (6-dim + noise) | ~0.00 | 6.0 | 5.9 | 5.3 |

Two mirrors of the book: raw cosine magnitude is space-relative (0.00 vs
0.49, cf. the 0.06 vs 0.34-0.45 spread across encoders), and nominal
dimension is not effective dimension (256 nominal, single-digit summaries
for the low-rank space).

## Invariants (`invariants.json`, asserted in `run.py`)

- Rotation (`X @ Q`): cosine mean/std drift < 1e-9, effective-rank ratio 1.
  Coordinates are not semantic.
- Scaling (`c * X`): cosine unchanged, norms scale by `c`. Metric choice is
  part of the measurement definition.
- Translation (`X + b`): cosine moves substantially on the concentrated
  space.

## Running

```text
python benchmarks/geometry/run.py          # write expected/
python benchmarks/geometry/run.py --check  # verify byte-identical
```

Sampling is deterministic (`PairSamplingSpec`, seed recorded in every
report); TwoNN carries its method name, sample count, and parameters
because an intrinsic-dimension estimate is estimator-dependent, never a
property of the space.
