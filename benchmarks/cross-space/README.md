# Native cross-space benchmark

What relationships between the same items survive when the representation
changes -- never "are these spaces compatible".

## Conditions

| condition | CKA | 10-NN overlap | counterpart top-1 | hard-neg Δ |
|---|---|---|---|---|
| identity (X vs X) | 1.000 | 1.000 | 1.000 | 0.0 |
| native pair (X vs Y) | 0.948 | 0.513 | 1.000 | −0.123 |
| permuted (chance floor) | 0.006 | 0.005 | 0.000 | — |

The native pair is a direct same-dimension comparison with no fitted map:
decent coarse geometry is evidence, never authorization for mixing (the
mixing guard is asserted denied in `run.py`).

## The thesis shape, per distinction

```text
negation            degraded  (delta < -0.10)
temporal-mismatch   degraded  (delta < -0.10)
relation-swap       degraded  (delta < -0.03)
topic-related       preserved (|delta| < 0.05)
threshold transfer  FAIL      (FAR 0.10 -> ~0.19 on Y)
```

Polarity/time distinctions carried by Y's weakened dims collapse while
content-level distinctions survive -- Part VI made executable. The
`SpaceComparisonReport` holds geometry, neighborhoods, counterpart
recovery, and the hard-negative delta together without reconciling them
into one score, and carries the correspondence hash plus scorer identity
as provenance.

## Running

```text
python benchmarks/cross-space/run.py          # write expected/
python benchmarks/cross-space/run.py --check  # verify byte-identical
```

Row order never defines correspondence (`CorrespondenceSet` is resolved
explicitly and hashed); calibration transfer uses 3C machinery in this
runner, never inside the comparison evaluator. No bridge is fitted
anywhere in this benchmark -- 4B will hand its candidates to this exact
path.
