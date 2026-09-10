# Bridge-producer benchmark

Deliberately boring producers, judged exclusively through the 4A path.
No table here is calculated by `bridges/`.

## Producers (fit on train anchors, judged on held-out eval rows)

| producer | counterpart top-1 | 10-NN overlap | hard-neg Δ vs native target |
|---|---|---|---|
| procrustes | 1.000 | 0.596 | −0.122 |
| linear | 1.000 | 0.794 | −0.035 |
| ridge | 1.000 | 0.788 | −0.032 |
| identity (no-op control) | 1.000 | 0.596 | −0.122 |
| constant_target_centroid (floor) | 0.002 | 0.013 | +0.613 |
| random_map (seeded) | 0.005 | 0.386 | −0.120 |

Delta convention (uniform with 4A): target-native minus candidate. A
*negative* delta means the candidate beats the native target -- expected
for fitted maps, which carry X's fine signal that Y never had. A
*positive* delta means the candidate is worse (the centroid floor:
every margin ties at zero). Procrustes coincides with identity here
because the shared nuisance backbone dominates and the rigid fit cannot
express dim-weakening; linear/ridge buy neighborhood fidelity by
relaxing orthogonality. Fixture-specific, honestly so.

Training reconstruction (cosine, labeled FIT DIAGNOSTIC in
`fit-diagnostics.json`) is near-perfect for fitted maps and must never
be read as preservation evidence: train and eval correspondences are
distinct objects with distinct hashes (asserted in `run.py`).

## Derived identity

Every candidate set receives a derived space (`derived-identities.json`)
with parent = source, bridge id, and target reference -- asserted
different from the native target hash in `run.py`. Coordinate
compatibility is not representation identity.

## Running

```text
python benchmarks/bridges/run.py          # write expected/
python benchmarks/bridges/run.py --check  # verify byte-identical
```

Deleting `src/relate/bridges/` would remove every producer above but
none of the machinery that judged them: fits, controls, and candidates
enter through one affine contract (`transform`), and all verdicts come
from `compare_native_spaces`.
