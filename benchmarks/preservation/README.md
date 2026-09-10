# Preservation benchmark

A map proves coordinates can be transformed. A preservation profile
determines what that transformation is actually good for. Direction
under test: coarse source into fine target -- candidates inherit the
source's blindness, so fine distinctions fail while coarse retrieval
transfers.

## Verdicts (declared policies in `policies.json`)

| producer | retrieval | threshold_transfer | neighborhood_use |
|---|---|---|---|
| procrustes | PASS | FAIL | PASS |
| linear | PASS | FAIL | PASS |
| ridge | PASS | FAIL | PASS |
| identity (no-op) | PASS | FAIL | PASS |
| constant_target_centroid | FAIL | FAIL | FAIL |
| random_map | FAIL | FAIL | FAIL |

No producer passes every scope; controls fail retrieval outright.
Fitted maps and the no-op agree coarsely (counterpart recall@1 1.00,
neighborhood ~0.60) yet all fail threshold transfer (e.g. ridge FAR
0.10 → 0.12 with FRR blowing past its bound) -- retrieval PASS
coexisting with transfer FAIL is the result, not an inconsistency.

## Per-relation visibility

Overall hard-negative accuracy never hides a failed class: every
profile carries per-relation deltas (native target minus candidate),
all positive here because candidates lack the fine target's signal:

```text
negation            +0.20 to +0.23 (failed distinction, visible per producer)
topic-related       ~0.00 (preserved)
```

## Evidence discipline

- Results come from 4A comparison, 3C transfer reuse, and reverse-fit
  round-trips; `preservation.py` computes no metric.
- Round-trip results carry the source-native frame and gate no
  target-scope verdict.
- The centroid's constant candidates cannot even be calibrated:
  transfer is unmeasured and fails closed (never silently passed).
- Fit diagnostics (train reconstruction) live in 4B, never in profiles.

## Running

```text
python benchmarks/preservation/run.py          # write expected/
python benchmarks/preservation/run.py --check  # verify byte-identical
```

`explain(scope)` on any profile renders the failed gates with their
bounds, e.g. `FAIL threshold_transfer: failed:
calibration_transfer/frr_increase (max_delta 0.05, gate not met)`.
