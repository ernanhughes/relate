# Historical Option B replay checkpoint

Date: 2026-08-05

## Status

```text
OPTION_B_HISTORICAL_REPLAY_COMPLETE
```

The preserved historical Option B result was reconstructed from the two independently generated local SQLite embedding caches and the committed canonical artifacts in `similarity_is_relative`.

No embedding was regenerated. No probe was refitted. No threshold, selected row, hard-negative pair, method, or decision rule was changed.

## Exact result

| Method | Query-equal hard-negative accuracy |
|---|---:|
| Raw cosine | `0.532458984375` |
| Raw Euclidean | `0.533314453125` |
| Token length | `0.49868359375` |
| True primitive oracle | `1.0` |
| Predicted primitive executor | `0.7328515625` |

```text
raw_best = 0.533314453125
predicted_executor_gap = 0.19953710937500002
threshold = 0.1
outcome = REAL_PREMISE_SUPPORTED
```

## Reconstructed embeddings

Both copied caches reconstructed the same arrays element-for-element.

| Split | Shape | Array SHA-256 | Cache A/B exact |
|---|---:|---|---|
| Train | `20000 × 768` | `ac23f6b58076c4f5aaa5eeb44046480ebafe93793a100abd764749d3f5c13767` | yes |
| Validation | `4000 × 768` | `7e74953f4bb69d598b663b88e6e42b6a1105bd75ebdc264ccbbd0300346fe242` | yes |
| Test | `4000 × 768` | `e24c9797f5068e9c84a170bacdc85e06a1c4fe6bbc2d7b1f64045bc166ef4320` | yes |

The replay evaluated all `4,000` test queries and all `512,000` frozen hard-negative pairs. No query lacked pairs.

## Exact verification

```text
primary_query_score_array_sha256 = dccf0698934142ceaf1fe0ccd5d35713600ef45f9719e3864468c40a5274dc70
published_verification_file_sha256 = da1b9cf1244b47c71ac7adce91b7db502b4fd2d3b663e126d8cde7c87e239d6c
exact_decision_match = true
```

The complete replay decision object exactly matched the previously published independent Option B verification.

The machine-produced replay record is preserved in:

```text
docs/results/option-b-historical-replay-2026-08-05.json
```

## Scientific boundary

This checkpoint supports only the historical Option B result:

> Projected relation coordinates recovered from frozen CodeBERT embeddings ordered the frozen real-code hard negatives substantially better than raw cosine or raw Euclidean distance.

It does not alter or reopen RELATE-E01. RELATE-E01 remains:

```text
EXPERIMENT_INVALID
```
