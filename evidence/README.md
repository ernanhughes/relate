# Evidence ledger

Every operation involving an embedding must know which space it belongs to,
which relation is being asked about, and what evidence authorizes it.

- `claims.md`: stable public concepts and their measured status.
- `measured-results.md`: pointers to measured artifacts (book waves + replays).
- `manifests/`: reproducible manifests for benchmarks and replays.

Claim boundary: the original result (supervised relation readout ~73.3%
ordering accuracy vs ~53.3% raw cosine/Euclidean) is preserved byte-for-byte
by `RelationProjection` replays. New capabilities cite their own records;
they never rewrite that result.
