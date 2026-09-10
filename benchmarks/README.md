# Benchmarks (capability-organized, not wave-organized)

- `relation-search/`:hard-negative ordering, raw geometry vs relation readout.
- `retrieval/`: margins, density, hubs, stability, signal bundles.
- `calibration/`: distributions, FAR/FRR, ambiguity regions, stale thresholds.
- `bridges/`: Procrustes/Ridge/linear, preservation profiles, usable_for.
- `compression/`: retention vs faithfulness (geometry != verification).
- `transformations/`: operator classes IDENTITY / CONSTANT_DELTA / NONE.

Each benchmark ships a manifest (corpus release hash + space hashes + code
version) so a result is reproducible without reopening history.
