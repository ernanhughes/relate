# RELATE v0.2 — the hard query set

`relate-0.2.0/` is **`relate-0.1.0` plus a harder query set**. The item pool and
the typed pairs are **byte-identical to v0.1** (`items.jsonl`, `pairs.jsonl` diff
clean); only `queries.jsonl` grows.

- v0.1 queries: 269 (mostly near-restatements of the answer sentence)
- v0.2 adds: **142 hard queries** — indirect phrasing with a genuine
  lexical / semantic gap to the answer, 5 competing distractors each (vs ~2.5 in
  v0.1), covering: role disambiguation ("which firm ended up owning the other"),
  claim-strength ("for the whole group, not just a subset"), temporal
  qualification ("the most recent count", "not the earlier ones"), and indirect
  reference ("where does {country} run its ministries from").
- Total: 411 queries.

**Why it exists.** Wave 1 found v0.1 is an excellent *pair-level* diagnostic but
its query set is too easy for *system-level* comparisons — several Wave-1 rows
saturated (row 1.9: one model won every relevance definition; nDCG@10 pinned at
0.93–0.95). v0.2's mission is narrow: create enough query→answer distance that
retrieval policies, rerankers, diagnostics, and model comparisons stop
saturating. It is **not** a general benchmark.

**Corpus hash:** `corpus_hash.txt`. Deterministic rebuild:
`python ../build_full.py --v02 --check`.

**Re-run of the saturated Wave-1 rows on v0.2:** `../../wave1/artifacts/v02/`
(routed there by `RELATE_RELEASE=relate-0.2.0`). See `../../wave1/WAVE1_FINDINGS.md`
§v0.2.
