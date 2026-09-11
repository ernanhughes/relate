# Evidence ledger

Every operation involving an embedding must know which space it belongs to,
which relation is being asked about, and what evidence authorizes it.

- Public book: [Embeddings From First Principles](https://programmer.ie/books/embeddings-from-first-principles/)
- Public solution: [RELATE](https://programmer.ie/solutions/relate/)
- Public article: [RELATE: Searching Embeddings by Relation, Not Just Similarity](https://programmer.ie/post/relate/)
- Repository guide: [`docs/reference-map.md`](../docs/reference-map.md)

- `RELATE-1.0-AUDIT.md`: reconciliation verdict for the runtime and book.
- `BOOK-EVIDENCE-MAP.md`: benchmark families mapped back to book claims.
- `CHAPTER-26-RECONCILIATION.md`: book edits identified by the runtime audit.
- `manifests/`: reserved for reproducible benchmark and replay manifests.

Claim boundary: the original result (supervised relation readout ~73.3%
ordering accuracy vs ~53.3% raw cosine/Euclidean) is preserved byte-for-byte
by `RelationProjection` replays. New capabilities cite their own records;
they never rewrite that result.
