# RELATE reference map

RELATE is the implementation and evidence reference for
*Embeddings From First Principles*.

The public pages explain the work at different distances from the code. This
repository keeps the runnable package, frozen corpora, benchmark families,
replay tools, and audit trail.

## Public references

- Book: [Embeddings From First Principles](https://programmer.ie/books/embeddings-from-first-principles/)
- Solution page: [RELATE](https://programmer.ie/solutions/relate/)
- Article: [RELATE: Searching Embeddings by Relation, Not Just Similarity](https://programmer.ie/post/relate/)
- Interactive demo: [RELATE on Hugging Face Spaces](https://huggingface.co/spaces/ernanhughes/relate-demo)

## Programmer.ie source

- Book source: [`content/books/embeddings-from-first-principles`](https://github.com/ernanhughes/next-books/tree/main/content/books/embeddings-from-first-principles)
- Solution source: [`content/solutions/relate.md`](https://github.com/ernanhughes/next-books/blob/main/content/solutions/relate.md)
- Article source: [`content/post/relate.md`](https://github.com/ernanhughes/next-books/blob/main/content/post/relate.md)

## Repository surfaces

- [`README.md`](../README.md): package overview, core usage, replay paths, and runtime doctrine.
- [`src/relate/`](../src/relate): the runtime package.
- [`corpus/README.md`](../corpus/README.md): RELATE and RELATE-DOC corpus ownership, releases, hashes, and invariants.
- [`benchmarks/README.md`](../benchmarks/README.md): capability-organized benchmark families.
- [`evidence/README.md`](../evidence/README.md): evidence ledger and claim boundary.
- [`evidence/RELATE-1.0-AUDIT.md`](../evidence/RELATE-1.0-AUDIT.md): reconciliation verdict for the book and runtime.
- [`evidence/BOOK-EVIDENCE-MAP.md`](../evidence/BOOK-EVIDENCE-MAP.md): benchmark families mapped back to book claims.
- [`docs/scientific-state.md`](scientific-state.md): retained scientific claims and boundaries from the earlier projects.
- [`docs/hugging-face-space.md`](hugging-face-space.md): how the public Space demo is built and what it does not prove.

## Reading order

1. Read the book for the full argument: why embedding geometry needs identity,
   calibration, cross-space comparison, preservation profiles, and scoped
   usability.
2. Read the solution page for the applied summary: what problem RELATE solves
   and how the runtime answers it.
3. Read the article for the original relation-search result and historical
   motivation.
4. Use this repository to inspect the implementation, evidence boundaries,
   corpus releases, and benchmark artifacts.

## Boundary

The book is the explanation layer. The solution page is the applied public
entry point. The article is historical context for the original relation-aware
search result. This repository is the reference implementation and evidence
surface.

When these disagree, treat the repository evidence files as the authority for
what has actually been implemented and measured.
