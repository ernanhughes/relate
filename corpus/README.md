# RELATE corpus (canonical ownership lives here)

`relate` owns the RELATE diagnostic corpus: ontology, generators, validators,
releases, manifests, content hashes, task views and datasheets.

Canonical frozen releases (currently built from
`C:\Projects\new-books\experiments\embeddings-from-first-principles\relate`):

- `relate-0.1.0`: 1,173 items / 1,181 typed pairs / 269 queries / 11 relations
  (`corpus_hash 8cad6816…9589b3`)
- `relate-0.2.0`: v0.1 byte-identical + 142 hard queries (411 total)
- `relate-doc-0.1.0`: 45 structured documents, 595 compressions, 260
  transformation pairs

Migration rule: `next-books` experiments consume tagged RELATE releases; they
do not maintain a second canonical implementation. Until the vendor step
lands the JSONL releases under `corpus/releases/`, this directory records the
ontology copy plus the release pointer file.
