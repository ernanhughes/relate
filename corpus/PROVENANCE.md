# Corpus provenance (Step 2 milestone)

Canonical owner: the RELATE project (`github.com/ernanhughes/relate`, this
directory). Release tag: `step2-corpus-canonical`.

## Vendored from

`next-books/experiments/embeddings-from-first-principles/relate/` on
2026-09-10, copied byte-identically with `robocopy /E` excluding only
`__pycache__`, `*.pyc`, `.pytest_cache`, `claims.md` (the book's claim ledger,
which stays with the book) and `.gitignore` files. The only intentional
content edit in this tree is the ownership banner at the top of `README.md`;
every other file is byte-identical to the source (audited by sha256 over all
relative paths: 0 mismatches, 0 missing, 0 extra).

`ontology.json` was already vendored in Step 1 and re-verified identical.

## Frozen releases

| Release | `corpus_hash.txt` | Contents |
|---|---|---|
| `relate-0.1.0` | `8cad6816d90e06bc49e5b0b64cd460945e17ea4b1ec3c46054409669eda525b3` | 1,173 items / 1,181 pairs / 269 queries / 11 relations |
| `relate-0.2.0` | `695654cbd0bdadcbe0140f85927ad8799a6a3306ab431bc1ec48fdec195f7719` | v0.1 items+pairs byte-identical + 142 hard queries (411 total) |
| `doc/relate-doc-0.1.0` | `111cc2bb9008557d632061f98a0e4f847b91293e31bcdb2d78285a35de7f7914` | 45 documents / 595 compressions / 260 transformation pairs |

## Verification gates (all run from this directory, stdlib only)

- `python build_full.py --check` → MATCH (`8cad6816…`)
- `python build_full.py --v02 --check` → MATCH (`695654cb…`)
- `python doc/build_doc.py --check` → MATCH (`111cc2bb…`)
- `python validate.py relate-0.1.0` → 0 errors, 0 warnings
- `python validate.py relate-0.2.0` → 0 errors, 0 warnings
- `python tests/test_validators.py` → 16/16 passed
- `python -m pytest corpus/tests/test_releases.py` → manifest sha256, counts,
  views, and v0.2⊃v0.1 byte-identity all pass

Runtime tier separation: `pytest` at the repo root collects `tests/` only
(31 tests, NumPy-only, no corpus assets required). Corpus tests are the
separate tier under `corpus/tests/`.

## Consumer rule

`next-books` experiments consume tagged RELATE releases and record the
**RELATE release tag + corpus hash** in their provenance — never a relative
path into the book repo as the source of truth. The historical experiment-tree
copy is retained until waves re-verify against this tree; it carries
`CANONICAL-SOURCE.md` pointing here.
