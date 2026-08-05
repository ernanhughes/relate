# RELATE

**Search frozen embeddings by a relation they contain, not only by the cosine geometry they expose by default.**

RELATE is deliberately small. It takes embeddings you already have, learns a projection into one or more measurable relation coordinates, and ranks targets in that relation space.

```python
import numpy as np
from relate import RelationProjection

model = RelationProjection.fit(
    training_embeddings,
    training_relation_coordinates,
    relation_names=("complexity", "depth", "call_sites"),
)

hits = model.search(
    source_embedding,
    target_embeddings,
    k=10,
)

for hit in hits:
    print(hit.index, hit.relation_distance, hit.cosine_distance)
```

That is the implementation. There is no experiment framework inside the package.

## Why this exists

Cosine similarity collapses an embedding into one fixed notion of closeness. A frozen embedding can contain information that is predictable from its coordinates but poorly exposed by that default geometry.

The original real-code result used frozen CodeBERT embeddings and three objective Python AST coordinates:

1. cyclomatic complexity;
2. maximum control nesting depth;
3. distinct call-site count.

A ridge projection recovered those coordinates from the embeddings. Chebyshev distance in the recovered coordinate space reached `0.7328515625` hard-negative ordering accuracy, compared with `0.532458984375` for raw cosine and `0.533314453125` for raw Euclidean distance.

The product claim is intentionally narrower than the old research programme:

> Given useful relation labels, a small learned projection can expose relation-specific information in frozen embeddings that cosine search misses.

## Cosine can still help

For a large target set, cosine can cheaply generate candidates and RELATE can rerank them:

```python
hits = model.search(
    source_embedding,
    target_embeddings,
    k=10,
    candidate_pool=500,
)
```

This does not blend unlike distance units. Cosine chooses a broad candidate pool; the learned relation defines the final order.

## Python structural coordinates

The three coordinates from the successful code result are included as a small adapter:

```python
from relate import PYTHON_RELATION_NAMES, extract_python_structure

training_relation_coordinates = np.vstack(
    [extract_python_structure(source).as_array() for source in training_functions]
)

model = RelationProjection.fit(
    training_embeddings,
    training_relation_coordinates,
    relation_names=PYTHON_RELATION_NAMES,
)
```

Embedding generation is intentionally external. RELATE accepts NumPy-compatible arrays from any encoder.

## Replay preserved assets

The old benchmark SQLite database and preserved NPZ/NPY arrays can be inspected and replayed without generating a single new embedding.

First inventory the local assets:

```powershell
python -m relate.inventory C:\Projects\relate `
    --output C:\Projects\relate\replay-inventory.json
```

This opens `.writer/benchmarks/embedding-cache.sqlite3` read-only and records every embedding contract, row count, dimension, NPZ key, array shape, dtype, model identifier, dataset hash, and standalone NPY file it can find. Large array values are not loaded merely to produce the inventory.

The SQLite database contains text and embedding vectors. It does not contain the frozen pair labels and split assignments. A faithful PAWS or BigClone replay therefore uses the per-run `real_embeddings.npz` together with its sibling `manifests` directory:

```powershell
python -m relate.replay replay-pairs `
    --snapshot experiments\benchmarks\outputs\paws\mxbai-embed-large\real_embeddings.npz `
    --manifests experiments\benchmarks\outputs\paws\mxbai-embed-large\manifests `
    --output replay-paws.json
```

The replay refuses to run when the NPZ dataset hash, manifest metadata, split hashes, row counts, labels, IDs, texts, or embedding dimensions do not match. It reproduces the original comparison between cosine-only, absolute difference, elementwise product, residual, full-pair, and shuffled-label readouts.

This is a replay of preserved inputs, not a reopening of the invalid RELATE-E01 identity.

## Install and test

```bash
python -m pip install -e ".[dev]"
pytest
```

For users who only need replay support without the test dependency:

```bash
python -m pip install -e ".[replay]"
```

## Scope

RELATE contains:

- a NumPy-only ridge projection;
- robust scaling of relation coordinates;
- Chebyshev relation search;
- optional cosine candidate generation;
- the Python AST coordinates behind the original result;
- read-only SQLite, NPZ, and NPY inventory;
- deterministic replay of preserved external pair benchmarks;
- focused unit tests.

RELATE does **not** contain an experiment manager, artifact ledger, authorization system, publication workflow, model downloader, or agent architecture.
