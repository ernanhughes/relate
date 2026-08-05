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

## Install and test

```bash
python -m pip install -e ".[dev]"
pytest
```

## Scope

RELATE contains:

- a NumPy-only ridge projection;
- robust scaling of relation coordinates;
- Chebyshev relation search;
- optional cosine candidate generation;
- the Python AST coordinates behind the original result;
- focused unit tests.

RELATE does **not** contain an experiment manager, artifact ledger, authorization system, publication workflow, benchmark framework, model downloader, or agent architecture.
