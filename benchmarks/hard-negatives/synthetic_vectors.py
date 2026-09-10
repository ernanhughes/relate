"""Deterministic vector fixture mirroring the Python-structure experiment.

The original result used frozen CodeBERT embeddings of Python functions with
three objective AST coordinates (complexity, depth, call-sites): a ridge
readout reached ~0.733 ordering accuracy where raw cosine/Euclidean managed
~0.53. The CodeBERT assets are not on this machine, so this fixture mirrors
the *mechanism* with synthetic vectors rather than fabricating history:

- latent relation coordinates Z (3 dims, the "AST truth");
- frozen unit-norm embeddings E = normalize(signal(Z) + nuisance), where the
  nuisance dominates raw geometry but carries no relation signal;
- case triples where the positive is latent-nearest to the anchor while the
  negative is the most cosine-similar item among latent-far items
  (deceptively close: the hard-negative regime).

Unit norm makes cosine and Euclidean orderings identical by construction --
the same reason the historic cosine/Euclidean figures coincide (~0.53).

Seeded and frozen: same seed, same vectors, same cases, same reports.
"""

from __future__ import annotations

import numpy as np

from relate.evaluation import HardNegativeCase
from relate.model import RelationProjection

SEED = 0
N_ITEMS = 2000
N_RELATION_DIMS = 3
N_EMBEDDING_DIMS = 32
N_TRAIN = 1500
N_CASES = 200
SIGNAL_NORM = 1.5
NUISANCE_NORM = 3.0
RELATION_NAMES = ("complexity", "depth", "call_sites")
ALPHA = 1.0


def build_fixture(seed: int = SEED) -> dict:
    """Return vectors, cases, fitted projection and split metadata."""
    rng = np.random.default_rng(seed)
    latent = rng.uniform(-3.0, 3.0, size=(N_ITEMS, N_RELATION_DIMS))
    mixing = rng.normal(size=(N_RELATION_DIMS, N_EMBEDDING_DIMS))
    nuisance = rng.normal(size=(N_ITEMS, N_EMBEDDING_DIMS))
    nuisance *= NUISANCE_NORM / np.linalg.norm(nuisance, axis=1, keepdims=True)
    signal = latent @ mixing
    signal *= SIGNAL_NORM / np.linalg.norm(signal, axis=1, keepdims=True)
    embeddings = signal + nuisance
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    projection = RelationProjection.fit(
        embeddings[:N_TRAIN],
        latent[:N_TRAIN],
        alpha=ALPHA,
        relation_names=RELATION_NAMES,
    )

    test_latent = latent[N_TRAIN:]
    test_embeddings = embeddings[N_TRAIN:]
    latent_d = np.linalg.norm(
        test_latent[:, None, :] - test_latent[None, :, :], axis=2
    )
    np.fill_diagonal(latent_d, np.inf)
    cosine = test_embeddings @ test_embeddings.T
    np.fill_diagonal(cosine, -np.inf)
    far = latent_d > np.median(latent_d)
    deceptive = np.where(far, cosine, -np.inf)

    vectors = {f"item-{i}": embeddings[i] for i in range(N_ITEMS)}
    cases = []
    for row in range(N_CASES):
        anchor = N_TRAIN + row
        positive = N_TRAIN + int(np.argmin(latent_d[row]))
        negative = N_TRAIN + int(np.argmax(deceptive[row]))
        band = "low" if row % 2 == 0 else "high"
        cases.append(
            HardNegativeCase(
                case_id=f"syn-{anchor}:{positive}:{negative}",
                anchor_id=f"item-{anchor}",
                positive_id=f"item-{positive}",
                negative_id=f"item-{negative}",
                relation="same_structure",
                negative_relation="nuisance_match",
                group=band,
                metadata={"fixture": "synthetic-v1", "seed": seed, "band": band},
            )
        )
    return {
        "vectors": vectors,
        "cases": cases,
        "projection": projection,
        "seed": seed,
        "n_train": N_TRAIN,
        "relation_names": list(RELATION_NAMES),
    }
