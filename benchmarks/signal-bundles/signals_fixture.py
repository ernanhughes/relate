"""Ablation fixture: hard-case decisions with inspectable geometry.

Self-contained synthetic instrument (seeded): background items plus tight
clusters (so density/hubness vary like the book's hub story), latent
relation coordinates, unit-norm embeddings. Anchors yield positive
instances, easy-negative instances, and -- for every third anchor --
adversarial hard-negative instances. Train/test split by anchor: no
instance of a test anchor is ever seen in training.
"""

from __future__ import annotations

import numpy as np

from relate.evaluation import HardNegativeCase

SEED = 0
N_BACKGROUND = 700
N_CLUSTERS = 5
PER_CLUSTER = 60
N_DIMS = 32
N_ANCHORS = 200
N_TRAIN_ANCHORS = 100


def build_fixture(seed: int = SEED) -> dict:
    """Return vectors, anchors, hard/easy cases, and instance blocks."""
    rng = np.random.default_rng(seed)
    n = N_BACKGROUND + N_CLUSTERS * PER_CLUSTER
    latent = rng.uniform(-3.0, 3.0, size=(n, 3))
    mixing = rng.normal(size=(3, N_DIMS))
    nuisance = rng.normal(size=(n, N_DIMS))
    nuisance *= 2.0 / np.linalg.norm(nuisance, axis=1, keepdims=True)
    signal = latent @ mixing
    signal *= 3.0 / np.linalg.norm(signal, axis=1, keepdims=True)
    vectors = signal + nuisance
    centers = rng.normal(size=(N_CLUSTERS, N_DIMS)) * 2.0
    for c in range(N_CLUSTERS):
        rows = N_BACKGROUND + c * PER_CLUSTER + np.arange(PER_CLUSTER)
        vectors[rows] = centers[c] + 0.15 * rng.normal(size=(PER_CLUSTER, N_DIMS))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    distances = np.linalg.norm(latent[:, None, :] - latent[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    cosine = vectors @ vectors.T
    np.fill_diagonal(cosine, -np.inf)
    far = distances > np.median(distances)
    deceptive = np.where(far, cosine, -np.inf)

    anchors = rng.choice(N_BACKGROUND, size=N_ANCHORS, replace=False)
    hard_cases: list[HardNegativeCase] = []
    easy_cases: list[HardNegativeCase] = []
    for i, anchor in enumerate(anchors):
        positive = int(np.argmin(distances[anchor]))
        far_rows = np.where(far[anchor])[0]
        easy = int(far_rows[rng.integers(len(far_rows))])
        if i % 3 == 0:
            hard_cases.append(
                HardNegativeCase(
                    case_id=f"hard-{anchor}",
                    anchor_id=f"item-{anchor}",
                    positive_id=f"item-{positive}",
                    negative_id=f"item-{int(np.argmax(deceptive[anchor]))}",
                    relation="same_structure",
                    negative_relation="nuisance_match",
                    group="hard",
                    metadata={"fixture": "signals-v1", "seed": seed},
                )
            )
        easy_cases.append(
            HardNegativeCase(
                case_id=f"easy-{anchor}",
                anchor_id=f"item-{anchor}",
                positive_id=f"item-{positive}",
                negative_id=f"item-{easy}",
                relation="same_structure",
                negative_relation="random_far",
                group="easy",
                metadata={"fixture": "signals-v1", "seed": seed},
            )
        )
    hard_by_anchor = {c.anchor_id: c for c in hard_cases}
    blocks = []
    for case in easy_cases:
        anchor = int(case.anchor_id.split("-")[1])
        positive = int(case.positive_id.split("-")[1])
        easy = int(case.negative_id.split("-")[1])
        block = [
            {"anchor": anchor, "candidate": positive, "label": 1,
             "case_id": case.case_id, "sign": +1},
            {"anchor": anchor, "candidate": easy, "label": 0,
             "case_id": case.case_id, "sign": -1},
        ]
        hard = hard_by_anchor.get(case.anchor_id)
        if hard is not None:
            block.append(
                {"anchor": anchor,
                 "candidate": int(hard.negative_id.split("-")[1]),
                 "label": 0, "case_id": hard.case_id, "sign": -1}
            )
        blocks.append(block)
    return {
        "vectors": vectors,
        "ids": [f"item-{i}" for i in range(n)],
        "anchors": [int(a) for a in anchors],
        "hard_cases": hard_cases,
        "easy_cases": easy_cases,
        "blocks": blocks,
        "seed": seed,
    }
