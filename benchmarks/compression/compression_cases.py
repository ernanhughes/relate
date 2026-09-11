"""Compression fixture: typed items with a fit/eval anchor split.

Native 128-d embeddings from latent content/polarity/order/tense
structure plus nuisance. Anchors 0-169 fit PCA; anchors 170-249 judge
every producer. Typed hard cases (negation/swap/temporal/topic) on the
eval anchors expose which distinctions survive each width.
Seeded and frozen; self-contained.
"""

from __future__ import annotations

import numpy as np

from relate.evaluation import CorrespondenceSet, HardNegativeCase

SEED = 0
N_ANCHORS = 250
N_TRAIN_ANCHORS = 170
N_DIMS = 128
WIDTHS = (64, 32, 16, 8)
NEGATIVE_TYPES = ("negation", "relation-swap", "temporal-mismatch", "topic-related")


def build_compression_cases(seed: int = SEED) -> dict:
    """Return native vectors, ids, fit/eval correspondences, eval cases."""
    rng = np.random.default_rng(seed)
    content = rng.uniform(-3.0, 3.0, size=(N_ANCHORS, 4))
    rest = np.column_stack([
        rng.choice([-1.0, 1.0], size=N_ANCHORS),
        rng.choice([0.0, 1.0], size=N_ANCHORS),
        rng.uniform(-2.0, 2.0, size=N_ANCHORS),
    ])
    base = np.hstack([content, rest])

    def variants(row: np.ndarray) -> dict[str, np.ndarray]:
        head, polarity, order, tense = row[:4], row[4], row[5], row[6]
        return {
            "base": row,
            "paraphrase": row + rng.normal(0, 0.05, size=7),
            "negation": np.hstack([head, [-polarity], [order], [tense]]),
            "relation-swap": np.hstack([head, [polarity], [1 - order], [tense]]),
            "temporal-mismatch": np.hstack([head, [polarity], [order], [-tense]]),
            "topic-related": np.hstack(
                [rng.uniform(-3.0, 3.0, size=4), [polarity], [order], [tense]]
            ),
        }

    latents, index = [], {}
    for i in range(N_ANCHORS):
        for kind, latent in variants(base[i]).items():
            index[(i, kind)] = len(latents)
            latents.append(latent)
    latent = np.array(latents)
    n_items = len(latents)

    mixing = rng.normal(size=(7, N_DIMS))
    nuisance = rng.normal(size=(n_items, N_DIMS))
    nuisance *= 2.0 / np.linalg.norm(nuisance, axis=1, keepdims=True)
    signal = latent @ mixing
    signal *= 2.5 / np.linalg.norm(signal, axis=1, keepdims=True)
    vectors = signal + nuisance
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = [f"item-{i}" for i in range(n_items)]

    kinds = ("base", "paraphrase", "negation", "relation-swap",
             "temporal-mismatch", "topic-related")

    def rows_for(anchors: range) -> list[int]:
        return [index[(anchor, kind)] for anchor in anchors for kind in kinds]

    def correspondence(rows: list[int]) -> CorrespondenceSet:
        return CorrespondenceSet(
            ids=tuple(ids[r] for r in rows),
            source_rows=tuple(rows),
            target_rows=tuple(rows),
        )

    cases = []
    for anchor in range(N_TRAIN_ANCHORS, N_ANCHORS):
        positive = index[(anchor, "paraphrase")]
        for negative_type in NEGATIVE_TYPES:
            cases.append(
                HardNegativeCase(
                    case_id=f"cq-{anchor}-{negative_type}",
                    anchor_id=ids[index[(anchor, "base")]],
                    positive_id=ids[positive],
                    negative_id=ids[index[(anchor, negative_type)]],
                    relation=negative_type,
                    group="typed",
                    metadata={"fixture": "compression-v1", "seed": seed},
                )
            )
    return {
        "vectors": vectors,
        "ids": ids,
        "fit_rows": rows_for(range(N_TRAIN_ANCHORS)),
        "eval_rows": rows_for(range(N_TRAIN_ANCHORS, N_ANCHORS)),
        "cases": cases,
        "seed": seed,
    }
