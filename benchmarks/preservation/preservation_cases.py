"""Preservation fixture: typed spaces with train/eval anchor split.

Full-fidelity X, fine-dim-weakened Y, typed paraphrase/negation/swap/
temporal/topic variants. Anchors 0-199 fit bridges (and their reverses);
anchors 200-299 judge candidates. Seeded and frozen; self-contained so
each benchmark reproduces from its own directory plus src and corpus.
"""

from __future__ import annotations

import numpy as np

from relate.evaluation import CorrespondenceSet, HardNegativeCase

SEED = 0
N_ANCHORS = 300
N_TRAIN_ANCHORS = 200
N_DIMS = 32
NEGATIVE_TYPES = ("negation", "relation-swap", "temporal-mismatch", "topic-related")


def build_preservation_cases(seed: int = SEED) -> dict:
    """Return X, Y, ids, train/eval correspondences, and typed eval cases."""
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
    coarse = mixing.copy()
    coarse[4:, :] *= 0.15
    nuisance = rng.normal(size=(n_items, N_DIMS))
    nuisance *= 2.0 / np.linalg.norm(nuisance, axis=1, keepdims=True)

    def embed(mixing_matrix: np.ndarray) -> np.ndarray:
        signal = latent @ mixing_matrix
        signal *= 2.5 / np.linalg.norm(signal, axis=1, keepdims=True)
        vectors = signal + nuisance
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    ids = [f"item-{i}" for i in range(n_items)]

    def rows_for(anchors: range) -> list[int]:
        rows = []
        for anchor in anchors:
            for kind in ("base", "paraphrase", "negation", "relation-swap",
                         "temporal-mismatch", "topic-related"):
                rows.append(index[(anchor, kind)])
        return rows

    train_rows = rows_for(range(N_TRAIN_ANCHORS))
    eval_rows = rows_for(range(N_TRAIN_ANCHORS, N_ANCHORS))

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
                    case_id=f"pq-{anchor}-{negative_type}",
                    anchor_id=ids[index[(anchor, "base")]],
                    positive_id=ids[positive],
                    negative_id=ids[index[(anchor, negative_type)]],
                    relation=negative_type,
                    group="typed",
                    metadata={"fixture": "preservation-v1", "seed": seed},
                )
            )
    return {
        "x": embed(mixing),
        "y": embed(coarse),
        "ids": ids,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "train_correspondence": correspondence(train_rows),
        "eval_correspondence": correspondence(eval_rows),
        "cases": cases,
        "seed": seed,
    }
