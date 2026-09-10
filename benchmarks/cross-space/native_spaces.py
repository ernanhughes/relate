"""Two native encoders over shared items: coarse transfer, fine failure.

Model X is full-fidelity; model Y shares the nuisance backbone but its
mixing weakens the polarity/order/tense latent dims (x0.15), so Y is
blind to exactly the distinctions the typed cases probe. Items carry
typed variants (paraphrase positive; negation / relation-swap /
temporal-mismatch / topic-related negatives), letting per-relation
deltas show which distinctions survive the representation change.
Seeded and frozen.
"""

from __future__ import annotations

import numpy as np

from relate.evaluation import HardNegativeCase

SEED = 0
N_ANCHORS = 300
N_DIMS = 32
N_LATENT = 7
FINE_WEAKEN = 0.15
SIGNAL_NORM = 2.5
NUISANCE_NORM = 2.0
NEGATIVE_TYPES = ("negation", "relation-swap", "temporal-mismatch", "topic-related")


def build_native_spaces(seed: int = SEED) -> dict:
    """Return X, Y, ids, typed cases, and fixture metadata."""
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
            "paraphrase": row + rng.normal(0, 0.05, size=N_LATENT),
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

    mixing_x = rng.normal(size=(N_LATENT, N_DIMS))
    mixing_y = mixing_x.copy()
    mixing_y[4:, :] *= FINE_WEAKEN
    nuisance = rng.normal(size=(n_items, N_DIMS))
    nuisance *= NUISANCE_NORM / np.linalg.norm(nuisance, axis=1, keepdims=True)

    def embed(mixing: np.ndarray) -> np.ndarray:
        signal = latent @ mixing
        signal *= SIGNAL_NORM / np.linalg.norm(signal, axis=1, keepdims=True)
        vectors = signal + nuisance
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    ids = [f"item-{i}" for i in range(n_items)]
    cases = []
    for i in range(N_ANCHORS):
        anchor = index[(i, "base")]
        positive = index[(i, "paraphrase")]
        for negative_type in NEGATIVE_TYPES:
            cases.append(
                HardNegativeCase(
                    case_id=f"q-{i}-{negative_type}",
                    anchor_id=ids[anchor],
                    positive_id=ids[positive],
                    negative_id=ids[index[(i, negative_type)]],
                    relation=negative_type,
                    group="typed",
                    metadata={"fixture": "native-v1", "seed": seed},
                )
            )
    return {
        "x": embed(mixing_x),
        "y": embed(mixing_y),
        "ids": ids,
        "cases": cases,
        "seed": seed,
    }
