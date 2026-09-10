"""Paired synthetic spaces: shared coarse skeleton, divergent fine signal.

X and Y share the same nuisance backbone (coarse neighborhoods agree) but
Y's relation-carrying signal is blended toward fresh latents
(``signal_alpha``). With the calibrated alpha=0.7 the pair shows the book's
signature split: CKA high, counterpart recovery perfect, neighborhoods
respectable -- yet the relation readout fitted on X loses fine ordering on
Y. Seeded and frozen.
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
SIGNAL_ALPHA = 0.7
RELATION_NAMES = ("complexity", "depth", "call_sites")
ALPHA = 1.0


def build_paired_spaces(seed: int = SEED, signal_alpha: float = SIGNAL_ALPHA) -> dict:
    """Return X, Y, ids, mined cases, and the X-fitted projection."""
    rng = np.random.default_rng(seed)
    latent = rng.uniform(-3.0, 3.0, size=(N_ITEMS, N_RELATION_DIMS))
    mixing = rng.normal(size=(N_RELATION_DIMS, N_EMBEDDING_DIMS))
    nuisance = rng.normal(size=(N_ITEMS, N_EMBEDDING_DIMS))
    nuisance *= 3.0 / np.linalg.norm(nuisance, axis=1, keepdims=True)

    def embed(current: np.ndarray) -> np.ndarray:
        signal = current @ mixing
        signal *= 1.5 / np.linalg.norm(signal, axis=1, keepdims=True)
        vectors = signal + nuisance
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    fresh = rng.uniform(-3.0, 3.0, size=(N_ITEMS, N_RELATION_DIMS))
    blended = signal_alpha * latent + (1.0 - signal_alpha) * fresh
    space_x = embed(latent)
    space_y = embed(blended)

    projection = RelationProjection.fit(
        space_x[:N_TRAIN],
        latent[:N_TRAIN],
        alpha=ALPHA,
        relation_names=RELATION_NAMES,
    )

    # Cases mined on X: latent-nearest positive, most-cosine-similar
    # latent-far negative. Fixed IDs let every scorer and space share them.
    test_latent = latent[N_TRAIN:]
    test_x = space_x[N_TRAIN:]
    latent_d = np.linalg.norm(
        test_latent[:, None, :] - test_latent[None, :, :], axis=2
    )
    np.fill_diagonal(latent_d, np.inf)
    cosine = test_x @ test_x.T
    np.fill_diagonal(cosine, -np.inf)
    deceptive = np.where(latent_d > np.median(latent_d), cosine, -np.inf)

    ids = [f"item-{i}" for i in range(N_ITEMS)]
    cases = []
    for row in range(N_CASES):
        anchor = N_TRAIN + row
        cases.append(
            HardNegativeCase(
                case_id=f"pair-{anchor}",
                anchor_id=ids[anchor],
                positive_id=ids[N_TRAIN + int(np.argmin(latent_d[row]))],
                negative_id=ids[N_TRAIN + int(np.argmax(deceptive[row]))],
                relation="same_structure",
                negative_relation="nuisance_match",
                group="test",
                metadata={"fixture": "paired-v1", "seed": seed},
            )
        )
    return {
        "x": space_x,
        "y": space_y,
        "ids": ids,
        "cases": cases,
        "projection": projection,
        "seed": seed,
        "signal_alpha": signal_alpha,
    }
