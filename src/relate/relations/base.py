"""Relation wrapper: a named question asked of a frozen representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from relate.model import RelationProjection


@dataclass(frozen=True, slots=True)
class Relation:
    """A fitted relation readout bound to one space."""

    name: str
    projection: RelationProjection
    space_hash: str = ""
    method: str = "ridge"

    def project(self, embeddings: npt.ArrayLike):
        return self.projection.project(embeddings)

    def search(self, source_embedding, target_embeddings, *, k: int = 10, **kwargs):
        return self.projection.search(
            source_embedding, target_embeddings, k=k, **kwargs
        )


def fit_relation(
    name: str,
    embeddings: npt.ArrayLike,
    coordinates: npt.ArrayLike,
    *,
    space_hash: str = "",
    method: str = "ridge",
    alpha: float = 1.0,
    relation_names: tuple[str, ...] | None = None,
) -> Relation:
    if not name:
        raise ValueError("name must be non-empty")
    if method != "ridge":
        raise ValueError(f"unknown relation method: {method}")
    projection = RelationProjection.fit(
        np.asarray(embeddings, dtype=np.float64),
        np.asarray(coordinates, dtype=np.float64),
        alpha=alpha,
        relation_names=relation_names,
    )
    return Relation(name=name, projection=projection, space_hash=space_hash, method=method)
