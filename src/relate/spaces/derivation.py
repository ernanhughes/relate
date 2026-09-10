"""Derived-space tracking: every transformation creates a new space."""

from __future__ import annotations

from relate.spaces.identity import SpaceIdentity, derive_space

DERIVATION_KINDS = (
    "pca",
    "whitening",
    "bridge",
    "matryoshka-truncate",
    "truncate",
    "quantize",
)


def derive(parent: SpaceIdentity, kind: str, **kwargs) -> SpaceIdentity:
    """Validate the kind, then delegate to :func:`derive_space`."""
    if kind not in DERIVATION_KINDS:
        raise ValueError(f"unknown derivation kind: {kind}")
    params = kwargs.pop("params", None)
    return derive_space(parent, kind, params, **kwargs)
