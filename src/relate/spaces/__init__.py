"""Spaces package."""

from relate.spaces.comparison import SpaceComparison, compare_spaces
from relate.spaces.derivation import DERIVATION_KINDS, derive
from relate.spaces.identity import SpaceIdentity, derive_space
from relate.spaces.registry import SpaceRegistry

__all__ = [
    "DERIVATION_KINDS",
    "SpaceComparison",
    "SpaceIdentity",
    "SpaceRegistry",
    "compare_spaces",
    "derive",
    "derive_space",
]
