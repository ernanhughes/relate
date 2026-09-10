"""Bridges package."""

from relate.bridges.base import Bridge
from relate.bridges.linear import fit_bridge, fit_linear, fit_procrustes, fit_ridge
from relate.bridges.registry import BridgeRegistry

__all__ = [
    "Bridge",
    "BridgeRegistry",
    "fit_bridge",
    "fit_linear",
    "fit_procrustes",
    "fit_ridge",
]
