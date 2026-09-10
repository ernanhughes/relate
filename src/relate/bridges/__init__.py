"""Bridges: directional candidate producers with provenance, never judgment."""

from relate.bridges.base import (
    BRIDGE_METHODS,
    VALID_STATUSES,
    Bridge,
    BridgeFitProvenance,
    BridgeSpec,
    bridge_output_space,
    make_bridge_id,
)
from relate.bridges.controls import (
    constant_centroid_bridge,
    identity_bridge,
    random_map_bridge,
)
from relate.bridges.fit import fit_bridge
from relate.bridges.linear import fit_linear
from relate.bridges.procrustes import fit_procrustes
from relate.bridges.registry import BridgeRegistry
from relate.bridges.ridge import fit_ridge

__all__ = [
    "BRIDGE_METHODS",
    "VALID_STATUSES",
    "Bridge",
    "BridgeFitProvenance",
    "BridgeRegistry",
    "BridgeSpec",
    "bridge_output_space",
    "constant_centroid_bridge",
    "fit_bridge",
    "fit_linear",
    "fit_procrustes",
    "fit_ridge",
    "identity_bridge",
    "make_bridge_id",
    "random_map_bridge",
]
