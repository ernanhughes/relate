"""Bridge registry: measured crossings between explicitly identified spaces."""

from __future__ import annotations

from relate.bridges.base import Bridge
from relate.model import RelateError


class BridgeRegistry:
    def __init__(self) -> None:
        self._bridges: dict[tuple[str, str, str], Bridge] = {}

    def register(self, bridge: Bridge) -> Bridge:
        self._bridges[
            (bridge.source_space_hash, bridge.target_space_hash, bridge.direction)
        ] = bridge
        return bridge

    def lookup(self, source_hash: str, target_hash: str) -> Bridge | None:
        for (s, t, _), bridge in self._bridges.items():
            if s == source_hash and t == target_hash:
                return bridge
        return None

    def require(self, source_hash: str, target_hash: str) -> Bridge:
        bridge = self.lookup(source_hash, target_hash)
        if bridge is None:
            raise RelateError(
                f"no measured bridge {source_hash}->{target_hash}: DENIED"
            )
        return bridge
