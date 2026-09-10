"""Bridge registry: which artifacts exist, in what lifecycle state.

Answers what bridges exist from A to B and which artifact a version
refers to. It never answers which bridge is safe for a task -- that
comes from measured preservation profiles in 4C, never from this file.
"""

from __future__ import annotations

from relate.bridges.base import Bridge
from relate.model import RelateError


class BridgeRegistry:
    def __init__(self) -> None:
        self._bridges: dict[str, Bridge] = {}

    def register(self, bridge: Bridge) -> Bridge:
        existing = self._bridges.get(bridge.bridge_id)
        if existing is not None:
            if (
                existing.parameter_hash == bridge.parameter_hash
                and existing.source_space_hash == bridge.source_space_hash
                and existing.target_space_hash == bridge.target_space_hash
                and existing.method == bridge.method
                and existing.direction == bridge.direction
                and existing.anchor_set_hash == bridge.anchor_set_hash
            ):
                return existing
            raise RelateError(
                f"bridge id collision with different content: {bridge.bridge_id}"
            )
        self._bridges[bridge.bridge_id] = bridge
        return bridge

    def get(self, bridge_id: str) -> Bridge | None:
        return self._bridges.get(bridge_id)

    def find_all(self, source_hash: str, target_hash: str) -> list[Bridge]:
        """Every candidate artifact from A to B. No ranking, no verdict."""
        return [
            bridge
            for bridge in self._bridges.values()
            if bridge.source_space_hash == source_hash
            and bridge.target_space_hash == target_hash
        ]

    def lookup(self, source_hash: str, target_hash: str) -> Bridge | None:
        matches = self.find_all(source_hash, target_hash)
        return matches[0] if matches else None

    def require(self, source_hash: str, target_hash: str) -> Bridge:
        bridge = self.lookup(source_hash, target_hash)
        if bridge is None:
            raise RelateError(
                f"no measured bridge {source_hash}->{target_hash}: DENIED"
            )
        return bridge

    def __len__(self) -> int:
        return len(self._bridges)
