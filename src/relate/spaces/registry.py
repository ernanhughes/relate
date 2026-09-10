"""Space registry: exact lookup of space identities by hash."""

from __future__ import annotations

from relate.model import RelateError
from relate.spaces.identity import SpaceIdentity


class SpaceRegistry:
    """Maps ``space_hash`` -> :class:`SpaceIdentity`."""

    def __init__(self) -> None:
        self._spaces: dict[str, SpaceIdentity] = {}

    def register(self, space: SpaceIdentity) -> SpaceIdentity:
        self._spaces[space.space_hash] = space
        return space

    def lookup(self, space_hash: str) -> SpaceIdentity | None:
        return self._spaces.get(space_hash)

    def require(self, space_hash: str) -> SpaceIdentity:
        try:
            return self._spaces[space_hash]
        except KeyError:
            raise RelateError(f"unknown space_hash: {space_hash}") from None

    def __len__(self) -> int:
        return len(self._spaces)

    def __contains__(self, space_hash: object) -> bool:
        return space_hash in self._spaces
