"""Space identity: exact identification of the pipeline that produced a vector.

A vector carries a ``space_hash`` derived from the model weights/revision,
tokenizer, prefixes, sequence length/truncation, quantization/dtype and
post-processing. Derived spaces (PCA, whitening, bridge output, Matryoshka
truncation) get new identities that record their parent.

Three-layer rule enforced by the runtime:

- IDENTITY:      exact, ``space_hash`` equality permits an operation.
- COMPATIBILITY: empirical, measured evidence (bridge + preservation profile).
- USABILITY:     policy, ``usable_for(scope)`` gates the use.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SpaceIdentity:
    """Exactly identify the pipeline that produced a vector."""

    model: str
    dimensions: int
    revision: str = ""
    tokenizer: str = ""
    prefixes: str = ""
    sequence_length: int = 0
    truncation: str = ""
    dtype: str = "float32"
    normalize: bool = True
    post_processing: str = ""
    derived_from: str = ""
    derivation: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if not isinstance(self.dimensions, int) or isinstance(self.dimensions, bool):
            raise ValueError("dimensions must be an integer")
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "revision": self.revision,
            "tokenizer": self.tokenizer,
            "prefixes": self.prefixes,
            "sequence_length": self.sequence_length,
            "truncation": self.truncation,
            "dtype": self.dtype,
            "normalize": self.normalize,
            "post_processing": self.post_processing,
            "derived_from": self.derived_from,
            "derivation": self.derivation,
        }

    @classmethod
    def from_dict(cls, values: dict) -> "SpaceIdentity":
        return cls(
            model=str(values["model"]),
            dimensions=int(values["dimensions"]),
            revision=str(values.get("revision", "")),
            tokenizer=str(values.get("tokenizer", "")),
            prefixes=str(values.get("prefixes", "")),
            sequence_length=int(values.get("sequence_length", 0)),
            truncation=str(values.get("truncation", "")),
            dtype=str(values.get("dtype", "float32")),
            normalize=bool(values.get("normalize", True)),
            post_processing=str(values.get("post_processing", "")),
            derived_from=str(values.get("derived_from", "")),
            derivation=str(values.get("derivation", "")),
        )

    @property
    def space_hash(self) -> str:
        """Canonical sha256 over the sorted identity dict (first 16 hex chars)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @property
    def short(self) -> str:
        return f"{self.model} [{self.space_hash}]"


def derive_space(
    parent: SpaceIdentity,
    kind: str,
    params: dict | None = None,
    *,
    dimensions: int | None = None,
    post_processing: str = "",
) -> SpaceIdentity:
    """Create a new derived-space identity (PCA / whitening / bridge / truncation).

    The derived space always gets a new ``space_hash``; the parent hash is
    recorded in ``derived_from`` so lineage is explicit.
    """
    if not kind:
        raise ValueError("kind must be non-empty")
    derivation = kind if not params else f"{kind}:{json.dumps(params, sort_keys=True)}"
    return SpaceIdentity(
        model=parent.model,
        dimensions=dimensions if dimensions is not None else parent.dimensions,
        revision=parent.revision,
        tokenizer=parent.tokenizer,
        prefixes=parent.prefixes,
        sequence_length=parent.sequence_length,
        truncation=parent.truncation,
        dtype=parent.dtype,
        normalize=parent.normalize,
        post_processing=post_processing or f"derived:{derivation}",
        derived_from=parent.space_hash,
        derivation=derivation,
    )
