"""RELATE v0.1 record schemas — frozen structure for items, pairs, and queries.

Stdlib only. These dataclasses define the on-disk JSONL record shape. Semantic
validation (relation constraints, adjudication precedence, overlap bands) lives
in validate.py; this module only enforces structure and types.

Corpus files:
  items.jsonl    - one Item per line
  pairs.jsonl    - one Pair per line
  queries.jsonl  - one Query per line
"""
from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ONTOLOGY = json.loads((Path(__file__).parent / "ontology.json").read_text(encoding="utf-8"))

RELATION_IDS: set[str] = {r["id"] for r in ONTOLOGY["relations"]}
DOMAIN_IDS: set[str] = {d["id"] for d in ONTOLOGY["domains"]}
PROVENANCE_KINDS: set[str] = set(ONTOLOGY["provenance_kinds"])
QUERY_STYLES: set[str] = set(ONTOLOGY["query_styles"])
HARD_NEG_METHODS: set[str] = set(ONTOLOGY["hard_negative"]["methods"])
HARD_NEG_UNDERLYING: set[str] = set(ONTOLOGY["hard_negative"]["underlying_relations_allowed"])
_ID_RE = {k: re.compile(v) for k, v in ONTOLOGY["id_patterns"].items()}

_TIME_RE = re.compile(r"^\d{3,4}(-\d{2})?$")  # 2004, 2004-06


class SchemaError(ValueError):
    """Raised when a record does not match the frozen structure."""


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #
@dataclass
class Provenance:
    kind: str                         # authored | adapted | perturbed
    source: str | None = None         # for 'adapted': a citation + as-of date
    parent_id: str | None = None      # for 'perturbed': the item it derives from
    transformation: str | None = None # for 'perturbed': which perturbation

    def check(self) -> None:
        if self.kind not in PROVENANCE_KINDS:
            raise SchemaError(f"provenance.kind {self.kind!r} not in {sorted(PROVENANCE_KINDS)}")
        if self.kind == "adapted" and not self.source:
            raise SchemaError("provenance.kind 'adapted' requires a source")
        if self.kind == "perturbed" and not (self.parent_id and self.transformation):
            raise SchemaError("provenance.kind 'perturbed' requires parent_id and transformation")


@dataclass
class Item:
    id: str
    text: str
    domain: str
    template_family: str
    entity_family: str
    provenance: Provenance
    time_index: str | None = None            # '2004' or '2004-06' or None
    atomic_claims: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)  # normalized named entities present

    def check(self) -> None:
        if not _ID_RE["item"].match(self.id):
            raise SchemaError(f"item id {self.id!r} does not match {ONTOLOGY['id_patterns']['item']}")
        if not self.text.strip():
            raise SchemaError(f"{self.id}: empty text")
        if self.domain not in DOMAIN_IDS:
            raise SchemaError(f"{self.id}: domain {self.domain!r} not in {sorted(DOMAIN_IDS)}")
        if not _ID_RE["template_family"].match(self.template_family):
            raise SchemaError(f"{self.id}: bad template_family {self.template_family!r}")
        if not _ID_RE["entity_family"].match(self.entity_family):
            raise SchemaError(f"{self.id}: bad entity_family {self.entity_family!r}")
        if self.time_index is not None and not _TIME_RE.match(self.time_index):
            raise SchemaError(f"{self.id}: time_index {self.time_index!r} not YYYY or YYYY-MM")
        if not isinstance(self.atomic_claims, list) or any(not isinstance(c, str) for c in self.atomic_claims):
            raise SchemaError(f"{self.id}: atomic_claims must be a list[str]")
        self.provenance.check()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Item":
        d = dict(d)
        prov = d.pop("provenance", None)
        if not isinstance(prov, dict):
            raise SchemaError(f"{d.get('id')}: provenance missing or not an object")
        try:
            return cls(provenance=Provenance(**prov), **d)
        except TypeError as e:
            raise SchemaError(f"{d.get('id')}: bad item fields: {e}") from e


# --------------------------------------------------------------------------- #
# Pairs
# --------------------------------------------------------------------------- #
@dataclass
class Pair:
    id: str
    a_id: str
    b_id: str
    relation: str                     # one of RELATION_IDS
    rule_fired: str                   # which adjudication rule fired (== relation)
    runner_up: str | None             # the competing relation considered next-best
    lexical_overlap: float            # Jaccard over content lemmas, [0, 1]
    entity_overlap: int               # count of shared normalized entities
    direction: str | None = None      # 'a->b' for directional relations, else None
    char_3gram_overlap: float | None = None
    bm25: float | None = None
    annotator_ids: list[str] = field(default_factory=list)
    agreement: float | None = None    # Krippendorff-style per-pair proxy, optional
    notes: str = ""

    def check(self) -> None:
        if not _ID_RE["pair"].match(self.id):
            raise SchemaError(f"pair id {self.id!r} does not match pattern")
        for f in ("a_id", "b_id"):
            if not _ID_RE["item"].match(getattr(self, f)):
                raise SchemaError(f"{self.id}: {f} {getattr(self, f)!r} not an item id")
        if self.a_id == self.b_id:
            raise SchemaError(f"{self.id}: a_id == b_id")
        if self.relation not in RELATION_IDS:
            raise SchemaError(f"{self.id}: relation {self.relation!r} not in ontology")
        if self.rule_fired != self.relation:
            raise SchemaError(f"{self.id}: rule_fired {self.rule_fired!r} != relation {self.relation!r}")
        if self.runner_up is not None and self.runner_up not in RELATION_IDS:
            raise SchemaError(f"{self.id}: runner_up {self.runner_up!r} not in ontology")
        if self.runner_up == self.relation:
            raise SchemaError(f"{self.id}: runner_up equals relation")
        if not (0.0 <= self.lexical_overlap <= 1.0001):
            raise SchemaError(f"{self.id}: lexical_overlap {self.lexical_overlap} out of [0,1]")
        if not isinstance(self.entity_overlap, int) or self.entity_overlap < 0:
            raise SchemaError(f"{self.id}: entity_overlap must be a non-negative int")
        if self.direction not in (None, "a->b"):
            raise SchemaError(f"{self.id}: direction {self.direction!r} must be None or 'a->b'")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Pair":
        try:
            return cls(**d)
        except TypeError as e:
            raise SchemaError(f"{d.get('id')}: bad pair fields: {e}") from e


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
@dataclass
class Positive:
    item_id: str
    grade: int    # 1, 2, or 3


@dataclass
class HardNegative:
    item_id: str
    underlying_relation: str
    method: str


@dataclass
class Query:
    id: str
    text: str
    query_style: str
    target_item_id: str                # the base item this query is about
    positives: list[Positive] = field(default_factory=list)
    hard_negatives: list[HardNegative] = field(default_factory=list)

    def check(self) -> None:
        if not _ID_RE["query"].match(self.id):
            raise SchemaError(f"query id {self.id!r} does not match pattern")
        if not self.text.strip():
            raise SchemaError(f"{self.id}: empty text")
        if self.query_style not in QUERY_STYLES:
            raise SchemaError(f"{self.id}: query_style {self.query_style!r} not in {sorted(QUERY_STYLES)}")
        if not _ID_RE["item"].match(self.target_item_id):
            raise SchemaError(f"{self.id}: target_item_id not an item id")
        grades = {p.grade for p in self.positives}
        if not self.positives:
            raise SchemaError(f"{self.id}: no positives")
        if 3 not in grades:
            raise SchemaError(f"{self.id}: needs at least one grade-3 positive")
        for p in self.positives:
            if p.grade not in (1, 2, 3):
                raise SchemaError(f"{self.id}: positive grade {p.grade} not in 1..3")
            if not _ID_RE["item"].match(p.item_id):
                raise SchemaError(f"{self.id}: positive item_id {p.item_id!r} not an item id")
        for hn in self.hard_negatives:
            if hn.underlying_relation not in HARD_NEG_UNDERLYING:
                raise SchemaError(f"{self.id}: hard-neg underlying_relation {hn.underlying_relation!r} not allowed")
            if hn.method not in HARD_NEG_METHODS:
                raise SchemaError(f"{self.id}: hard-neg method {hn.method!r} not in {sorted(HARD_NEG_METHODS)}")
            if not _ID_RE["item"].match(hn.item_id):
                raise SchemaError(f"{self.id}: hard-neg item_id {hn.item_id!r} not an item id")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Query":
        d = dict(d)
        pos = [Positive(**p) for p in d.pop("positives", [])]
        hns = [HardNegative(**h) for h in d.pop("hard_negatives", [])]
        try:
            return cls(positives=pos, hard_negatives=hns, **d)
        except TypeError as e:
            raise SchemaError(f"{d.get('id')}: bad query fields: {e}") from e


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #
def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SchemaError(f"{path}:{n}: invalid JSON: {e}") from e
    return out


def dump_jsonl(records: list[Any], path: str | Path) -> None:
    def _default(o):
        if dataclasses.is_dataclass(o):
            return {k: v for k, v in dataclasses.asdict(o).items() if v is not None or k in ("time_index",)}
        raise TypeError(o)
    with Path(path).open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=_default, ensure_ascii=False, sort_keys=True) + "\n")
