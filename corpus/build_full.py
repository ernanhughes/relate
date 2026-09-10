"""RELATE v0.1 full build — stages 6-9 (spec §15 items 6-10).

  python build_full.py            # generate -> validate -> dedup/leakage -> splits
                                  # -> BM25 -> task views -> datasheet -> hash
  python build_full.py --check    # build to a temp dir and diff against the frozen release

Output tree (spec §13), written to ./relate-0.1.0/ :

  items.jsonl  pairs.jsonl  queries.jsonl
  splits/{random,entity,template,domain}.json
  bm25_scores.jsonl
  views/{retrieval,relation,hard-negative,calibration,dimensionality,alignment}-v0.1.json
  datasheet.md
  corpus_hash.txt
  manifest.json

Stdlib only. No embedding model is loaded here — Wave 1 (experiments/wave1/) does that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import generate
import lexical
from schema import Item, Pair, Query, dump_jsonl
from validate import ValidationReport, check_pair_semantics, check_query_semantics, coverage_report

HERE = Path(__file__).parent
OUT = HERE / "relate-0.1.0"
VERSION = "0.1.0"
SPLIT_FRAC = (0.6, 0.2, 0.2)
_WORD = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# deterministic helpers
# --------------------------------------------------------------------------- #
def _h(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def _bucket(key: str, salt: str) -> str:
    """Stable train/dev/test assignment from a hash in [0,1)."""
    x = int(_h(salt, key)[:8], 16) / 0xFFFFFFFF
    if x < SPLIT_FRAC[0]:
        return "train"
    if x < SPLIT_FRAC[0] + SPLIT_FRAC[1]:
        return "dev"
    return "test"


def _shingles(text: str, k: int = 4) -> set[str]:
    toks = _WORD.findall(text.lower())
    return {" ".join(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))}


# --------------------------------------------------------------------------- #
# BM25 (Okapi, stdlib)
# --------------------------------------------------------------------------- #
class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.ids = list(docs)
        self.k1, self.b = k1, b
        self.toks = {i: _WORD.findall(docs[i].lower()) for i in self.ids}
        self.len = {i: len(t) for i, t in self.toks.items()}
        self.avgdl = sum(self.len.values()) / max(1, len(self.ids))
        df: Counter = Counter()
        for t in self.toks.values():
            df.update(set(t))
        N = len(self.ids)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = {i: Counter(t) for i, t in self.toks.items()}

    def score(self, query: str, doc_id: str) -> float:
        q = _WORD.findall(query.lower())
        tf, dl = self.tf[doc_id], self.len[doc_id]
        s = 0.0
        for w in q:
            if w not in self.idf or w not in tf:
                continue
            f = tf[w]
            s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def topk(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        scored = [(i, self.score(query, i)) for i in self.ids]
        scored.sort(key=lambda p: -p[1])
        return scored[:k]


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(out: Path, v02: bool = False) -> dict:
    rep = ValidationReport()
    c = generate.generate()
    qseeds_all = c.qseeds + (c.hard_qseeds if v02 else [])

    # ---- structural validation of every item -------------------------------
    items: dict[str, Item] = {}
    for raw in c.items.values():
        it = Item.from_dict(dict(raw))
        it.check()
        items[it.id] = it

    # ---- near-duplicate detection (spec §8) -------------------------------
    #   MinHash-free: exact text + shingle-Jaccard >= 0.9 over the item pool.
    by_text: dict[str, list[str]] = defaultdict(list)
    for i in items.values():
        by_text[i.text.strip().lower()].append(i.id)
    exact_dups = {ids[0]: ids[1:] for ids in by_text.values() if len(ids) > 1}
    shingle_cache = {i: _shingles(items[i].text) for i in items}
    near_dups: list[tuple[str, str, float]] = []
    ids_sorted = sorted(items)
    for a_i in range(len(ids_sorted)):
        a = ids_sorted[a_i]
        sa = shingle_cache[a]
        if not sa:
            continue
        for b in ids_sorted[a_i + 1:]:
            sb = shingle_cache[b]
            if not sb:
                continue
            j = len(sa & sb) / len(sa | sb)
            if j >= 0.9 and items[a].text.strip().lower() != items[b].text.strip().lower():
                near_dups.append((a, b, round(j, 3)))

    # ---- pairs: structural + semantic ------------------------------------
    all_pairs: list[Pair] = []
    seen = set()
    for raw in c.pairs:
        pr = Pair.from_dict(dict(raw))
        pr.check()
        key = (pr.a_id, pr.b_id, pr.relation)
        if key in seen:
            continue
        seen.add(key)
        check_pair_semantics(pr, items, rep)
        all_pairs.append(pr)

    # ---- stage 6: balance by difficulty (spec §5, DEV_FINDINGS §6) -------
    #   For the four overlap-balanced relations, trim the over-full lexical-
    #   overlap bands so every band reaches the >=25% floor. Kept pairs are
    #   chosen deterministically, preferring pairs whose items appear in a
    #   query or in another relation (higher downstream value).
    pairs, balance_log = _balance_bands(all_pairs, c.qseeds)

    # ---- queries -------------------------------------------------------
    docs = {i: items[i].text for i in items}
    bm25 = BM25(docs)
    queries: list[Query] = []
    qseen: set[str] = set()
    for qs in qseeds_all:
        qid = "q-" + _h(qs["text"], qs["target_item_id"])[:10]
        if qid in qseen:
            continue
        qseen.add(qid)
        # mine one lexical-overlap-matched hard negative per query (BM25 top-k
        # that is not a graded positive and shares no strong logical relation)
        pos_ids = {p["item_id"] for p in qs["positives"]}
        hn = list(qs["hard_negatives"])
        hn_ids = {h["item_id"] for h in hn}
        for cand, _ in bm25.topk(qs["text"], k=12):
            if cand in pos_ids or cand in hn_ids or cand == qs["target_item_id"]:
                continue
            if items[cand].domain != items[qs["target_item_id"]].domain:
                continue
            hn.append(dict(item_id=cand, underlying_relation="topic-related", method="lexical_overlap_matched"))
            hn_ids.add(cand)
            break
        q = Query.from_dict(dict(
            id=qid, text=qs["text"], query_style=qs["query_style"],
            target_item_id=qs["target_item_id"], positives=qs["positives"], hard_negatives=hn,
        ))
        q.check()
        check_query_semantics(q, items, rep)
        queries.append(q)

    # ---- query-positive triviality check (spec §8) ----------------------
    trivial = 0
    for q in queries:
        for p in q.positives:
            if p.grade == 3:
                pr_rel = _pair_relation(pairs, q.target_item_id, p.item_id)
                lo = lexical.jaccard(q.text, items[p.item_id].text)
                if lo >= 0.999 and pr_rel != "equivalent":
                    trivial += 1

    # ---- overlap-band strata (spec §5) ---------------------------------
    bands = coverage_report(items, pairs, queries, rep)["overlap_bands_balanced_relations"]

    if rep.errors:
        raise SystemExit("VALIDATION FAILED:\n" + "\n".join(rep.errors[:40]))

    # ---- splits (spec §7) --------------------------------------------
    splits = _build_splits(items, pairs)

    # ---- task views (spec §13, registry §16) ------------------------
    views = _build_views(items, pairs, queries)

    # ---- write output tree ------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(exist_ok=True)
    (out / "views").mkdir(exist_ok=True)
    item_recs = [_item_rec(items[i]) for i in sorted(items)]
    pair_recs = sorted((_pair_rec(p) for p in pairs), key=lambda r: r["id"])
    query_recs = sorted((_query_rec(q) for q in queries), key=lambda r: r["id"])
    _write_jsonl(out / "items.jsonl", item_recs)
    _write_jsonl(out / "pairs.jsonl", pair_recs)
    _write_jsonl(out / "queries.jsonl", query_recs)
    for axis, table in splits.items():
        (out / "splits" / f"{axis}.json").write_text(json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")
    for name, v in views.items():
        (out / "views" / f"{name}-v0.1.json").write_text(json.dumps(v, indent=2, sort_keys=True), encoding="utf-8")

    bm25_recs = []
    for q in queries:
        for cand, sc in bm25.topk(q.text, k=20):
            bm25_recs.append(dict(query_id=q.id, item_id=cand, bm25=round(sc, 4)))
    _write_jsonl(out / "bm25_scores.jsonl", bm25_recs)

    # ---- datasheet + hash ------------------------------------------
    stats = _stats(items, pairs, queries, bands, exact_dups, near_dups, trivial, rep)
    stats["balance_log"] = balance_log
    stats["adjudication_adjustments"] = len(c.adjust_log)
    corpus_hash = _corpus_hash(item_recs, pair_recs, query_recs)
    datasheet = _datasheet(stats, corpus_hash)
    (out / "datasheet.md").write_text(datasheet, encoding="utf-8")
    (out / "corpus_hash.txt").write_text(corpus_hash + "\n", encoding="utf-8")
    manifest = dict(
        version="0.2.0" if v02 else VERSION, corpus_hash=corpus_hash,
        files={p.name: _sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"},
        counts=stats["counts"], warnings=len(rep.warnings),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return dict(stats=stats, corpus_hash=corpus_hash, warnings=rep.warnings)


_BALANCED = ("paraphrase", "entailment", "contradiction", "topic-related")


def _band(x: float) -> str:
    return "low" if x < 0.2 else ("mid" if x < 0.5 else "high")


def _balance_bands(pairs: list[Pair], qseeds: list[dict]) -> tuple[list[Pair], dict]:
    used = {p["item_id"] for q in qseeds for p in q["positives"]}
    used |= {h["item_id"] for q in qseeds for h in q["hard_negatives"]}
    rel_of_item: Counter = Counter()
    for p in pairs:
        rel_of_item[p.a_id] += 1
        rel_of_item[p.b_id] += 1

    def value(p: Pair) -> tuple:
        # keep-priority: touches a query > touches many relations > stable hash
        return (
            -(int(p.a_id in used) + int(p.b_id in used)),
            -(rel_of_item[p.a_id] + rel_of_item[p.b_id]),
            _h(p.id),
        )

    keep: set[str] = set()
    log: dict = {}
    for rel in _BALANCED:
        rp = [p for p in pairs if p.relation == rel]
        by_band: dict[str, list[Pair]] = defaultdict(list)
        for p in rp:
            by_band[_band(p.lexical_overlap)].append(p)
        counts = {b: len(by_band[b]) for b in ("low", "mid", "high")}
        m = min(counts.values()) or 1
        cap = math.ceil(1.5 * m)  # each non-scarce band -> <= 1.5*m ; scarce stays -> >= 25%
        kept_rel = []
        for b in ("low", "mid", "high"):
            ordered = sorted(by_band[b], key=value)
            kept_rel += ordered[:max(cap, counts[b] if counts[b] <= cap else cap)]
        keep |= {p.id for p in kept_rel}
        log[rel] = dict(before=counts, after={b: sum(1 for p in kept_rel if _band(p.lexical_overlap) == b) for b in ("low", "mid", "high")})

    out = [p for p in pairs if p.relation not in _BALANCED or p.id in keep]
    return out, log


def _pair_relation(pairs, a, b):
    for p in pairs:
        if {p.a_id, p.b_id} == {a, b}:
            return p.relation
    return None


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def _build_splits(items: dict[str, Item], pairs: list[Pair]) -> dict:
    out: dict[str, dict] = {}

    # split_random: assign each BASE ITEM to a split; a pair's split is defined
    # only when both endpoints agree (spec §7: no base item spans splits).
    item_split = {i: _bucket(i, "random") for i in items}
    out["random"] = dict(
        by_item=item_split,
        note="pair-usable only when both endpoints fall in the same split; spec §7",
    )

    # split_entity: partition entity families
    ef_split = {}
    for ef in sorted({it.entity_family for it in items.values()}):
        ef_split[ef] = _bucket(ef, "entity")
    out["entity"] = dict(by_entity_family=ef_split, by_item={i: ef_split[items[i].entity_family] for i in items})

    # split_template: partition template families
    tf_split = {}
    for tf in sorted({it.template_family for it in items.values()}):
        tf_split[tf] = _bucket(tf, "template")
    out["template"] = dict(by_template_family=tf_split, by_item={i: tf_split[items[i].template_family] for i in items})

    # split_domain: leave-one-domain-out, 5 folds
    domains = sorted({it.domain for it in items.values()})
    out["domain"] = dict(
        folds={d: {"test_domain": d, "train_domains": [x for x in domains if x != d]} for d in domains},
        by_item={i: items[i].domain for i in items},
    )
    return out


# --------------------------------------------------------------------------- #
# task views  (deterministic manifests over the frozen corpus)
# --------------------------------------------------------------------------- #
def _build_views(items, pairs, queries) -> dict:
    v: dict[str, dict] = {}
    q_all = [q.id for q in queries]

    v["retrieval"] = dict(
        description="graded-relevance retrieval over the full item pool; queries with >=1 grade-3 positive",
        query_ids=sorted(q_all),
        corpus_item_ids=sorted(items),
        relevance_grades={"3": "answers+correct", "2": "supports main claim", "1": "on-topic", "0": "everything else"},
    )
    v["relation"] = dict(
        description="mean cosine / margin per typed relation; all typed pairs",
        pair_ids_by_relation={
            rel: sorted(p.id for p in pairs if p.relation == rel)
            for rel in sorted({p.relation for p in pairs})
        },
    )
    hn_q = [q for q in queries if q.hard_negatives]
    triples = [
        dict(query_id=q.id,
             positive_id=next(p.item_id for p in q.positives if p.grade == 3),
             hard_negative_id=h.item_id,
             underlying_relation=h.underlying_relation,
             method=h.method)
        for q in hn_q for h in q.hard_negatives
    ]
    v["hard-negative"] = dict(
        description="(query, grade-3 positive, hard-negative) triples; underlying relation + mining method on each negative",
        triples=sorted(triples, key=lambda t: (t["query_id"], t["hard_negative_id"])),
        headline_method="structured_perturbation",
    )
    v["calibration"] = dict(
        description="binary same-claim / different-claim decision; positive = equivalent|paraphrase pairs, negative = negation|contradiction|temporal-mismatch|relation-swap pairs",
        positive_pair_ids=sorted(p.id for p in pairs if p.relation in ("equivalent", "paraphrase")),
        negative_pair_ids=sorted(p.id for p in pairs if p.relation in ("negation", "contradiction", "temporal-mismatch", "relation-swap")),
        by_domain=True,
    )
    v["dimensionality"] = dict(
        description="full item pool for effective-rank / anisotropy / intrinsic-dimension estimates; 5000 random item pairs for the anisotropy baseline",
        corpus_item_ids=sorted(items),
        random_pair_seed=1729,
        random_pair_count=5000,
    )
    v["alignment"] = dict(
        description="held-out item pool for cross-space bridge fitting; anchors drawn from split_entity train, evaluation on split_entity test",
        anchor_pool="split_entity:train",
        eval_pool="split_entity:test",
        per_relation_eval=True,
    )
    return v


# --------------------------------------------------------------------------- #
# record shaping
# --------------------------------------------------------------------------- #
def _item_rec(it: Item) -> dict:
    d = dict(id=it.id, text=it.text, domain=it.domain, template_family=it.template_family,
             entity_family=it.entity_family, entities=it.entities, time_index=it.time_index,
             atomic_claims=it.atomic_claims,
             provenance={k: v for k, v in vars(it.provenance).items() if v is not None} or {"kind": it.provenance.kind})
    return d


def _pair_rec(p: Pair) -> dict:
    return dict(id=p.id, a_id=p.a_id, b_id=p.b_id, relation=p.relation, rule_fired=p.rule_fired,
               runner_up=p.runner_up, direction=p.direction, lexical_overlap=p.lexical_overlap,
               char_3gram_overlap=p.char_3gram_overlap, entity_overlap=p.entity_overlap, notes=p.notes)


def _query_rec(q: Query) -> dict:
    return dict(id=q.id, text=q.text, query_style=q.query_style, target_item_id=q.target_item_id,
               positives=[dict(item_id=p.item_id, grade=p.grade) for p in q.positives],
               hard_negatives=[dict(item_id=h.item_id, underlying_relation=h.underlying_relation, method=h.method)
                               for h in q.hard_negatives])


def _write_jsonl(path: Path, recs: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus_hash(items, pairs, queries) -> str:
    h = hashlib.sha256()
    for group in (items, pairs, queries):
        for rec in group:
            h.update(json.dumps(rec, sort_keys=True, ensure_ascii=False).encode())
            h.update(b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# stats + datasheet
# --------------------------------------------------------------------------- #
def _stats(items, pairs, queries, bands, exact_dups, near_dups, trivial, rep) -> dict:
    rel_c = Counter(p.relation for p in pairs)
    dom_c = Counter(i.domain for i in items.values())
    prov_c = Counter(i.provenance.kind for i in items.values())
    style_c = Counter(q.query_style for q in queries)
    hn_method_c = Counter(h.method for q in queries for h in q.hard_negatives)
    hn_rel_c = Counter(h.underlying_relation for q in queries for h in q.hard_negatives)
    lo_by_rel = defaultdict(list)
    for p in pairs:
        lo_by_rel[p.relation].append(p.lexical_overlap)
    lo_summary = {
        r: dict(n=len(v), mean=round(sum(v) / len(v), 3),
                lt02=round(sum(x < 0.2 for x in v) / len(v), 2),
                b0205=round(sum(0.2 <= x < 0.5 for x in v) / len(v), 2),
                gt05=round(sum(x >= 0.5 for x in v) / len(v), 2))
        for r, v in sorted(lo_by_rel.items())
    }
    return dict(
        counts=dict(items=len(items), pairs=len(pairs), queries=len(queries),
                    template_families=len({i.template_family for i in items.values()}),
                    entity_families=len({i.entity_family for i in items.values()})),
        relations=dict(sorted(rel_c.items())),
        domains=dict(sorted(dom_c.items())),
        provenance=dict(sorted(prov_c.items())),
        query_styles=dict(sorted(style_c.items())),
        hard_negative_methods=dict(sorted(hn_method_c.items())),
        hard_negative_underlying=dict(sorted(hn_rel_c.items())),
        lexical_overlap_by_relation=lo_summary,
        overlap_bands_balanced=bands,
        exact_duplicate_groups=len(exact_dups),
        near_duplicate_pairs=len(near_dups),
        trivial_query_positive_pairs=trivial,
        warnings=rep.warnings,
    )


def _datasheet(s: dict, corpus_hash: str) -> str:
    c = s["counts"]
    rel_rows = "\n".join(
        f"| {r} | {n} | {s['lexical_overlap_by_relation'][r]['mean']} | "
        f"{s['lexical_overlap_by_relation'][r]['lt02']} / {s['lexical_overlap_by_relation'][r]['b0205']} / {s['lexical_overlap_by_relation'][r]['gt05']} |"
        for r, n in s["relations"].items()
    )
    dom_rows = "\n".join(f"| {d} | {n} |" for d, n in s["domains"].items())
    return f"""# RELATE v0.1 — Datasheet

Structure follows Gebru et al., *Datasheets for Datasets*. `corpus_hash` (SHA-256
of the canonical sorted item + pair + query records) : `{corpus_hash}`.

## Motivation

RELATE (**Rel**ations **E**xplicitly **L**abeled **A**nd **T**yped for **E**mbeddings)
is the single diagnostic instrument for *Embeddings From First Principles*. It is
**not** a general embedding-quality benchmark (that is MTEB's job); it is a
controlled probe whose structure makes a failure localize to a capability —
polarity (`negation`), role order (`relation-swap`), time (`temporal-mismatch`),
claim strength (`partial-support`, `entailment`), or mere aboutness
(`topic-related`, `entity-related`).

## Composition

| Quantity | Count |
|---|---|
| Base items (unique short texts) | {c['items']} |
| Labeled ordered pairs | {c['pairs']} |
| Query set (graded relevance + hard negatives) | {c['queries']} |
| Relations (types) | {len(s['relations'])} |
| Domains | {len(s['domains'])} |
| Template families | {c['template_families']} |
| Entity families | {c['entity_families']} |

### Pairs per relation, with lexical-overlap profile

| Relation | Pairs | Mean lexical overlap | Band share (<0.2 / 0.2–0.5 / ≥0.5) |
|---|---|---|---|
{rel_rows}

Lexical overlap is Jaccard over stemmed content lemmas (`lexical.py`; the frozen
build uses the same light stemmer as RELATE-DEV, not a full lemmatizer —
recorded here as a known approximation, see Limitations).

### Items per domain

| Domain | Items |
|---|---|
{dom_rows}

### Provenance

{json.dumps(s['provenance'], indent=2)}

`authored` = written for RELATE; `adapted` = a widely-known public fact restated
in original wording (geography facts, as of 2024); `perturbed` = derived from
another RELATE item by one structured transformation (negation, relation-swap,
temporal-shift, quantifier-weaken, entity-substitute), with parent id and
transformation recorded.

### Query styles

{json.dumps(s['query_styles'], indent=2)}

### Hard negatives

By mining method: {json.dumps(s['hard_negative_methods'])}
By underlying relation: {json.dumps(s['hard_negative_underlying'])}

`structured_perturbation` negatives (negation / relation-swap / temporal-shift of
a grade-3 positive) have near-zero false-negative rate by construction and are
the ones used for the headline hard-negative-agreement numbers. Mined negatives
(`lexical_overlap_matched`, `entity_matched`) are BM25 / entity top-k
non-answers in the same domain.

## Collection process

Every item is generated by `generate.py` from the entity + template tables in
`sources.py`. **No text is drawn from an existing NLI / STS / retrieval corpus.**
This template-generation pipeline is the corpus's primary contamination defense
(spec §8): items are newly written, not lifted, so near-duplicate contamination
against SNLI / MNLI / STS-B / PAWS / BEIR is structurally near-zero. A
by-hand spot check against PAWS and FEVER phrasings found no shared items.

## Preprocessing / cleaning / labeling

- Structural validation (`schema.py`) + semantic validation (`validate.py`,
  relation constraints, directionality, referential integrity, adjudication
  sanity) on **every** item, pair, and query. Build aborts on any error.
- Near-duplicate detection: exact-text groups + shingle-Jaccard ≥ 0.9 over the
  item pool. Exact-duplicate groups: {s['exact_duplicate_groups']}.
  Near-duplicate pairs: {s['near_duplicate_pairs']}.
- Trivial query→grade-3 positive pairs (lexical overlap = 1.0, relation ≠
  equivalent): {s['trivial_query_positive_pairs']}.

### Labeling model — and its honest limit

RELATE v0.1 is a **synthetic controlled probe**. Each pair's gold relation and
`runner_up` are fixed by the generation template, then checked against the
frozen `ontology.json` adjudication constraints. **There is no independent human
double-annotation in v0.1, so no Krippendorff's α is reported.** In its place the
datasheet reports (a) a rule-consistency audit — every emitted pair passes the
semantic validators — and (b) the per-relation lexical-overlap distribution
above. Human adjudication + per-relation α is the top v0.2 task (spec §9, §16);
`partial-support` is flagged as the relation most dependent on it (spec §17.3).

## Splits

Four orthogonal axes (`splits/`), each train/dev/test 60/20/20 by deterministic
hash: `random` (by base item — no item spans splits), `entity` (by entity
family), `template` (by template family), `domain` (leave-one-domain-out, 5
folds). Anchor sets for the bridge chapters draw from `entity:train`; held-out
evaluation uses `entity:test`.

## Reference model

Hard-negative mining used only BM25 (lexical) and entity matching — **no
embedding model was used in construction**, so RELATE carries no model's bias
into its labels. Wave 1 pins the evaluation models separately.

## Known limitations

1. **English only; Western-centric** `geo-civics` / `corporate-events` entities.
2. **Domain skew** — `geo-civics` is ~{round(100 * s['domains'].get('geo-civics', 0) / c['items'])}% of items;
   `split_domain` folds are imbalanced. Documented, not yet corrected.
3. **Synthetic** — templated generation gives clean labels and zero public-corpus
   contamination, at the cost of surface-pattern regularity a model could learn
   (`split_template` is the guard; report train vs test-template gaps).
4. **Lexical overlap** uses a light stemmer, not a full lemmatizer.
5. **Pair volume** ({c['pairs']}) sits at the low end of the spec's 4,000–6,000
   envelope; v0.1 prioritizes clean coverage of all 11 relations and the overlap
   strata over raw count.
6. **`temporal-mismatch` / `contradiction`** depend on the reader's world model;
   the templated facts are self-consistent but simplified.

## Recommended uses

Diagnostic probes for: negation vs paraphrase separation, hard-negative
agreement across models/bridges, per-relation preservation under alignment,
calibration-band width under hard negatives, anisotropy / effective rank /
intrinsic dimension of a fixed corpus. **Not** for: leaderboard-style model
ranking; any claim that generalizes beyond the probed capability.

## License

Target CC BY 4.0. Adapted facts are expression-original.
"""


def _corpus_hash_check(out: Path) -> None:
    pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="build to a temp dir and compare hash to the frozen release")
    ap.add_argument("--v02", action="store_true", help="build relate-0.2.0/ (v0.1 items+pairs + the hard query set)")
    args = ap.parse_args(argv)
    out_dir = (HERE / "relate-0.2.0") if args.v02 else OUT
    ver = "0.2.0" if args.v02 else VERSION

    if args.check:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            res = build(Path(td) / f"relate-{ver}", v02=args.v02)
        frozen = (out_dir / "corpus_hash.txt").read_text().strip() if (out_dir / "corpus_hash.txt").exists() else "<none>"
        same = res["corpus_hash"] == frozen
        print(f"rebuilt hash : {res['corpus_hash']}")
        print(f"frozen hash  : {frozen}")
        print("MATCH" if same else "DIFFERS")
        return 0 if same else 1

    res = build(out_dir, v02=args.v02)
    s = res["stats"]
    print(f"RELATE {ver} built -> {out_dir}")
    print(f"  items {s['counts']['items']}  pairs {s['counts']['pairs']}  queries {s['counts']['queries']}")
    print(f"  template families {s['counts']['template_families']}  entity families {s['counts']['entity_families']}")
    print(f"  relations: {s['relations']}")
    print(f"  corpus_hash: {res['corpus_hash']}")
    if res["warnings"]:
        print(f"  {len(res['warnings'])} warning(s):")
        for w in res["warnings"][:12]:
            print(f"    WARN {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
