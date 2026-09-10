"""RELATE v0.1 generator — stages 5-6 (overgenerate + balance by difficulty).

Emits candidate items and typed pairs whose relation label is *known by
construction* from the template scheme that produced them (spec §0: "designed so
that a failure on RELATE localizes to a capability"). `generate()` returns plain
dicts ready for `schema.Item` / `schema.Pair`; `build_full.py` runs the
validators, dedups, splits, and freezes.

Labeling model for v0.1 (recorded in the datasheet, honest about its limits):
RELATE v0.1 is a *synthetic controlled probe*. Each pair's gold relation and its
`runner_up` are fixed by the generation template, then every pair is checked
against the frozen `ontology.json` adjudication constraints (`validate.py`).
There is no human double-annotation in v0.1, so no Krippendorff's α is reported;
instead the datasheet reports a rule-consistency audit (every emitted pair
passes the semantic validators) and the per-relation lexical-overlap
distribution. Human adjudication + α is the top v0.2 task (spec §9, §16).
"""
from __future__ import annotations

import hashlib
import re

import lexical
import sources as S

_TOKEN = re.compile(r"[a-z0-9]+")


def _iid(text: str, tag: str) -> str:
    return "i-" + hashlib.sha1(f"{tag}|{text}".encode()).hexdigest()[:10]


def _norm_ent(e: str) -> str:
    return e.lower().strip()


class Corpus:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self.pairs: list[dict] = []
        self.qseeds: list[dict] = []
        self.hard_qseeds: list[dict] = []
        self.adjust_log: list[tuple] = []
        self._pairkey: set[tuple] = set()

    def qseed(self, text, style, target, positives, hard_negs=(), hard=False):
        rec = dict(
            text=text, query_style=style, target_item_id=target,
            positives=[dict(item_id=i, grade=g) for i, g in positives],
            hard_negatives=[dict(item_id=i, underlying_relation=r, method=m) for i, r, m in hard_negs],
        )
        (self.hard_qseeds if hard else self.qseeds).append(rec)

    def add_item(self, text, domain, tf, ef, entities, *, time_index=None,
                 atomic_claims=None, provenance=None) -> str:
        text = " ".join(text.split())
        iid = _iid(text, f"{tf}|{ef}")
        if iid not in self.items:
            self.items[iid] = dict(
                id=iid, text=text, domain=domain,
                template_family=f"tf-{tf}", entity_family=f"ef-{ef}",
                entities=list(entities),
                time_index=time_index,
                atomic_claims=list(atomic_claims or []),
                provenance=provenance or dict(kind="authored"),
            )
        return iid

    def pair(self, a, b, relation, runner_up, *, direction=None, notes=""):
        if a == b:
            return
        ia, ib = self.items[a], self.items[b]
        lo = round(lexical.jaccard(ia["text"], ib["text"]), 4)
        eo = lexical.entity_overlap(ia["entities"], ib["entities"])
        # spec-compliant auto-adjudication: a "negation" that is not lexically
        # close is really a `contradiction` (spec §3.6); a "relation-swap" with
        # no shared entity is really a `contradiction` (spec §3.11 near-miss).
        if relation == "negation" and lo < 0.7:
            relation, runner_up, direction = "contradiction", "negation", None
            notes = (notes + " [auto: negation->contradiction, overlap<0.7]").strip()
            self.adjust_log.append((a, b, "neg->contra", lo))
        if relation == "relation-swap" and eo < 1:
            relation, runner_up, direction = "contradiction", "relation-swap", None
            notes = (notes + " [auto: relation-swap->contradiction, no shared entity]").strip()
            self.adjust_log.append((a, b, "swap->contra", eo))
        key = (a, b, relation)
        if key in self._pairkey:
            return
        self._pairkey.add(key)
        self.pairs.append(dict(
            id="p-" + hashlib.sha1(f"{a}|{b}|{relation}".encode()).hexdigest()[:10],
            a_id=a, b_id=b, relation=relation, rule_fired=relation,
            runner_up=runner_up, direction=direction,
            lexical_overlap=lo, char_3gram_overlap=round(lexical.dice_3gram(ia["text"], ib["text"]), 4),
            entity_overlap=eo, notes=notes,
        ))


# --------------------------------------------------------------------------- #
# geo-civics
# --------------------------------------------------------------------------- #
def gen_geo_capitals(c: Corpus) -> None:
    for slug, country, cap, wrong, region, city2 in S.GEO_CAPITALS:
        ef = f"cap-{slug}"
        ents = [cap, country]
        adapted = dict(kind="adapted", source=f"public geography fact, as of 2024 ({country})")
        s1 = c.add_item(f"{cap} is the capital of {country}.", "geo-civics", "x-is-capital-of-y", ef, ents, provenance=adapted)
        s2 = c.add_item(f"The capital of {country} is {cap}.", "geo-civics", "capital-of-y-is-x", ef, ents, provenance=adapted)
        s3 = c.add_item(f"{cap} serves as the capital city of {country}.", "geo-civics", "x-serves-as-capital-of-y", ef, ents, provenance=adapted)
        s4 = c.add_item(f"The seat of government of {country} sits in {cap}.", "geo-civics", "seat-of-government-of-y-in-x", ef, ents, provenance=adapted)
        n1 = c.add_item(f"{cap} is not the capital of {country}.", "geo-civics", "x-is-capital-of-y", ef, ents,
                        provenance=dict(kind="perturbed", parent_id=s1, transformation="negation"))
        c1 = c.add_item(f"{wrong} is the capital of {country}.", "geo-civics", "x-is-capital-of-y", ef, [wrong, country],
                        provenance=dict(kind="perturbed", parent_id=s1, transformation="entity_substitute"))
        e1 = c.add_item(f"{cap} is a city in {country}.", "geo-civics", "x-is-a-city-in-y", ef, ents, provenance=adapted)
        e2 = c.add_item(f"{cap} lies within {country}'s borders.", "geo-civics", "x-lies-within-y", ef, ents, provenance=adapted)
        t1 = c.add_item(f"{country} is a country in {region}.", "geo-civics", "y-is-a-country-in-region", ef, [country], provenance=adapted)
        er1 = c.add_item(f"{cap} has an international airport.", "geo-civics", "x-has-an-airport", ef, [cap], provenance=dict(kind="authored"))
        er2 = c.add_item(f"{city2} is a city in {country}.", "geo-civics", "x-is-a-city-in-y", f"cap-{slug}", [city2, country], provenance=adapted)

        c.pair(s1, s2, "equivalent", "paraphrase", notes="word order only")
        c.pair(s2, s1, "equivalent", "paraphrase")
        c.pair(s1, s3, "paraphrase", "equivalent", notes="'serves as ... city' adds role framing (mid overlap)")
        c.pair(s1, s4, "paraphrase", "equivalent", notes="seat of government wording (low overlap)")
        c.pair(s3, s4, "paraphrase", "equivalent")
        c.pair(s1, n1, "negation", "contradiction", direction="a->b", notes="insert 'not' on main predicate")
        c.pair(s2, n1, "contradiction", "negation", notes="incompatible, not the minimal-edit form")
        c.pair(s1, c1, "contradiction", "topic-related", notes="rival city as capital; atemporal incompatibility, high overlap")
        c.pair(s4, c1, "contradiction", "topic-related", notes="same incompatibility, low overlap")
        c.pair(s1, e1, "entailment", "paraphrase", direction="a->b", notes="capital -> a city in the country (high overlap)")
        c.pair(s1, e2, "entailment", "paraphrase", direction="a->b", notes="capital -> lies within borders (mid/low overlap)")
        c.pair(s1, t1, "topic-related", "entity-related", notes="same country, independent claim")
        c.pair(s3, t1, "topic-related", "entity-related")
        c.pair(s1, er1, "entity-related", "topic-related", notes="shares the city, different topic (airport)")
        c.pair(s1, er2, "entity-related", "topic-related", notes="shares the country, different city/claim")

        hn = [(n1, "negation", "structured_perturbation"),
              (c1, "entity-related", "lexical_overlap_matched"),
              (er2, "entity-related", "entity_matched")]
        c.qseed(f"Is {cap} the capital of {country}?", "question", s1,
                [(s1, 3), (s2, 3), (s3, 3), (s4, 2), (e1, 2), (e2, 2), (t1, 1)], hn)
        c.qseed(f"capital of {country}", "keyword", s1, [(s1, 3), (s2, 3), (s3, 3), (s4, 2)], hn)
        c.qseed(f"I need to confirm which city is the capital of {country} for a form.", "long-nl", s1,
                [(s1, 3), (s2, 3), (s4, 2)], hn)
        c.qseed(f"{country}'s capital city is {cap}.", "restatement", s1,
                [(s1, 3), (s2, 3), (s3, 3), (s4, 2)], hn)
        # v0.2 HARD: indirect phrasing (no "capital"/city name), competing same-topic distractors
        hard_hn = [(n1, "negation", "structured_perturbation"),
                   (c1, "entity-related", "lexical_overlap_matched"),
                   (er1, "entity-related", "entity_matched"),
                   (er2, "entity-related", "entity_matched"),
                   (t1, "topic-related", "structured_perturbation")]
        c.qseed(f"A diplomat travelling to {country} for talks with the national government would be based in which city?",
                "long-nl", s1, [(s1, 3), (s2, 3), (s4, 2)], hard_hn, hard=True)
        c.qseed(f"Where does {country} run its ministries from?", "question", s1,
                [(s1, 3), (s2, 3), (s3, 3), (s4, 2)], hard_hn, hard=True)


def gen_geo_landlocked(c: Corpus) -> None:
    for slug, country, landlocked, neighbour, nowater in S.GEO_LANDLOCKED:
        ef = f"geo-{slug}"
        adapted = dict(kind="adapted", source=f"public geography fact, as of 2024 ({country})")
        if landlocked:
            s1 = c.add_item(f"{country} is a landlocked country.", "geo-civics", "x-is-landlocked", ef, [country], provenance=adapted)
            s2 = c.add_item(f"{country} has no coastline of its own.", "geo-civics", "x-has-no-coastline", ef, [country], provenance=adapted)
            cc = c.add_item(f"{country} has a long {nowater} coastline.", "geo-civics", "x-has-coastline-on-water", ef, [country], provenance=dict(kind="authored"))
            n1 = c.add_item(f"{country} is not a landlocked country.", "geo-civics", "x-is-landlocked", ef, [country],
                            provenance=dict(kind="perturbed", parent_id=s1, transformation="negation"))
            c.pair(s1, s2, "paraphrase", "equivalent", notes="landlocked == no coastline (low overlap)")
            c.pair(s1, cc, "contradiction", "topic-related", notes="landlocked vs long coastline")
            c.pair(s2, cc, "contradiction", "topic-related")
            c.pair(s1, n1, "negation", "contradiction", direction="a->b")
        else:
            s1 = c.add_item(f"{country} has a coastline.", "geo-civics", "x-has-coastline", ef, [country], provenance=adapted)
            n1 = c.add_item(f"{country} does not have a coastline.", "geo-civics", "x-has-coastline", ef, [country],
                            provenance=dict(kind="perturbed", parent_id=s1, transformation="negation"))
            ll = c.add_item(f"{country} is a landlocked country.", "geo-civics", "x-is-landlocked", ef, [country], provenance=dict(kind="authored"))
            c.pair(s1, n1, "negation", "contradiction", direction="a->b")
            c.pair(s1, ll, "contradiction", "topic-related", notes="has a coastline vs landlocked")
        b1 = c.add_item(f"{country} shares a border with {neighbour}.", "geo-civics", "x-borders-y", ef, [country, neighbour], provenance=adapted)
        b2 = c.add_item(f"{neighbour} shares a border with {country}.", "geo-civics", "x-borders-y", ef, [country, neighbour], provenance=adapted)
        c.pair(b1, b2, "equivalent", "paraphrase", notes="border is symmetric; role order carries no truth change")


def gen_geo_direction(c: Corpus) -> None:
    # directional geography -> clean relation-swap (asymmetric relation)
    DIRS = [
        ("iberia", "Portugal", "Spain", "west"),
        ("scandinavia", "Norway", "Sweden", "west"),
        ("benelux", "Belgium", "Germany", "west"),
        ("baltics", "Estonia", "Latvia", "north"),
        ("andes", "Ecuador", "Peru", "north"),
        ("maghreb", "Morocco", "Algeria", "west"),
        ("horn", "Djibouti", "Ethiopia", "east"),
        ("mekong", "Laos", "Vietnam", "west"),
    ]
    opp = {"west": "east", "east": "west", "north": "south", "south": "north"}
    for slug, x, y, d in DIRS:
        ef = f"dir-{slug}"
        adapted = dict(kind="adapted", source="public geography fact, as of 2024")
        s1 = c.add_item(f"{x} lies to the {d} of {y}.", "geo-civics", "x-lies-direction-of-y", ef, [x, y], provenance=adapted)
        sw = c.add_item(f"{y} lies to the {d} of {x}.", "geo-civics", "x-lies-direction-of-y", ef, [x, y],
                        provenance=dict(kind="perturbed", parent_id=s1, transformation="relation_swap"))
        eq = c.add_item(f"{y} lies to the {opp[d]} of {x}.", "geo-civics", "x-lies-direction-of-y", ef, [x, y], provenance=adapted)
        para = c.add_item(f"{x} is {d} of {y}.", "geo-civics", "x-is-direction-of-y", ef, [x, y], provenance=adapted)
        n1 = c.add_item(f"{x} does not lie to the {d} of {y}.", "geo-civics", "x-lies-direction-of-y", ef, [x, y],
                        provenance=dict(kind="perturbed", parent_id=s1, transformation="negation"))
        c.pair(s1, sw, "relation-swap", "contradiction", direction="a->b", notes="roles reversed; also a contradiction")
        c.pair(s1, eq, "equivalent", "paraphrase", notes="X d of Y == Y opp(d) of X")
        c.pair(s1, para, "paraphrase", "equivalent")
        c.pair(s1, n1, "negation", "contradiction", direction="a->b")


def gen_geo_pop(c: Corpus) -> None:
    for slug, place, series in S.GEO_POP:
        ef = f"pop-{slug}"
        adapted = dict(kind="adapted", source=f"illustrative municipal population series ({place})")
        items = []
        for year, pop in series:
            it = c.add_item(f"As of the {year} count, {place} had a population of about {pop}.",
                            "geo-civics", "place-population-at-time", ef, [place], time_index=str(year), provenance=adapted)
            items.append((year, it))
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                c.pair(items[i][1], items[j][1], "temporal-mismatch", "contradiction", direction="a->b",
                       notes=f"same city population, {items[i][0]} vs {items[j][0]}")
        # a topic-related, same place different claim
        tr = c.add_item(f"{place} sits on a river and floods in spring.", "geo-civics", "place-geography-note", ef, [place], provenance=dict(kind="authored"))
        c.pair(items[0][1], tr, "topic-related", "entity-related", notes="same place, independent claim")

        ty, tid = items[-1]
        hn = [(items[k][1], "temporal-mismatch", "structured_perturbation") for k in range(len(items) - 1)]
        c.qseed(f"What was the population of {place} in {ty}?", "question", tid, [(tid, 3)], hn)
        c.qseed(f"{place} population {ty}", "keyword", tid, [(tid, 3)], hn)
        # v0.2 HARD: temporal qualification with indirect phrasing
        c.qseed(f"How many people lived in {place} at the most recent count?", "question", tid,
                [(tid, 3)], hn + [(tr, "topic-related", "entity_matched")], hard=True)
        c.qseed(f"Give the {place} headcount from the {ty} census, not the earlier ones.", "long-nl", tid,
                [(tid, 3)], hn, hard=True)


def gen_geo_civics(c: Corpus) -> None:
    for slug, country, chambers, seats, org, yr in S.GEO_CIVICS:
        ef = f"civ-{slug}"
        au = dict(kind="authored")
        ch = c.add_item(f"The national parliament of {country} has {'two chambers' if chambers == 2 else 'a single chamber'}.",
                        "geo-civics", "parliament-structure", ef, [country], provenance=au)
        se = c.add_item(f"The lower house of {country}'s parliament has {seats} seats.",
                        "geo-civics", "parliament-seats", ef, [country], provenance=au)
        jo = c.add_item(f"{country} joined {org} in {yr}.", "geo-civics", "y-joined-org-in-year", ef, [country], time_index=str(yr), provenance=au)
        c.pair(ch, se, "topic-related", "entailment", notes="both about the same parliament; independent claims (chambers vs seat count)")
        c.pair(ch, jo, "entity-related", "topic-related", notes="shares the country; parliament structure vs treaty membership")
        c.pair(se, jo, "entity-related", "topic-related")


def gen_geo_treaty(c: Corpus) -> None:
    TREATIES = [
        ("union-1957", "the 1957 founding treaty", 1957, "created a customs union", "abolished internal tariffs between members", "six founding states"),
        ("accord-1994", "the 1994 trade accord", 1994, "opened agricultural markets", "set a common external tariff", "three signatory states"),
        ("pact-2003", "the 2003 mobility pact", 2003, "removed passport checks at internal borders", "established a shared visa list", "nine member states"),
    ]
    for slug, name, yr, prov1, prov2, signed in TREATIES:
        ef = f"treaty-{slug}"
        au = dict(kind="authored")
        multi = c.add_item(f"{name.capitalize()} {prov1} and {prov2}.", "geo-civics", "treaty-multi-provision", ef, [name],
                           time_index=str(yr), atomic_claims=[f"{name} {prov1}", f"{name} {prov2}"], provenance=au)
        one = c.add_item(f"Under {name}, {prov1.split(' ', 1)[1] if ' ' in prov1 else prov1} was the immediate effect.",
                         "geo-civics", "treaty-single-provision", ef, [name], provenance=au)
        sig = c.add_item(f"{name.capitalize()} was signed by {signed}.", "geo-civics", "treaty-signatories", ef, [name], provenance=au)
        neg = c.add_item(f"{name.capitalize()} did not {prov1} or {prov2}.", "geo-civics", "treaty-multi-provision", ef, [name],
                         provenance=dict(kind="perturbed", parent_id=multi, transformation="negation"))
        c.pair(multi, one, "partial-support", "entailment", direction="a->b", notes="b supports provision 1, silent on provision 2")
        c.pair(multi, sig, "topic-related", "entity-related", notes="same treaty; provisions vs signatories (high overlap, independent)")
        c.pair(one, sig, "topic-related", "entity-related")
        c.pair(multi, neg, "negation", "contradiction", direction="a->b", notes="De Morgan of the conjunction (spec §17.1)")


# --------------------------------------------------------------------------- #
# corporate-events
# --------------------------------------------------------------------------- #
def gen_corp_acq(c: Corpus) -> None:
    for slug, acq, tgt, yr, struct, sector in S.CORP_ACQ:
        ef = f"acq-{slug}"
        ents = [acq, tgt]
        au = dict(kind="authored")
        s1 = c.add_item(f"{acq} acquired {tgt} in {struct}.", "corporate-events", "x-acquired-y", ef, ents, time_index=str(yr), provenance=au)
        s2 = c.add_item(f"{tgt} was acquired by {acq} in {struct}.", "corporate-events", "y-was-acquired-by-x", ef, ents, time_index=str(yr), provenance=au)
        s3 = c.add_item(f"{acq} took over {tgt} in {struct}.", "corporate-events", "x-took-over-y", ef, ents, provenance=au)
        s4 = c.add_item(f"{tgt} became part of {acq} that year.", "corporate-events", "y-became-part-of-x", ef, ents, provenance=au)
        sw = c.add_item(f"{tgt} acquired {acq} in {struct}.", "corporate-events", "x-acquired-y", ef, ents,
                        provenance=dict(kind="perturbed", parent_id=s1, transformation="relation_swap"))
        neg = c.add_item(f"{acq} did not acquire {tgt} in {struct}.", "corporate-events", "x-acquired-y", ef, ents,
                         provenance=dict(kind="perturbed", parent_id=s1, transformation="negation"))
        ceo = c.add_item(f"{acq} appointed a new chief financial officer in {yr + 1}.", "corporate-events", "x-appointed-officer", ef, [acq], provenance=au)
        tr = c.add_item(f"{acq} and {tgt} had competed in {sector} for a decade.", "corporate-events", "x-and-y-competed", ef, ents, provenance=au)
        c.pair(s1, s2, "equivalent", "paraphrase", notes="active/passive, no nuance change")
        c.pair(s2, s1, "equivalent", "paraphrase")
        c.pair(s1, s3, "paraphrase", "equivalent", notes="acquired vs took over")
        c.pair(s1, s4, "paraphrase", "equivalent", notes="low overlap restatement")
        c.pair(s1, sw, "relation-swap", "contradiction", direction="a->b", notes="acquirer/target reversed")
        c.pair(s2, sw, "contradiction", "relation-swap", notes="incompatible, not the role-reversed surface form")
        c.pair(s1, neg, "negation", "contradiction", direction="a->b")
        c.pair(s1, ceo, "entity-related", "topic-related", notes="shares the acquirer; deal vs officer appointment")
        c.pair(s1, tr, "topic-related", "entity-related", notes="same two firms and sector; independent claim")
        c.pair(s3, tr, "topic-related", "entity-related")

        hn = [(sw, "relation-swap", "structured_perturbation"),
              (neg, "negation", "structured_perturbation"),
              (ceo, "entity-related", "entity_matched")]
        c.qseed(f"Which company acquired {tgt}?", "question", s1, [(s1, 3), (s2, 3), (s3, 3), (s4, 2)], hn)
        c.qseed(f"{tgt} acquisition", "keyword", s1, [(s1, 3), (s2, 3), (s3, 3)], hn)
        c.qseed(f"{acq} bought {tgt}.", "restatement", s1, [(s1, 3), (s2, 3), (s3, 3), (s4, 2)], hn)
        # v0.2 HARD: role-disambiguation + competing same-sector distractors
        hard_hn = [(sw, "relation-swap", "structured_perturbation"),
                   (neg, "negation", "structured_perturbation"),
                   (ceo, "entity-related", "entity_matched"),
                   (tr, "topic-related", "structured_perturbation")]
        c.qseed(f"After the {sector} tie-up involving {tgt}, which firm ended up owning the other?",
                "long-nl", s1, [(s1, 3), (s2, 3), (s4, 2)], hard_hn, hard=True)
        c.qseed(f"Who took control when {tgt} changed hands?", "question", s1,
                [(s1, 3), (s2, 3), (s3, 3)], hard_hn, hard=True)


def gen_corp_head(c: Corpus) -> None:
    for slug, firm, series, event, ev_yr in S.CORP_HEAD:
        ef = f"head-{slug}"
        au = dict(kind="authored")
        items = []
        for year, hc in series:
            it = c.add_item(f"At the end of {year}, {firm} employed {hc} people.", "corporate-events", "firm-headcount-at-time", ef, [firm],
                            time_index=str(year), provenance=au)
            items.append((year, it))
        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:
                    c.pair(items[i][1], items[j][1], "temporal-mismatch", "contradiction", direction="a->b",
                           notes=f"same firm headcount, {items[i][0]} vs {items[j][0]}")
        ev = c.add_item(f"{firm} {event} in {ev_yr}.", "corporate-events", "firm-governance-event", ef, [firm], time_index=str(ev_yr), provenance=au)
        c.pair(items[0][1], ev, "entity-related", "topic-related", notes="shares the firm; headcount vs governance event")


# --------------------------------------------------------------------------- #
# biomed-claims
# --------------------------------------------------------------------------- #
def gen_bio_multi(c: Corpus) -> None:
    for slug, agent, e1, e2, pop, cond in S.BIO_MULTI:
        ef = f"bio-{slug}"
        au = dict(kind="authored")
        multi = c.add_item(f"In a study of {pop}, {agent} {e1} and {e2}.", "biomed-claims", "agent-multi-effect-population", ef, [agent],
                           atomic_claims=[f"{agent} {e1} in {pop}", f"{agent} {e2} in {pop}"], provenance=au)
        one = c.add_item(f"In a trial enrolling {pop}, {agent} {e1}.", "biomed-claims", "trial-single-effect", ef, [agent], provenance=au)
        para = c.add_item(f"Among {pop}, {agent} both {e1} and {e2}.", "biomed-claims", "agent-both-effects", ef, [agent], provenance=au)
        neg = c.add_item(f"In a study of {pop}, {agent} did not {_deverb(e1)} or {_deverb(e2)}.", "biomed-claims", "agent-multi-effect-population", ef, [agent],
                         provenance=dict(kind="perturbed", parent_id=multi, transformation="negation"))
        ent = c.add_item(f"{agent} was evaluated for {cond}.", "biomed-claims", "agent-evaluated-for-condition", ef, [agent], provenance=au)
        weaker = c.add_item(f"In a subgroup of {pop}, {agent} {e1}; the overall effect was not significant.",
                            "biomed-claims", "agent-subgroup-effect", ef, [agent], provenance=au)
        c.pair(multi, one, "partial-support", "entailment", direction="a->b", notes="b supports effect 1, silent on effect 2")
        c.pair(multi, para, "paraphrase", "equivalent", notes="both effects, reworded")
        c.pair(multi, neg, "negation", "contradiction", direction="a->b", notes="De Morgan of the two effects")
        c.pair(multi, ent, "entailment", "topic-related", direction="a->b", notes="studied both effects -> was evaluated for the condition")
        c.pair(multi, weaker, "partial-support", "contradiction", direction="a->b", notes="b supports effect 1 in a subgroup, undercuts the general claim without negating it")
        c.pair(one, ent, "entailment", "topic-related", direction="a->b")

        hn = [(neg, "negation", "structured_perturbation")]
        c.qseed(f"Did {agent} {e1} and {e2} in {pop}?", "question", multi,
                [(multi, 3), (para, 3), (one, 2), (weaker, 2)], hn)
        c.qseed(f"{agent} effect on {pop}", "keyword", multi, [(multi, 3), (para, 3), (one, 2)], hn)
        # v0.2 HARD: claim-strength distinction (full vs subgroup vs negative)
        hard_hn = [(neg, "negation", "structured_perturbation"),
                   (weaker, "partial-support", "structured_perturbation"),
                   (ent, "topic-related", "entity_matched")]
        c.qseed(f"Did the {cond} study find {agent} worked on both measures for the whole group, not just a subset?",
                "long-nl", multi, [(multi, 3), (para, 3)], hard_hn, hard=True)
        c.qseed(f"What did the research show {agent} does for people with {cond}?", "question", multi,
                [(multi, 3), (para, 3), (one, 2)], hard_hn, hard=True)


def _deverb(phrase: str) -> str:
    """'lowered blood pressure' -> 'lower blood pressure' for a negation edit."""
    w, rest = (phrase.split(" ", 1) + [""])[:2]
    irregular = {"raised": "raise", "cut": "cut", "shortened": "shorten", "eased": "ease",
                 "lowered": "lower", "reduced": "reduce", "improved": "improve", "increased": "increase",
                 "delayed": "delay", "decreased": "decrease"}
    return (irregular.get(w, w[:-2] if w.endswith("ed") else w) + " " + rest).strip()


def gen_bio_time(c: Corpus) -> None:
    for slug, subj, effect, series, size in S.BIO_TIME:
        ef = f"biotime-{slug}"
        au = dict(kind="authored")
        items = []
        for year, verdict in series:
            it = c.add_item(f"A {year} analysis of {subj} reported {verdict}.", "biomed-claims", "analysis-verdict-at-time", ef, [subj],
                            time_index=str(year), provenance=au)
            items.append((year, it))
        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:
                    c.pair(items[i][1], items[j][1], "temporal-mismatch", "contradiction", direction="a->b",
                           notes=f"same cohort re-analysed, {items[i][0]} vs {items[j][0]}")
        tr = c.add_item(f"The {subj} was recruited from three hospitals.", "biomed-claims", "cohort-recruitment-note", ef, [subj], provenance=au)
        c.pair(items[0][1], tr, "topic-related", "entity-related", notes="same cohort, independent methods claim")


# --------------------------------------------------------------------------- #
# product-support
# --------------------------------------------------------------------------- #
def gen_prod(c: Corpus) -> None:
    for slug, prod, fault, cond, ver, feat, loc in S.PROD:
        ef = f"prod-{slug}"
        au = dict(kind="authored")
        s1 = c.add_item(f"{prod} {fault} {cond}", "product-support", "device-fault-condition", ef, [prod], provenance=au)
        s2 = c.add_item(f"every time I try, {prod} {fault} {cond}", "product-support", "device-fault-condition-user", ef, [prod], provenance=au)
        s3 = c.add_item(f"having trouble because {prod} {fault} {cond} and I cannot get past it", "product-support", "device-fault-condition-verbose", ef, [prod], provenance=au)
        fixed = c.add_item(f"{prod} no longer {fault} {cond} since I installed {ver}", "product-support", "device-fault-resolved", ef, [prod],
                           provenance=dict(kind="perturbed", parent_id=s1, transformation="negation"))
        how = c.add_item(f"how do I turn on {feat} for {prod}", "product-support", "howto-feature", ef, [prod], provenance=au)
        where = c.add_item(f"{feat} for {prod} is under {loc}", "product-support", "feature-location", ef, [prod], provenance=au)
        c.pair(s1, s2, "paraphrase", "equivalent", notes="same complaint, user rewording (mid overlap)")
        c.pair(s1, s3, "paraphrase", "equivalent", notes="verbose restatement (high overlap)")
        c.pair(s2, s3, "paraphrase", "equivalent")
        c.pair(s1, fixed, "negation", "contradiction", direction="a->b", notes="fault vs fault resolved, high lexical overlap")
        c.pair(how, where, "entailment", "topic-related", direction="a->b", notes="answer location entails the feature exists")
        c.pair(s1, how, "entity-related", "topic-related", notes="same product; a fault report vs a how-to question")
        c.pair(s1, where, "entity-related", "topic-related")
        c.pair(fixed, how, "entity-related", "topic-related")

        hn = [(fixed, "negation", "structured_perturbation"),
              (how, "entity-related", "entity_matched")]
        c.qseed(f"why does {prod} {fault} {cond}", "keyword", s1, [(s1, 3), (s2, 3), (s3, 3)], hn)
        c.qseed(f"{prod} {fault}", "keyword", s1, [(s1, 3), (s2, 3), (s3, 2)], hn)


# --------------------------------------------------------------------------- #
# everyday-statements
# --------------------------------------------------------------------------- #
def gen_everyday(c: Corpus) -> None:
    syn = {"rejected": "declined", "approved": "cleared", "spiked": "killed", "shortlisted": "advanced",
           "declined": "refused", "granted": "approved", "allowed": "permitted", "dismissed": "threw out"}
    for slug, subj, verb, obj, adverb, passive in S.EVERYDAY:
        ef = f"eve-{slug}"
        au = dict(kind="authored")
        base = c.add_item(f"{subj.capitalize()} {verb} {obj}.", "everyday-statements", "body-verbed-thing", ef, [subj], provenance=au)
        pas = c.add_item(f"{passive.capitalize()}.", "everyday-statements", "thing-verbed-by-body-passive", ef, [subj], provenance=au)
        syn_s = c.add_item(f"{subj.capitalize()} {syn.get(verb, verb)} {obj}.", "everyday-statements", "body-verbed-thing", ef, [subj], provenance=au)
        adv = c.add_item(f"{subj.capitalize()} {verb} {obj} {adverb}.", "everyday-statements", "body-verbed-thing-adverb", ef, [subj], provenance=au)
        neg = c.add_item(f"{subj.capitalize()} did not {_present(verb)} {obj}.", "everyday-statements", "body-verbed-thing", ef, [subj],
                         provenance=dict(kind="perturbed", parent_id=base, transformation="negation"))
        c.pair(base, pas, "equivalent", "paraphrase", notes="active/passive")
        c.pair(base, syn_s, "paraphrase", "equivalent", notes="verb synonym (mid overlap)")
        c.pair(adv, base, "entailment", "paraphrase", direction="a->b", notes="'X verbed Y <adverb>' -> 'X verbed Y'; reverse fails")
        c.pair(base, neg, "negation", "contradiction", direction="a->b")
        c.pair(pas, neg, "contradiction", "negation")

        c.qseed(f"Did {subj} {_present(verb)} {obj}?", "question", base,
                [(base, 3), (pas, 3), (syn_s, 3), (adv, 2)],
                [(neg, "negation", "structured_perturbation")])


def _present(verb_past: str) -> str:
    irr = {"rejected": "reject", "approved": "approve", "spiked": "spike", "shortlisted": "shortlist",
           "declined": "decline", "granted": "grant", "allowed": "allow", "dismissed": "dismiss"}
    return irr.get(verb_past, verb_past[:-2] if verb_past.endswith("ed") else verb_past)


def gen_everyday_scenes(c: Corpus) -> None:
    for slug, spec, state, loc, gen in S.EVERYDAY_SCENES:
        ef = f"scene-{slug}"
        au = dict(kind="authored")
        s1 = c.add_item(f"{spec.capitalize()} {state} {loc}.", "everyday-statements", "specific-entity-state-location", ef, [], provenance=au)
        g1 = c.add_item(f"{gen.capitalize()} is {loc}.", "everyday-statements", "general-entity-location", ef, [], provenance=au)
        g2 = c.add_item(f"There is {gen} somewhere {loc.split()[0]} the {loc.split()[-1]}.", "everyday-statements", "general-entity-vague-location", ef, [], provenance=au)
        neg = c.add_item(f"{spec.capitalize()} {state.replace('is ', 'is not ')} {loc}.", "everyday-statements", "specific-entity-state-location", ef, [],
                         provenance=dict(kind="perturbed", parent_id=s1, transformation="negation"))
        c.pair(s1, g1, "entailment", "paraphrase", direction="a->b", notes="specific + state -> general entity at location")
        c.pair(s1, g2, "entailment", "topic-related", direction="a->b", notes="low overlap entailment")
        c.pair(s1, neg, "negation", "contradiction", direction="a->b")


# --------------------------------------------------------------------------- #
# unrelated (cross-domain) — built by build_full after all items exist
# --------------------------------------------------------------------------- #
def gen_compete(c: Corpus) -> None:
    for slug, win, lose, event, margin, domain in S.COMPETE:
        ef = f"compete-{slug}"
        au = dict(kind="authored")
        s1 = c.add_item(f"{win.capitalize()} beat {lose} in {event}.", domain, "x-beat-y-in-event", ef, [win, lose], provenance=au)
        s2 = c.add_item(f"{lose.capitalize()} lost to {win} in {event}.", domain, "y-lost-to-x-in-event", ef, [win, lose], provenance=au)
        s3 = c.add_item(f"{win.capitalize()} won {event}, {margin}.", domain, "x-won-event-margin", ef, [win, lose], provenance=au)
        sw = c.add_item(f"{lose.capitalize()} beat {win} in {event}.", domain, "x-beat-y-in-event", ef, [win, lose],
                        provenance=dict(kind="perturbed", parent_id=s1, transformation="relation_swap"))
        neg = c.add_item(f"{win.capitalize()} did not beat {lose} in {event}.", domain, "x-beat-y-in-event", ef, [win, lose],
                         provenance=dict(kind="perturbed", parent_id=s1, transformation="negation"))
        tr = c.add_item(f"{win.capitalize()} and {lose} had met twice before {event}.", domain, "x-and-y-met-before", ef, [win, lose], provenance=au)
        c.pair(s1, s2, "equivalent", "paraphrase", notes="beat == lost to, roles consistent")
        c.pair(s1, s3, "entailment", "paraphrase", direction="a->b", notes="won by <margin> -> won/beat")
        c.pair(s3, s1, "entailment", "paraphrase", direction="a->b")
        c.pair(s1, sw, "relation-swap", "contradiction", direction="a->b", notes="winner/loser reversed")
        c.pair(s2, sw, "contradiction", "relation-swap")
        c.pair(s1, neg, "negation", "contradiction", direction="a->b")
        c.pair(s1, tr, "topic-related", "entity-related", notes="same two parties and event; independent history claim")
        c.pair(s3, tr, "topic-related", "entity-related")

        hn = [(sw, "relation-swap", "structured_perturbation"),
              (neg, "negation", "structured_perturbation")]
        c.qseed(f"Who won {event}?", "question", s1, [(s1, 3), (s2, 3), (s3, 3)], hn)
        c.qseed(f"{event} result", "keyword", s1, [(s1, 3), (s2, 3), (s3, 3)], hn)
        # v0.2 HARD: winner/loser disambiguation, indirect
        hard_hn = [(sw, "relation-swap", "structured_perturbation"), (neg, "negation", "structured_perturbation"),
                   (tr, "topic-related", "structured_perturbation")]
        c.qseed(f"Between {win} and {lose}, which side came out on top in {event}?", "long-nl", s1,
                [(s1, 3), (s2, 3), (s3, 3)], hard_hn, hard=True)


def gen_overlap_fillers(c: Corpus) -> None:
    """Stage 6 — balance by difficulty. Hand-authored pairs that deliberately
    fill the low / mid / high lexical-overlap bands the templated generators
    under-fill (DEV_FINDINGS §6): low-overlap paraphrase + entailment (same
    meaning, disjoint vocabulary), high-overlap topic-related (near-identical
    surface, logically independent claims), mid-overlap contradiction.
    """
    au = dict(kind="authored")

    # low-overlap PARAPHRASE (same proposition, minimal shared content words)
    LOW_PARA = [
        ("The board postponed the vote.", "Directors pushed the decision to a later meeting.", "corporate-events"),
        ("Sales fell sharply last quarter.", "Revenue dropped a lot in the last three months.", "corporate-events"),
        ("The bridge is closed for repairs.", "Crews have shut the crossing while they fix it.", "geo-civics"),
        ("The application was rejected.", "They turned the request down.", "everyday-statements"),
        ("The flight was cancelled.", "The airline scrapped the service.", "everyday-statements"),
        ("The trial found no benefit.", "The study showed the treatment did not help.", "biomed-claims"),
        ("The device overheats under load.", "It gets too hot when you push it hard.", "product-support"),
        ("The company is hiring aggressively.", "The firm is bringing on staff at a rapid pace.", "corporate-events"),
        ("The river burst its banks.", "The waterway flooded the surrounding land.", "geo-civics"),
        ("The report was leaked before publication.", "Someone released the document early.", "corporate-events"),
        ("The medication must be taken with food.", "Patients should not use the drug on an empty stomach.", "biomed-claims"),
        ("The update fixed the crash.", "After the patch, the program stopped failing.", "product-support"),
        ("The strike ended after nine days.", "Workers went back on the job following a week and a half of action.", "corporate-events"),
        ("Attendance doubled this year.", "Twice as many people showed up compared with last time.", "everyday-statements"),
        ("The manuscript was accepted without revisions.", "Reviewers approved the paper as submitted.", "biomed-claims"),
        ("The tunnel project is over budget.", "The underground crossing has cost more than planned.", "geo-civics"),
        ("The battery lasts about a day.", "You get roughly twenty-four hours of use per charge.", "product-support"),
        ("The council raised parking fees.", "It now costs more to leave a car in town.", "geo-civics"),
        ("The film was panned by critics.", "Reviewers gave the movie poor notices.", "everyday-statements"),
        ("The merger cleared regulators.", "Competition authorities let the tie-up proceed.", "corporate-events"),
        ("The crop failed this season.", "Farmers brought in almost nothing from the fields this year.", "geo-civics"),
        ("The server was down for an hour.", "The service was unavailable for sixty minutes.", "product-support"),
        ("The senator withdrew from the race.", "She pulled out of the contest.", "everyday-statements"),
        ("The vaccine requires two doses.", "You need to be immunised twice for full protection.", "biomed-claims"),
        ("The store is closing all its branches.", "Every outlet in the chain will shut.", "corporate-events"),
        ("The lake froze early this winter.", "Ice covered the water sooner than usual this year.", "geo-civics"),
        ("The keynote was moved online.", "Organisers switched the main talk to a virtual format.", "everyday-statements"),
    ]
    for i, (t1, t2, dom) in enumerate(LOW_PARA):
        ef = f"lowpara-{i:02d}"
        a = c.add_item(t1, dom, "lowpara-a", ef, [], provenance=au)
        b = c.add_item(t2, dom, "lowpara-b", ef, [], provenance=au)
        c.pair(a, b, "paraphrase", "equivalent", notes="low-overlap paraphrase (disjoint vocabulary, same claim)")
        c.pair(b, a, "paraphrase", "equivalent")

    # low-overlap ENTAILMENT (a -> b, very different wording)
    LOW_ENT = [
        ("The vault was emptied overnight.", "Money is missing.", "corporate-events"),
        ("Every window on the ground floor was shattered.", "The building was damaged.", "geo-civics"),
        ("She scored a century before lunch.", "She batted well.", "everyday-statements"),
        ("The reactor was shut down within minutes.", "Operators responded quickly.", "geo-civics"),
        ("The patient's fever broke on the third day.", "The patient improved.", "biomed-claims"),
        ("The app was downloaded ten million times in a week.", "The app was popular.", "product-support"),
        ("The negotiations collapsed without a signature.", "No deal was reached.", "corporate-events"),
        ("The hikers were airlifted off the ridge at dusk.", "The hikers were rescued.", "everyday-statements"),
        ("Frost killed the seedlings before dawn.", "The plants died.", "everyday-statements"),
        ("The keynote ran forty minutes over its slot.", "The talk was long.", "everyday-statements"),
        ("The auditors flagged seventeen irregular transactions.", "Something was wrong with the accounts.", "corporate-events"),
        ("The glacier retreated two kilometres in a decade.", "The ice is shrinking.", "geo-civics"),
        ("He solved the puzzle in under a minute.", "He is good at puzzles.", "everyday-statements"),
        ("The dose was halved after the side effects appeared.", "The treatment was adjusted.", "biomed-claims"),
        ("The queue stretched around the block before opening.", "Many people were waiting.", "everyday-statements"),
        ("The firmware bricked every unit in the field test.", "The update caused serious problems.", "product-support"),
        ("The dam released water for six hours straight.", "A lot of water was let out.", "geo-civics"),
        ("The startup burned through its funding in eight months.", "The company spent its money quickly.", "corporate-events"),
        ("Turnout topped ninety percent in three districts.", "Many people voted.", "geo-civics"),
        ("The samples were contaminated before analysis.", "The results are unreliable.", "biomed-claims"),
    ]
    for i, (t1, t2, dom) in enumerate(LOW_ENT):
        ef = f"lowent-{i:02d}"
        a = c.add_item(t1, dom, "lowent-a", ef, [], provenance=au)
        b = c.add_item(t2, dom, "lowent-b", ef, [], provenance=au)
        c.pair(a, b, "entailment", "paraphrase", direction="a->b", notes="low-overlap strict entailment; reverse clearly fails")

    # high-overlap TOPIC-RELATED (near-identical surface, independent claims)
    HIGH_TOPIC = [
        ("The first chapter of the report covers data collection.", "The first chapter of the report covers the budget.", "corporate-events"),
        ("The northern line runs every ten minutes on weekdays.", "The northern line runs every twenty minutes on weekends.", "geo-civics"),
        ("The 2019 survey recorded a rise in cycling.", "The 2019 survey recorded a fall in car ownership.", "geo-civics"),
        ("The committee met on Tuesday to discuss staffing.", "The committee met on Thursday to discuss funding.", "everyday-statements"),
        ("The drug is taken once a day in the morning.", "The drug is stored in a refrigerator between uses.", "biomed-claims"),
        ("The museum's east wing houses the ceramics collection.", "The museum's east wing houses the print collection.", "geo-civics"),
        ("The release notes mention a new dark mode.", "The release notes mention a new export format.", "product-support"),
        ("The quarterly report lists three new hires in sales.", "The quarterly report lists two office closures in the north.", "corporate-events"),
        ("The park closes at dusk in winter.", "The park closes at ten in summer.", "geo-civics"),
        ("Section two of the contract sets the payment schedule.", "Section two of the contract sets the delivery schedule.", "corporate-events"),
        ("The trial's primary endpoint was blood pressure.", "The trial's secondary endpoint was resting heart rate.", "biomed-claims"),
        ("The building's third floor is offices.", "The building's third floor has a server room.", "corporate-events"),
        ("The grant covers salaries for two years.", "The grant covers equipment for two years.", "biomed-claims"),
        ("The east platform is closed for lift repairs.", "The east platform is closed for tiling work.", "geo-civics"),
        ("The onboarding guide explains the leave policy.", "The onboarding guide explains the expenses policy.", "corporate-events"),
        ("The morning session focuses on data handling.", "The afternoon session focuses on report writing.", "everyday-statements"),
        ("The north car park has two hundred spaces.", "The south car park has ninety spaces.", "geo-civics"),
        ("The app's settings page has a language option.", "The app's settings page has a backup option.", "product-support"),
        ("The first phase of the trial enrolled adults.", "The second phase of the trial enrolled adolescents.", "biomed-claims"),
        ("Chapter three of the manual covers installation.", "Chapter four of the manual covers maintenance.", "product-support"),
        ("The annual review praised the sales team.", "The annual review criticised the logistics team.", "corporate-events"),
        ("The reservoir supplies the eastern suburbs.", "The reservoir supplies the industrial park.", "geo-civics"),
        ("The playlist opens with a slow track.", "The playlist closes with an upbeat track.", "everyday-statements"),
        ("The budget allocates funds to road repair.", "The budget allocates funds to street lighting.", "geo-civics"),
    ]
    for i, (t1, t2, dom) in enumerate(HIGH_TOPIC):
        ef = f"hitopic-{i:02d}"
        a = c.add_item(t1, dom, "hitopic-a", ef, [], provenance=au)
        b = c.add_item(t2, dom, "hitopic-b", ef, [], provenance=au)
        c.pair(a, b, "topic-related", "entailment", notes="high-overlap topic-related; near-identical surface, logically independent claims")
        c.pair(b, a, "topic-related", "entailment")

    # mid-overlap CONTRADICTION (some shared vocabulary, atemporal incompatibility)
    MID_CONTRA = [
        ("The report concluded the policy saved money.", "An audit found the policy raised overall spending.", "corporate-events"),
        ("The witness said the car was blue.", "The dashcam footage shows a green car.", "everyday-statements"),
        ("The label states the jar contains no added sugar.", "Lab analysis found the jar has a high sugar content.", "biomed-claims"),
        ("The manual says the part is user-replaceable.", "The technician confirmed the part is soldered to the board.", "product-support"),
        ("The council claims the road is fully repaved.", "Residents report the road is still full of potholes.", "geo-civics"),
        ("The company said the factory runs on renewable power.", "Inspectors found the factory is connected only to a coal grid.", "corporate-events"),
        ("The brochure describes the hotel as beachfront.", "Guests report the hotel is two miles inland.", "geo-civics"),
        ("The study's abstract reports a large effect.", "The results table shows the effect was not significant.", "biomed-claims"),
        ("The vendor states the software is open source.", "The license file forbids redistribution of the code.", "product-support"),
        ("The minutes record that the motion passed.", "The vote tally shows the motion was defeated.", "corporate-events"),
        ("The spokesperson said no layoffs were planned.", "An internal memo lists three hundred roles to be cut.", "corporate-events"),
        ("The packaging claims the product is recyclable.", "The waste authority says the product cannot be recycled.", "product-support"),
        ("The coach insisted the striker was fit to play.", "The medical staff ruled the striker out with an injury.", "everyday-statements"),
        ("The prospectus describes the fund as low risk.", "The regulator classified the fund as high risk.", "corporate-events"),
        ("The sign says the trail is open year round.", "The ranger station confirms the trail closes each winter.", "geo-civics"),
        ("The press release says the drug is approved for children.", "The regulator's letter restricts the drug to adults.", "biomed-claims"),
        ("The invoice lists the work as completed.", "The site inspection found the work unfinished.", "corporate-events"),
        ("The listing advertises parking included.", "The building manager says there are no parking spaces.", "geo-civics"),
        ("The summary says emissions fell last year.", "The dataset shows emissions rose last year.", "geo-civics"),
        ("The vendor states the update is optional.", "The security bulletin says the update is mandatory.", "product-support"),
        ("The witness testified the room was empty.", "The security log records four entries that hour.", "everyday-statements"),
        ("The abstract claims the method is faster.", "The benchmark table shows the method is slower.", "product-support"),
    ]
    for i, (t1, t2, dom) in enumerate(MID_CONTRA):
        ef = f"midcontra-{i:02d}"
        a = c.add_item(t1, dom, "midcontra-a", ef, [], provenance=au)
        b = c.add_item(t2, dom, "midcontra-b", ef, [], provenance=au)
        c.pair(a, b, "contradiction", "topic-related", notes="mid-overlap contradiction; atemporal incompatibility")
        c.pair(b, a, "contradiction", "topic-related")


def gen_fillers(c: Corpus) -> list[str]:
    ids = []
    for slug, domain, text in S.FILLER:
        ids.append(c.add_item(text, domain, "filler-standalone", f"filler-{slug}", [], provenance=dict(kind="authored")))
    return ids


GENERATORS = [
    gen_geo_capitals, gen_geo_landlocked, gen_geo_direction, gen_geo_pop,
    gen_geo_civics, gen_geo_treaty, gen_corp_acq, gen_corp_head,
    gen_bio_multi, gen_bio_time, gen_prod, gen_everyday, gen_everyday_scenes,
    gen_compete, gen_overlap_fillers,
]


def generate() -> Corpus:
    c = Corpus()
    for g in GENERATORS:
        g(c)
    filler_ids = gen_fillers(c)

    # unrelated: pair every filler with a handful of deterministically-chosen
    # cross-domain items that share no entity and ~0 lexical overlap
    all_ids = sorted(c.items)
    for fid in filler_ids:
        fitem = c.items[fid]
        cands = [i for i in all_ids
                 if c.items[i]["domain"] != fitem["domain"]
                 and not (set(map(_norm_ent, c.items[i]["entities"])) & set(map(_norm_ent, fitem["entities"])))]
        # stable pseudo-random selection
        cands.sort(key=lambda i: hashlib.sha1((fid + i).encode()).hexdigest())
        picked = 0
        for cid in cands:
            if lexical.jaccard(fitem["text"], c.items[cid]["text"]) <= 0.1 and lexical.entity_overlap(fitem["entities"], c.items[cid]["entities"]) == 0:
                c.pair(fid, cid, "unrelated", None, notes="cross-domain, ~0 lexical + entity overlap")
                picked += 1
            if picked >= 8:
                break
    return c


if __name__ == "__main__":
    c = generate()
    from collections import Counter
    print("items", len(c.items), "pairs", len(c.pairs))
    print("by relation:", dict(sorted(Counter(p["relation"] for p in c.pairs).items())))
    print("by domain:", dict(sorted(Counter(i["domain"] for i in c.items.values()).items())))
    print("template families:", len({i["template_family"] for i in c.items.values()}))
    print("entity families:", len({i["entity_family"] for i in c.items.values()}))
