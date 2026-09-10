"""RELATE-DOC v0.1 generator — assemble structured documents, claim ledgers,
Family-A controlled compressions, and Family-B transformation pairs.

Family A (per document, spec §11 + operator brief):
  faithful               human-style summary keeping every primary claim + conclusion  (CONTROL)
  number-dropped          faithful minus one numeric claim
  number-changed          one numeric value altered, everything else kept
  relation-reversed       the document's key relation has its arguments swapped
  negation-inserted       the conclusion's main assertion is negated
  minority-entity-dropped the rare/minority entity mention removed
  temporal-value-shifted  one year / date changed
  conclusion-changed      topic + primary facts kept, the takeaway replaced
  + method compressions:  truncation / extractive / abstractive at {50,25,10}%

Family B (independent of documents): each `TRANSFORM_BASES` sentence x 9 typed
transformations, split by entity / template / domain / lexical realisation.
"""
from __future__ import annotations

import hashlib
import re

import sources as S

_WORD = re.compile(r"[A-Za-z0-9']+")


def _id(prefix, *parts):
    return prefix + hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]


def _wc(text):
    return len(_WORD.findall(text))


# --------------------------------------------------------------------------- #
# claim ledger helper
# --------------------------------------------------------------------------- #
def _claim(cid, text, salience, ctype, rare=False):
    return dict(id=cid, text=text.strip(), salience=salience, type=ctype, is_rare_entity=rare)


# --------------------------------------------------------------------------- #
# geo-civics documents
# --------------------------------------------------------------------------- #
def build_geo(sc):
    ef = f"ef-doc-{sc['slug']}"
    p, c = sc["place"], sc["country"]
    (y0, n0), (y1, n1), (y2, n2) = sc["pop"]
    bpct, barea = sc["budget_pct"]
    upct, uyr = sc["unemployment"]
    ra, rv, rb = sc["rel"]

    body = (
        f"{p} is a settlement on {sc['region']} in {c}, governed by {sc['seat']}. "
        f"Its economy centres on {sc['industry']}. "
        f"The population was about {n0} in {y0}, {n1} in {y1}, and {n2} in {y2}. "
        f"Roughly {bpct} of the council budget is spent on {barea}. "
        f"Unemployment stood at {upct} in {uyr}. "
        f"Geographically, {ra} {rv} {rb}. "
        + " ".join(sc["secondary"]) + " "
        f"Among its lesser-known areas, {sc['rare_district']} sit at the edge of the built-up zone "
        f"and are home to a few hundred residents. "
        f"Overall, {sc['conclusion']}."
    )

    claims = [
        _claim("c1", f"{p} is in {c}", "primary", "fact"),
        _claim("c2", f"{p}'s economy centres on {sc['industry']}", "primary", "fact"),
        _claim("c3", f"{p}'s population was {n0} in {y0}", "primary", "numeric"),
        _claim("c4", f"{p}'s population was {n2} in {y2}", "primary", "numeric"),
        _claim("c5", f"{bpct} of the council budget goes to {barea}", "secondary", "numeric"),
        _claim("c6", f"unemployment in {p} was {upct} in {uyr}", "secondary", "numeric"),
        _claim("c7", f"{ra} {rv} {rb}", "primary", "relational"),
        _claim("c8", sc["secondary"][0], "secondary", "temporal"),
        _claim("c9", f"{sc['rare_district']} are on the edge of the built-up zone", "rare", "fact", rare=True),
        _claim("c10", sc["conclusion"], "primary", "fact"),
    ]

    faithful = (
        f"{p}, in {c}, is built around {sc['industry']}. Its population grew from about {n0} in {y0} "
        f"to {n2} in {y2}. {ra} {rv} {rb}. The outlying {sc['rare_district']} house a few hundred people. "
        f"Overall, {sc['conclusion']}."
    )

    corruptions = dict(
        faithful=faithful,
        number_dropped=(
            f"{p}, in {c}, is built around {sc['industry']}. Its population grew over the two decades to {y2}. "
            f"{ra} {rv} {rb}. The outlying {sc['rare_district']} house a few hundred people. Overall, {sc['conclusion']}."),
        number_changed=(
            f"{p}, in {c}, is built around {sc['industry']}. Its population grew from about {n0} in {y0} "
            f"to {_bump_number(n2)} in {y2}. {ra} {rv} {rb}. The outlying {sc['rare_district']} house a few hundred people. "
            f"Overall, {sc['conclusion']}."),
        relation_reversed=(
            f"{p}, in {c}, is built around {sc['industry']}. Its population grew from about {n0} in {y0} "
            f"to {n2} in {y2}. {rb} {rv} {ra}. The outlying {sc['rare_district']} house a few hundred people. "
            f"Overall, {sc['conclusion']}."),
        negation_inserted=(
            f"{p}, in {c}, is built around {sc['industry']}. Its population grew from about {n0} in {y0} "
            f"to {n2} in {y2}. {ra} {rv} {rb}. The outlying {sc['rare_district']} house a few hundred people. "
            f"Overall, it is not the case that {sc['conclusion']}."),
        minority_entity_dropped=(
            f"{p}, in {c}, is built around {sc['industry']}. Its population grew from about {n0} in {y0} "
            f"to {n2} in {y2}. {ra} {rv} {rb}. Overall, {sc['conclusion']}."),
        temporal_value_shifted=(
            f"{p}, in {c}, is built around {sc['industry']}. Its population grew from about {n0} in {y0 - 6} "
            f"to {n2} in {y2}. {ra} {rv} {rb}. The outlying {sc['rare_district']} house a few hundred people. "
            f"Overall, {sc['conclusion']}."),
        conclusion_changed=(
            f"{p}, in {c}, is built around {sc['industry']}. Its population grew from about {n0} in {y0} "
            f"to {n2} in {y2}. {ra} {rv} {rb}. The outlying {sc['rare_district']} house a few hundred people. "
            f"Overall, the city has kept housing supply ahead of demand and rents have been broadly flat."),
    )
    queries = [
        dict(text=f"What was the population of {p} in {y2}?", answer_claim="c4", grade=3),
        dict(text=f"Where does {p}'s council spend most of its budget?", answer_claim="c5", grade=3),
        dict(text=f"How does {p} sit relative to {rb}?", answer_claim="c7", grade=3),
        dict(text=f"What is the overall outlook for {p}?", answer_claim="c10", grade=3),
    ]
    return _doc("geo-civics", ef, f"tf-place-profile", sc["slug"], body, claims, corruptions, queries,
                key_relation=dict(a=ra, verb=rv, b=rb, claim="c7"),
                rare_claim="c9", conclusion_claim="c10",
                numeric_claims=["c3", "c4", "c5", "c6"], temporal_claims=["c8"])


# --------------------------------------------------------------------------- #
# biomed documents
# --------------------------------------------------------------------------- #
def build_bio(sc):
    ef = f"ef-doc-{sc['slug']}"
    a = sc["agent"]
    (y0, e0), (y1, e1) = sc["timeline"]
    n, nunit = sc["n"]
    ra, rv, rb = sc["rel"]

    body = (
        f"A randomised trial evaluated {a} for {sc['condition']} in {sc['population']}. "
        f"{n} {nunit} were enrolled for {sc['duration'][0]}. "
        f"The primary result was {sc['primary_effect']}; a secondary finding was {sc['secondary_effect']}. "
        f"In terms of comparison, {ra} {rv} {rb}. "
        + " ".join(sc["secondary"]) + " "
        f"A pre-specified subgroup of interest was {sc['rare_subgroup']}, who were few in number but showed a similar direction of effect. "
        f"Over time: in {y0}, {e0}; in {y1}, {e1}. "
        f"The authors concluded that {sc['conclusion']}."
    )

    claims = [
        _claim("c1", f"the trial evaluated {a} for {sc['condition']}", "primary", "fact"),
        _claim("c2", f"{n} {nunit} were enrolled", "secondary", "numeric"),
        _claim("c3", f"the primary result was {sc['primary_effect']}", "primary", "numeric"),
        _claim("c4", f"a secondary finding was {sc['secondary_effect']}", "secondary", "fact"),
        _claim("c5", f"{ra} {rv} {rb}", "primary", "relational"),
        _claim("c6", f"in {y0}, {e0}", "primary", "temporal"),
        _claim("c7", f"in {y1}, {e1}", "primary", "temporal"),
        _claim("c8", sc["secondary"][0], "secondary", "fact"),
        _claim("c9", f"{sc['rare_subgroup']} showed a similar direction of effect", "rare", "fact", rare=True),
        _claim("c10", sc["conclusion"], "primary", "fact"),
    ]
    faithful = (
        f"A trial of {a} for {sc['condition']} in {sc['population']} found {sc['primary_effect']}. "
        f"{ra} {rv} {rb}. The {sc['rare_subgroup']} subgroup showed a similar effect. "
        f"By {y1}, {e1}. The authors concluded that {sc['conclusion']}."
    )
    corruptions = dict(
        faithful=faithful,
        number_dropped=(
            f"A trial of {a} for {sc['condition']} in {sc['population']} found a reduction in the primary outcome. "
            f"{ra} {rv} {rb}. The {sc['rare_subgroup']} subgroup showed a similar effect. "
            f"By {y1}, {e1}. The authors concluded that {sc['conclusion']}."),
        number_changed=(
            f"A trial of {a} for {sc['condition']} in {sc['population']} found {_bump_effect(sc['primary_effect'])}. "
            f"{ra} {rv} {rb}. The {sc['rare_subgroup']} subgroup showed a similar effect. "
            f"By {y1}, {e1}. The authors concluded that {sc['conclusion']}."),
        relation_reversed=(
            f"A trial of {a} for {sc['condition']} in {sc['population']} found {sc['primary_effect']}. "
            f"{rb} {rv} {ra}. The {sc['rare_subgroup']} subgroup showed a similar effect. "
            f"By {y1}, {e1}. The authors concluded that {sc['conclusion']}."),
        negation_inserted=(
            f"A trial of {a} for {sc['condition']} in {sc['population']} found {sc['primary_effect']}. "
            f"{ra} {rv} {rb}. The {sc['rare_subgroup']} subgroup showed a similar effect. "
            f"By {y1}, {e1}. The authors concluded that it is not the case that {sc['conclusion']}."),
        minority_entity_dropped=(
            f"A trial of {a} for {sc['condition']} in {sc['population']} found {sc['primary_effect']}. "
            f"{ra} {rv} {rb}. By {y1}, {e1}. The authors concluded that {sc['conclusion']}."),
        temporal_value_shifted=(
            f"A trial of {a} for {sc['condition']} in {sc['population']} found {sc['primary_effect']}. "
            f"{ra} {rv} {rb}. The {sc['rare_subgroup']} subgroup showed a similar effect. "
            f"By {y1 + 3}, {e1}. The authors concluded that {sc['conclusion']}."),
        conclusion_changed=(
            f"A trial of {a} for {sc['condition']} in {sc['population']} found {sc['primary_effect']}. "
            f"{ra} {rv} {rb}. The {sc['rare_subgroup']} subgroup showed a similar effect. "
            f"By {y1}, {e1}. The authors concluded that the effect is large, durable, and should change first-line practice."),
    )
    queries = [
        dict(text=f"What was the primary result for {a}?", answer_claim="c3", grade=3),
        dict(text=f"How did {a} compare with {rb}?", answer_claim="c5", grade=3),
        dict(text=f"What did the later re-analysis of {a} show?", answer_claim="c7", grade=3),
        dict(text=f"What did the authors conclude about {a}?", answer_claim="c10", grade=3),
    ]
    return _doc("biomed-claims", ef, "tf-study-abstract", sc["slug"], body, claims, corruptions, queries,
                key_relation=dict(a=ra, verb=rv, b=rb, claim="c5"),
                rare_claim="c9", conclusion_claim="c10",
                numeric_claims=["c2", "c3"], temporal_claims=["c6", "c7"])


# --------------------------------------------------------------------------- #
# corporate documents
# --------------------------------------------------------------------------- #
def build_corp(sc):
    ef = f"ef-doc-{sc['slug']}"
    acq, tgt = sc["acquirer"], sc["target"]
    val, vunit = sc["value"]
    prem = sc["premium"][0]
    (y0, e0), (y1, e1) = sc["timeline"]
    ra, rv, rb = sc["rel"]
    hc = sc["headcount"]

    body = (
        f"{acq} agreed to buy {tgt}, a company in {sc['sector']}, in {sc['year']} through {sc['structure']}. "
        f"The deal valued {tgt} at about {val} {vunit}, a premium of {prem} to its prior share price. "
        f"In corporate terms, {ra} {rv} {rb}. "
        + " ".join(sc["secondary"]) + " "
        f"{acq}'s own headcount went from {hc[0][1]} in {hc[0][0]} to {hc[-1][1]} in {hc[-1][0]}. "
        f"A smaller part of the target, {sc['rare_unit']}, was retained rather than sold on. "
        f"On timing: {e0}; later, {e1}. "
        f"The assessment at the time was that {sc['conclusion']}."
    )
    claims = [
        _claim("c1", f"{acq} agreed to buy {tgt} in {sc['year']}", "primary", "temporal"),
        _claim("c2", f"the deal valued {tgt} at about {val} {vunit}", "primary", "numeric"),
        _claim("c3", f"the premium was {prem}", "secondary", "numeric"),
        _claim("c4", f"{ra} {rv} {rb}", "primary", "relational"),
        _claim("c5", sc["secondary"][0], "secondary", "fact"),
        _claim("c6", f"{acq}'s headcount was {hc[0][1]} in {hc[0][0]}", "secondary", "numeric"),
        _claim("c7", f"{acq}'s headcount was {hc[-1][1]} in {hc[-1][0]}", "secondary", "numeric"),
        _claim("c8", f"{e0}", "primary", "temporal"),
        _claim("c9", f"{sc['rare_unit']} was retained rather than sold on", "rare", "fact", rare=True),
        _claim("c10", sc["conclusion"], "primary", "fact"),
    ]
    faithful = (
        f"{acq} bought {tgt} ({sc['sector']}) in {sc['year']} via {sc['structure']}, valuing it at about {val} {vunit}. "
        f"{ra} {rv} {rb}. {sc['rare_unit']} was kept. Later, {e1}. "
        f"The assessment was that {sc['conclusion']}."
    )
    corruptions = dict(
        faithful=faithful,
        number_dropped=(
            f"{acq} bought {tgt} ({sc['sector']}) in {sc['year']} via {sc['structure']}. "
            f"{ra} {rv} {rb}. {sc['rare_unit']} was kept. Later, {e1}. The assessment was that {sc['conclusion']}."),
        number_changed=(
            f"{acq} bought {tgt} ({sc['sector']}) in {sc['year']} via {sc['structure']}, valuing it at about {_bump_number(val)} {vunit}. "
            f"{ra} {rv} {rb}. {sc['rare_unit']} was kept. Later, {e1}. The assessment was that {sc['conclusion']}."),
        relation_reversed=(
            f"{tgt} bought {acq} ({sc['sector']}) in {sc['year']} via {sc['structure']}, valuing it at about {val} {vunit}. "
            f"{rb} {rv} {ra}. {sc['rare_unit']} was kept. Later, {e1}. The assessment was that {sc['conclusion']}."),
        negation_inserted=(
            f"{acq} bought {tgt} ({sc['sector']}) in {sc['year']} via {sc['structure']}, valuing it at about {val} {vunit}. "
            f"{ra} {rv} {rb}. {sc['rare_unit']} was kept. Later, {e1}. The assessment was that it is not the case that {sc['conclusion']}."),
        minority_entity_dropped=(
            f"{acq} bought {tgt} ({sc['sector']}) in {sc['year']} via {sc['structure']}, valuing it at about {val} {vunit}. "
            f"{ra} {rv} {rb}. Later, {e1}. The assessment was that {sc['conclusion']}."),
        temporal_value_shifted=(
            f"{acq} bought {tgt} ({sc['sector']}) in {sc['year'] - 4} via {sc['structure']}, valuing it at about {val} {vunit}. "
            f"{ra} {rv} {rb}. {sc['rare_unit']} was kept. Later, {e1}. The assessment was that {sc['conclusion']}."),
        conclusion_changed=(
            f"{acq} bought {tgt} ({sc['sector']}) in {sc['year']} via {sc['structure']}, valuing it at about {val} {vunit}. "
            f"{ra} {rv} {rb}. {sc['rare_unit']} was kept. Later, {e1}. "
            f"The assessment was that the price was low, the timing excellent, and the integration a model of its kind."),
    )
    queries = [
        dict(text=f"How much did {acq} pay for {tgt}?", answer_claim="c2", grade=3),
        dict(text=f"Who acquired whom in the {tgt} deal?", answer_claim="c4", grade=3),
        dict(text=f"What happened to {acq} after the {tgt} deal?", answer_claim="c7", grade=3),
        dict(text=f"What was the verdict on the {tgt} acquisition?", answer_claim="c10", grade=3),
    ]
    return _doc("corporate-events", ef, "tf-deal-writeup", sc["slug"], body, claims, corruptions, queries,
                key_relation=dict(a=ra, verb=rv, b=rb, claim="c4"),
                rare_claim="c9", conclusion_claim="c10",
                numeric_claims=["c2", "c3", "c6", "c7"], temporal_claims=["c1", "c8"])


# --------------------------------------------------------------------------- #
# short documents
# --------------------------------------------------------------------------- #
def build_geo_short(slug, subj, when, size, effect, conclusion):
    ef = f"ef-doc-{slug}"
    body = (f"{subj} {when}. It has {size} and {effect}. Overall, {conclusion}.")
    claims = [
        _claim("c1", f"{subj} {when}", "primary", "temporal"),
        _claim("c2", f"{subj} has {size}", "primary", "numeric"),
        _claim("c3", f"{subj} {effect}", "primary", "fact"),
        _claim("c4", conclusion, "primary", "fact"),
    ]
    corr = dict(
        faithful=f"{subj} {when}; it has {size} and {effect}. Overall, {conclusion}.",
        number_dropped=f"{subj} {when} and {effect}. Overall, {conclusion}.",
        number_changed=f"{subj} {when}; it has {_bump_number(size)} and {effect}. Overall, {conclusion}.",
        negation_inserted=f"{subj} {when}; it has {size} and {effect}. Overall, it is not the case that {conclusion}.",
        temporal_value_shifted=f"{subj} {_shift_year(when)}; it has {size} and {effect}. Overall, {conclusion}.",
        conclusion_changed=f"{subj} {when}; it has {size} and {effect}. Overall, the project has made no measurable difference.",
    )
    queries = [dict(text=f"What is the size or capacity described for {subj}?", answer_claim="c2", grade=3),
               dict(text=f"What was the overall outcome for {subj}?", answer_claim="c4", grade=3)]
    return _doc("geo-civics", ef, "tf-infrastructure-note", slug, body, claims, corr, queries,
                key_relation=None, rare_claim=None, conclusion_claim="c4",
                numeric_claims=["c2"], temporal_claims=["c1"])


def build_bio_short(slug, agent, condition, e1, e2, conclusion):
    ef = f"ef-doc-{slug}"
    body = (f"A trial tested {agent} for {condition}. It {e1} and {e2}. The authors concluded that {conclusion}.")
    claims = [
        _claim("c1", f"the trial tested {agent} for {condition}", "primary", "fact"),
        _claim("c2", f"{agent} {e1}", "primary", "fact"),
        _claim("c3", f"{agent} {e2}", "secondary", "numeric"),
        _claim("c4", conclusion, "primary", "fact"),
    ]
    corr = dict(
        faithful=f"A trial of {agent} for {condition} found it {e1} and {e2}. The authors concluded that {conclusion}.",
        number_dropped=f"A trial of {agent} for {condition} found it {e1}. The authors concluded that {conclusion}.",
        number_changed=f"A trial of {agent} for {condition} found it {e1} and {_bump_effect(e2)}. The authors concluded that {conclusion}.",
        negation_inserted=f"A trial of {agent} for {condition} found it {e1} and {e2}. The authors concluded that it is not the case that {conclusion}.",
        conclusion_changed=f"A trial of {agent} for {condition} found it {e1} and {e2}. The authors concluded that the treatment is ineffective and possibly harmful.",
    )
    queries = [dict(text=f"What did the {agent} trial find?", answer_claim="c2", grade=3),
               dict(text=f"What did the authors conclude about {agent}?", answer_claim="c4", grade=3)]
    return _doc("biomed-claims", ef, "tf-trial-note", slug, body, claims, corr, queries,
                key_relation=None, rare_claim=None, conclusion_claim="c4",
                numeric_claims=["c3"], temporal_claims=[])


def build_corp_short(slug, acq, tgt, year, structure, sector, effect, conclusion):
    ef = f"ef-doc-{slug}"
    body = (f"{acq} acquired {tgt} in {year} through {structure}, a deal in {sector}. {effect.capitalize()}. "
            f"The assessment was that {conclusion}.")
    claims = [
        _claim("c1", f"{acq} acquired {tgt} in {year}", "primary", "temporal"),
        _claim("c2", f"the deal was in {sector}", "secondary", "fact"),
        _claim("c3", effect, "primary", "fact"),
        _claim("c4", conclusion, "primary", "fact"),
    ]
    corr = dict(
        faithful=f"{acq} acquired {tgt} in {year} ({sector}) via {structure}. {effect.capitalize()}. The assessment was that {conclusion}.",
        relation_reversed=f"{tgt} acquired {acq} in {year} ({sector}) via {structure}. {effect.capitalize()}. The assessment was that {conclusion}.",
        negation_inserted=f"{acq} acquired {tgt} in {year} ({sector}) via {structure}. {effect.capitalize()}. The assessment was that it is not the case that {conclusion}.",
        temporal_value_shifted=f"{acq} acquired {tgt} in {year - 4} ({sector}) via {structure}. {effect.capitalize()}. The assessment was that {conclusion}.",
        conclusion_changed=f"{acq} acquired {tgt} in {year} ({sector}) via {structure}. {effect.capitalize()}. The assessment was that the deal was a clear and immediate success.",
    )
    queries = [dict(text=f"Who acquired {tgt}?", answer_claim="c1", grade=3),
               dict(text=f"What was the verdict on the {tgt} deal?", answer_claim="c4", grade=3)]
    return _doc("corporate-events", ef, "tf-deal-note", slug, body, claims, corr, queries,
                key_relation=dict(a=acq, verb="acquired", b=tgt, claim="c1"),
                rare_claim=None, conclusion_claim="c4", numeric_claims=[], temporal_claims=["c1"])


# --------------------------------------------------------------------------- #
# corruption helpers
# --------------------------------------------------------------------------- #
def _bump_number(s):
    """Change the first integer-ish token by ~20% (keeps the surface form)."""
    m = re.search(r"[\d,.]+", s)
    if not m:
        return s
    raw = m.group(0)
    try:
        val = float(raw.replace(",", ""))
    except ValueError:
        return s
    new = val * 0.78
    if "," in raw or val >= 1000:
        rep = f"{int(round(new)):,}"
    elif "." in raw:
        rep = f"{new:.1f}"
    else:
        rep = str(int(round(new)))
    return s[:m.start()] + rep + s[m.end():]


def _bump_effect(s):
    return _bump_number(s) if re.search(r"\d", s) else s.replace("a ", "a smaller ", 1)


def _shift_year(s):
    return re.sub(r"\b(19|20)\d\d\b", lambda m: str(int(m.group(0)) - 5), s, count=1)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def _doc(domain, ef, tf, slug, body, claims, corruptions, queries, *, key_relation,
         rare_claim, conclusion_claim, numeric_claims, temporal_claims):
    body = " ".join(body.split())
    did = _id("d-", domain, slug)
    # method compressions from the faithful summary
    faithful = corruptions["faithful"]
    method = {}
    words = _WORD.findall(body)
    for ratio in (50, 25, 10):
        keep = max(8, int(len(words) * ratio / 100))
        method[f"truncation_{ratio}"] = " ".join(body.split()[:keep])
    sents = re.split(r"(?<=[.!?])\s+", body)
    method["extractive_50"] = " ".join(sents[:max(1, len(sents) // 2)])
    method["extractive_25"] = " ".join(sents[:max(1, len(sents) // 4)])
    method["abstractive_25"] = faithful
    method["abstractive_10"] = sents[0] + " " + (sents[-1] if len(sents) > 1 else "")

    comp_recs = []
    for kind, text in {**corruptions, **method}.items():
        text = " ".join(text.split())
        comp_recs.append(dict(
            id=_id("cmp-", did, kind),
            document_id=did, kind=kind,
            family="A-faithfulness" if kind in corruptions else "A-method",
            is_control=(kind == "faithful"),
            text=text,
            word_count=_wc(text),
            corruption_targets=_corruption_target(kind, key_relation, rare_claim, conclusion_claim, numeric_claims, temporal_claims),
        ))
    return dict(
        document=dict(
            id=did, domain=domain, entity_family=ef, template_family=tf, slug=slug,
            text=body, word_count=_wc(body),
            atomic_claims=claims,
            primary_claims=[c["id"] for c in claims if c["salience"] == "primary"],
            secondary_claims=[c["id"] for c in claims if c["salience"] == "secondary"],
            rare_claims=[c["id"] for c in claims if c["salience"] == "rare"],
            numeric_claims=numeric_claims, temporal_claims=temporal_claims,
            key_relation=key_relation, conclusion_claim=conclusion_claim,
            queries=[dict(id=_id("dq-", did, q["text"]), **q) for q in queries],
        ),
        compressions=comp_recs,
    )


def _corruption_target(kind, key_relation, rare_claim, conclusion_claim, numeric_claims, temporal_claims):
    return {
        "faithful": [],
        "number_dropped": numeric_claims[:1],
        "number_changed": numeric_claims[:1],
        "relation_reversed": [key_relation["claim"]] if key_relation else [],
        "negation_inserted": [conclusion_claim] if conclusion_claim else [],
        "minority_entity_dropped": [rare_claim] if rare_claim else [],
        "temporal_value_shifted": temporal_claims[:1],
        "conclusion_changed": [conclusion_claim] if conclusion_claim else [],
    }.get(kind, [])


# --------------------------------------------------------------------------- #
# Family B — transformation pairs  (content varies, transformation type fixed)
# --------------------------------------------------------------------------- #
_LEX = {  # lexical-realisation variants, used only as a split axis label
    "approves": ["approves", "signs off", "waves through"],
    "rejects": ["rejects", "turns down", "throws out"],
    "acquires": ["acquires", "buys", "takes over"],
    "reports": ["reports", "records", "notes"],
    "cuts": ["cuts", "sheds", "eliminates"],
}


def _base(vp):
    """present-3rd-singular -> bare infinitive."""
    irr = {"cuts": "cut", "shuts": "shut", "misses": "miss", "processes": "process",
           "switches": "switch", "closes": "close", "raises": "raise", "sells": "sell",
           "flags": "flag", "fines": "fine", "funds": "fund", "records": "record",
           "shows": "show", "upholds": "uphold", "flags ": "flag"}
    if vp in irr:
        return irr[vp]
    if vp.endswith("hes") or vp.endswith("ses") or vp.endswith("xes"):
        return vp[:-2]
    return vp[:-1] if vp.endswith("s") else vp


def generate_transform_pairs():
    pairs = []
    for row in S.TRANSFORM_BASES:
        tid, subj, vpres, vpast, obj, tail, domain, has_time, has_rel, claimy = row
        ef = f"ef-tr-{tid}"
        tf = f"tf-tr-{_base(vpres).replace(' ', '-')}"
        Subj = subj[0].upper() + subj[1:]
        Obj = obj[0].upper() + obj[1:]
        base = f"{vpast}"
        canonical = f"{Subj} {vpast} {obj} {tail}."
        present = f"{Subj} {vpres} {obj} {tail}."
        vinf = _base(vpres)

        transforms = {
            "verbose_to_concise": (
                f"It should be noted that, per the account on record, {subj} {vpast} {obj} {tail}, a step that had been expected for some time.",
                f"{Subj} {vpast} {obj}."),
            "formal_to_informal": (
                f"{Subj} formally {vpast} {obj} {tail}.",
                f"so {subj} went ahead and {vpast} {obj}."),
            "active_to_passive": (
                canonical,
                f"{Obj} was {vpast} by {subj} {tail}."),
            "present_to_past": (present, canonical),
            "statement_to_negation": (
                canonical,
                f"{Subj} did not {vinf} {obj} {tail}."),
            "relation_swap": None,   # generated from SWAP_BASES below, where both slots are entities
            "temporal_shift": (
                f"{Subj} {vpast} {obj} {tail}, in 2023.",
                f"{Subj} {vpast} {obj} {tail}, in 2016.") if has_time else None,
            "claim_strengthened": (
                canonical,
                f"{Subj} decisively {vpast} {obj} {tail}, a clear and substantial move.") if claimy else None,
            "claim_weakened": (
                canonical,
                f"{Subj} may have {vpast} {obj} {tail}, though the evidence is thin.") if claimy else None,
        }
        for ttype, pair in transforms.items():
            if pair is None:
                continue
            src, tgt = " ".join(pair[0].split()), " ".join(pair[1].split())
            if src == tgt:
                continue
            pairs.append(dict(
                id=_id("trp-", tid, ttype), base_id=tid, transformation=ttype,
                domain=domain, entity_family=ef, template_family=tf,
                lexical_realisation=_LEX.get(vpres, [vpres])[0],
                is_reversible=ttype in ("active_to_passive", "present_to_past"),
                source=src, target=tgt,
            ))
    # relation_swap + reversible active/passive from the two-entity bases
    for sid, a, verb, b, tail, domain in S.SWAP_BASES:
        ef, tf = f"ef-tr-{sid}", f"tf-tr-swap"
        rows = {
            "relation_swap": (f"{a} {verb} {b} {tail}.", f"{b} {verb} {a} {tail}."),
            "active_to_passive": (f"{a} {verb} {b} {tail}.", f"{b} was {verb} by {a} {tail}."),
            "statement_to_negation": (f"{a} {verb} {b} {tail}.", f"{a} did not {_deverb(verb)} {b} {tail}."),
        }
        for ttype, (src, tgt) in rows.items():
            src, tgt = " ".join(src.split()), " ".join(tgt.split())
            if src == tgt:
                continue
            pairs.append(dict(
                id=_id("trp-", sid, ttype), base_id=sid, transformation=ttype,
                domain=domain, entity_family=ef, template_family=tf,
                lexical_realisation=verb,
                is_reversible=ttype in ("relation_swap", "active_to_passive"),
                source=src, target=tgt,
            ))
    return pairs


def _deverb(verb_past):
    return {"acquired": "acquire", "outbid": "outbid", "beat": "beat", "defeated": "defeat",
            "lies north of": "lie north of", "lies upstream of": "lie upstream of",
            "sued": "sue"}.get(verb_past, verb_past.rstrip("d") if verb_past.endswith("ed") else verb_past)


# --------------------------------------------------------------------------- #
def generate():
    docs, comps = [], []
    for sc in S.GEO:
        r = build_geo(sc); docs.append(r["document"]); comps += r["compressions"]
    for sc in S.BIO:
        r = build_bio(sc); docs.append(r["document"]); comps += r["compressions"]
    for sc in S.CORP:
        r = build_corp(sc); docs.append(r["document"]); comps += r["compressions"]
    for t in S.GEO_SHORT:
        r = build_geo_short(*t); docs.append(r["document"]); comps += r["compressions"]
    for t in S.BIO_SHORT:
        r = build_bio_short(*t); docs.append(r["document"]); comps += r["compressions"]
    for t in S.CORP_SHORT:
        r = build_corp_short(*t); docs.append(r["document"]); comps += r["compressions"]
    tpairs = generate_transform_pairs()
    return dict(documents=docs, compressions=comps, transformation_pairs=tpairs)


if __name__ == "__main__":
    g = generate()
    from collections import Counter
    print("documents", len(g["documents"]), "avg words", round(sum(d["word_count"] for d in g["documents"]) / len(g["documents"])))
    print("compressions", len(g["compressions"]), dict(Counter(c["kind"] for c in g["compressions"])))
    print("transformation pairs", len(g["transformation_pairs"]), dict(Counter(t["transformation"] for t in g["transformation_pairs"])))
    print("domains", dict(Counter(d["domain"] for d in g["documents"])))
