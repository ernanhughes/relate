# RELATE-DOC v0.1 — Datasheet

Long-document extension of RELATE, for the compression / faithfulness chapter
(Ch 22) and the operator / transformation chapter (Ch 23). `corpus_hash`
(SHA-256 of the sorted document + compression + transformation records):
`111cc2bb9008557d632061f98a0e4f847b91293e31bcdb2d78285a35de7f7914`.

## Motivation

RELATE v0.1 is a *pair-level* probe. RELATE-DOC adds **structured source
documents** so the book can ask two document-level questions:

1. **Family A — faithfulness / compression (Ch 22).** For each document, a
   `faithful` control summary plus seven controlled corruptions, each targeting a
   *named atomic claim*: `number_dropped`, `number_changed`, `relation_reversed`,
   `negation_inserted`, `minority_entity_dropped`, `temporal_value_shifted`,
   `conclusion_changed`. Plus method compressions (`truncation` / `extractive` /
   `abstractive` at 50/25/10%). The Wave 4 question: **which corruption classes
   are invisible to whole-document embedding drift, neighbourhood preservation,
   query-conditioned preservation, and claim-conditioned preservation** — the
   blind-spot matrix. The adversarial cases (`relation_reversed`,
   `negation_inserted`, `number_changed`) deliberately keep the topic fixed so
   the coarse geometry *should* survive while one load-bearing fact is reversed.

2. **Family B — transformation (Ch 23).** 35 base sentences × 9 typed
   transformations (`verbose_to_concise`, `formal_to_informal`,
   `active_to_passive`, `present_to_past`, `statement_to_negation`,
   `relation_swap`, `temporal_shift`, `claim_strengthened`, `claim_weakened`),
   with content varied and the transformation type held fixed, split by entity /
   template / domain / lexical realisation so the operator bake-off cannot
   memorise a surface pattern.

## Composition

| Quantity | Count |
|---|---|
| Source documents | 45 |
| — by domain | {"geo-civics": 15, "biomed-claims": 15, "corporate-events": 15} |
| — word count (min / mean / max) | 34 / 78 / 173 |
| Atomic claims (total) | 270 |
| — rare / minority-entity claims | 15 |
| Compressions (total) | 595 |
| — Family A faithfulness corruptions | 280 |
| — Family A method compressions | 315 |
| Transformation pairs | 260 |
| — reversible (for inverse-consistency) | 90 |
| Entity families / template families | 45 / 6 |

Compressions by kind: {"faithful": 45, "number_dropped": 35, "number_changed": 35, "relation_reversed": 25, "negation_inserted": 45, "minority_entity_dropped": 15, "temporal_value_shifted": 35, "conclusion_changed": 45, "truncation_50": 45, "truncation_25": 45, "truncation_10": 45, "extractive_50": 45, "extractive_25": 45, "abstractive_25": 45, "abstractive_10": 45}

Transformation pairs by type: {"verbose_to_concise": 35, "formal_to_informal": 35, "active_to_passive": 45, "present_to_past": 35, "statement_to_negation": 45, "claim_strengthened": 15, "claim_weakened": 15, "temporal_shift": 25, "relation_swap": 10}

## Collection process

Every document is assembled by `generate.py` from the structured scenarios in
`sources.py` (invented, or public fact restated in original wording). **No text
is drawn from an existing summarisation / faithfulness corpus** (CNN/DM, XSum,
FActScore, SummEval, …) — the templated pipeline is the contamination defence.
Each corruption is applied by a documented rule to a *named* target claim, so the
gold label ("which claim was corrupted, and how") is exact.

## Splits

`splits/{entity,template,domain}.json` for documents (60/20/80 by deterministic
hash), `splits/lexical.json` for transformation pairs (by verb realisation and by
base sentence). Operator fits in the Transformation Wave train on
`entity:train`/`template:train`/`domain:train` and report held-out transfer on
each.

## Known limitations

1. **Short documents.** v0.1 documents average 78 words
   (spec §11 target 300–3,000). They are long enough to carry 4–10 atomic claims
   and a conclusion, which is what the blind-spot matrix needs, but a corruption
   has less room to hide than in a full-length article. A v0.2 would lengthen them.
2. **Synthetic.** Templated assembly gives exact corruption labels and zero
   public-corpus contamination, at the cost of surface regularity (the
   `template` split is the guard).
3. **No human faithfulness annotation** — the corruption target is known by
   construction; there is no human "is this summary faithful?" rating. Consistent
   with RELATE v0.1's labelling model.
4. **English only; Western-centric invented entities.**
5. **`minority_entity_dropped`** exists only for the 15
   full scenarios that carry a rare entity; the short documents do not.

## License

Target CC BY 4.0. Content is expression-original.
