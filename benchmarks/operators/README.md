# Operator bakeoff benchmark

RELATE tests whether a semantic edit admits a simple, reusable vector
operator in an embedding space -- and explicitly concludes when none
does. `NONE_PASS` is a result, not a failure.

## Selection (held-out native targets, 0.85 fidelity bar)

| transformation | native cosine | simplest passing operator |
|---|---|---|
| active_to_passive | 0.995 | identity_map |
| present_to_past | 0.995 | identity_map |
| relation_swap | 0.993 | identity_map |
| claim_strengthened | 0.995 | identity_map |
| temporal_shift | 0.994 | identity_map |
| claim_weakened | 0.831 | constant_delta |
| formal_to_informal | 0.682 | NONE_PASS |
| verbose_to_concise | 0.784 | NONE_PASS |
| statement_to_negation | 0.661 | NONE_PASS |

`identity_map` means operator identity -- no vector map sufficed under
this space and bar -- never semantic identity of the texts. The ladder
(identity 0, delta 1, linear 2, affine 3) is a complexity ordering, not
a quality ordering: linear/affine train to ~1.0 on the small NONE-class
sets yet fail held-out, which is reported as fit diagnostic, never as
evidence. A negation at 0.66 cosine with no passing operator shows
high similarity cannot authorize fine semantic preservation.

## Provenance discipline

Real frozen RELATE-DOC pairs supply case identities, the nine exact
transformation classes, per-class counts, and sha256 content hashes;
train/eval case sets carry distinct hashes (asserted unequal in
`run.py`). Vectors come from a documented deterministic mirror
encoder -- the geometry is synthetic, the case structure is not. The
Wave-5 numbers live labeled EXTERNAL in `historic-reference.json`.

## Running

```text
python benchmarks/operators/run.py          # write expected/
python benchmarks/operators/run.py --check  # verify byte-identical
```

Every held-out candidate flows through `evaluate_transformation`
against `TARGET_NATIVE` authority (the transformed content embedded
normally); selection reads scoped verdicts only. No text is rewritten
and no embedding model runs anywhere in this path.
