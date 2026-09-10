"""Validator tests — prove the validators reject bad records BEFORE any generator runs.

    python -m pytest tests/            # if pytest is available
    python tests/test_validators.py    # stdlib fallback runner
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from schema import Item, Pair, Positive, Provenance, Query, SchemaError  # noqa: E402
from validate import ValidationReport, check_pair_semantics  # noqa: E402


def _item(**kw):
    base = dict(
        id="i-test01", text="A test sentence.", domain="geo-civics",
        template_family="tf-test", entity_family="ef-test", entities=[],
        provenance=Provenance(kind="authored"),
    )
    base.update(kw)
    return Item(**base)


def _pair(**kw):
    base = dict(
        id="p-test01", a_id="i-aaa001", b_id="i-bbb001", relation="paraphrase",
        rule_fired="paraphrase", runner_up="equivalent", lexical_overlap=0.5,
        entity_overlap=0, direction=None,
    )
    base.update(kw)
    return Pair(**base)


CASES: list[tuple[str, callable]] = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


# --------------------------------------------------------------------------- #
# structural
# --------------------------------------------------------------------------- #
@case("item id must match pattern")
def _():
    try:
        _item(id="bad").check()
        assert False, "expected SchemaError"
    except SchemaError:
        pass


@case("perturbed provenance needs parent_id + transformation")
def _():
    try:
        _item(provenance=Provenance(kind="perturbed")).check()
        assert False
    except SchemaError:
        pass


@case("pair rule_fired must equal relation")
def _():
    try:
        _pair(rule_fired="equivalent", relation="paraphrase").check()
        assert False
    except SchemaError:
        pass


@case("pair runner_up may not equal relation")
def _():
    try:
        _pair(runner_up="paraphrase").check()
        assert False
    except SchemaError:
        pass


@case("directional relation stored without direction is a schema error at semantic layer")
def _():
    rep = ValidationReport()
    items = {"i-aaa001": _item(id="i-aaa001"), "i-bbb001": _item(id="i-bbb001")}
    p = _pair(relation="entailment", rule_fired="entailment", direction=None)
    check_pair_semantics(p, items, rep)
    assert any("directional" in e for e in rep.errors), rep.errors


# --------------------------------------------------------------------------- #
# semantic — relation constraints
# --------------------------------------------------------------------------- #
@case("negation below 0.7 lexical overlap is rejected")
def _():
    rep = ValidationReport()
    items = {"i-aaa001": _item(id="i-aaa001"), "i-bbb001": _item(id="i-bbb001")}
    p = _pair(relation="negation", rule_fired="negation", runner_up="contradiction",
              direction="a->b", lexical_overlap=0.5)
    check_pair_semantics(p, items, rep)
    assert any("lexical_overlap >= 0.7" in e for e in rep.errors), rep.errors


@case("negation at 0.8 lexical overlap passes")
def _():
    rep = ValidationReport()
    items = {"i-aaa001": _item(id="i-aaa001", entities=["X"]), "i-bbb001": _item(id="i-bbb001", entities=["X"])}
    p = _pair(relation="negation", rule_fired="negation", runner_up="contradiction",
              direction="a->b", lexical_overlap=0.8, entity_overlap=1)
    check_pair_semantics(p, items, rep)
    assert not rep.errors, rep.errors


@case("unrelated with entity overlap is rejected")
def _():
    rep = ValidationReport()
    items = {"i-aaa001": _item(id="i-aaa001", entities=["X"]), "i-bbb001": _item(id="i-bbb001", entities=["X"])}
    p = _pair(relation="unrelated", rule_fired="unrelated", runner_up=None,
              lexical_overlap=0.05, entity_overlap=1)
    check_pair_semantics(p, items, rep)
    assert any("entity_overlap <= 0" in e for e in rep.errors), rep.errors


@case("temporal-mismatch without time_index on both items is rejected")
def _():
    rep = ValidationReport()
    items = {
        "i-aaa001": _item(id="i-aaa001", time_index="2004", entities=["X"]),
        "i-bbb001": _item(id="i-bbb001", time_index=None, entities=["X"]),
    }
    p = _pair(relation="temporal-mismatch", rule_fired="temporal-mismatch", runner_up="contradiction",
              direction="a->b", lexical_overlap=0.5, entity_overlap=1)
    check_pair_semantics(p, items, rep)
    assert any("time_index on BOTH" in e for e in rep.errors), rep.errors


@case("temporal-mismatch with identical time_index is rejected")
def _():
    rep = ValidationReport()
    items = {
        "i-aaa001": _item(id="i-aaa001", time_index="2004", entities=["X"]),
        "i-bbb001": _item(id="i-bbb001", time_index="2004", entities=["X"]),
    }
    p = _pair(relation="temporal-mismatch", rule_fired="temporal-mismatch", runner_up="contradiction",
              direction="a->b", lexical_overlap=0.5, entity_overlap=1)
    check_pair_semantics(p, items, rep)
    assert any("identical time_index" in e for e in rep.errors), rep.errors


@case("partial-support requires >=2 atomic_claims on a")
def _():
    rep = ValidationReport()
    items = {
        "i-aaa001": _item(id="i-aaa001", atomic_claims=["one claim only"]),
        "i-bbb001": _item(id="i-bbb001"),
    }
    p = _pair(relation="partial-support", rule_fired="partial-support", runner_up="entailment",
              direction="a->b", lexical_overlap=0.4)
    check_pair_semantics(p, items, rep)
    assert any("atomic_claims" in e for e in rep.errors), rep.errors


@case("relation-swap requires shared entities")
def _():
    rep = ValidationReport()
    items = {"i-aaa001": _item(id="i-aaa001"), "i-bbb001": _item(id="i-bbb001")}
    p = _pair(relation="relation-swap", rule_fired="relation-swap", runner_up="contradiction",
              direction="a->b", lexical_overlap=0.9, entity_overlap=0)
    check_pair_semantics(p, items, rep)
    assert rep.errors, "expected relation-swap with 0 entity overlap to be rejected"


@case("equivalent with byte-identical text is rejected as a duplicate")
def _():
    rep = ValidationReport()
    items = {
        "i-aaa001": _item(id="i-aaa001", text="Dublin is the capital of Ireland."),
        "i-bbb001": _item(id="i-bbb001", text="Dublin is the capital of Ireland."),
    }
    p = _pair(relation="equivalent", rule_fired="equivalent", runner_up="paraphrase",
              lexical_overlap=1.0)
    check_pair_semantics(p, items, rep)
    assert any("byte-identical" in e for e in rep.errors), rep.errors


@case("query needs a grade-3 positive")
def _():
    q = Query(id="q-test01", text="q?", query_style="question", target_item_id="i-aaa001",
              positives=[Positive(item_id="i-aaa001", grade=2)])
    try:
        q.check()
        assert False
    except SchemaError:
        pass


@case("query with a grade-3 positive passes")
def _():
    q = Query(id="q-test01", text="q?", query_style="question", target_item_id="i-aaa001",
              positives=[Positive(item_id="i-aaa001", grade=3)])
    q.check()


# --------------------------------------------------------------------------- #
@case("the DEV build validates clean")
def _():
    from validate import validate_dir
    rep, _cov = validate_dir(HERE / "dev" / "corpus", do_report=True)
    assert rep.ok, "DEV build has validation errors:\n" + "\n".join(rep.errors)


def run() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
