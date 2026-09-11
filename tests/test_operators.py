"""Operator tests: hypotheses represented, verdicts selected, text untouched."""

from pathlib import Path

import numpy as np
import pytest

from relate import RelateError
from relate.transformations import (
    OPERATOR_COMPLEXITY,
    ContentTransformation,
    ContentTransformationCase,
    IdentityMap,
    OperatorSelectionOutcome,
    VectorTransformation,
    fit_affine_operator,
    fit_constant_delta,
    fit_linear_operator,
    hash_case_set,
    hash_content,
    identity_map,
    select_simplest_passing,
)


def _pairs(seed=0, n=24, d=16, shift=None):
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(n, d))
    if shift is None:
        target = source + 0.02 * rng.normal(size=(n, d))
    else:
        target = source + shift
    return source, target


def _cases():
    return [
        ContentTransformationCase(
            case_id=f"c-{i}",
            source_content_hash=hash_content(f"source {i}"),
            target_content_hash=hash_content(f"target {i}"),
            relation="tense_edit",
        )
        for i in range(6)
    ]


def test_content_cases_bind_hashes():
    cases = _cases()
    assert hash_case_set(cases) == hash_case_set(list(reversed(cases)))
    assert len(hash_case_set(cases)) == 16
    assert hash_case_set(cases[:3]) != hash_case_set(cases[3:])
    assert hash_content("x") == hash_content("x")
    with pytest.raises(RelateError, match="non-empty"):
        ContentTransformationCase(case_id="", source_content_hash="a",
                                  target_content_hash="b", relation="r")


def test_producers_satisfy_vector_contract():
    source, target = _pairs()
    producers = [
        identity_map(relation="tense_edit", source_space_hash="s"),
        fit_constant_delta(source, target, relation="tense_edit",
                           source_space_hash="s", train_case_set_hash="h"),
        fit_linear_operator(source, target, relation="tense_edit",
                            source_space_hash="s", train_case_set_hash="h"),
        fit_affine_operator(source, target, relation="tense_edit",
                            source_space_hash="s", train_case_set_hash="h"),
    ]
    for producer in producers:
        assert isinstance(producer, VectorTransformation)
        assert not isinstance(producer, ContentTransformation)
        out = producer.transform(source)
        assert out.shape == source.shape
        assert np.isfinite(out).all()
        assert producer.source_space_hash == "s"
        assert len(producer.transformation_id) == 16
    assert OPERATOR_COMPLEXITY == {
        "identity_map": 0, "constant_delta": 1, "linear": 2, "affine": 3}


def test_delta_recovers_consistent_direction():
    shift = np.zeros(16)
    shift[0] = 2.0
    source, target = _pairs(shift=shift)
    delta = fit_constant_delta(source, target, relation="weakened",
                               source_space_hash="s")
    np.testing.assert_allclose(delta.delta, shift, atol=0.05)
    np.testing.assert_allclose(delta.transform(source), target, atol=0.05)


def test_identity_means_operator_identity():
    producer = identity_map(relation="tense_edit", source_space_hash="s")
    sample = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(producer.transform(sample), sample)
    assert "semantic" not in producer.artifact.spec.kind


def test_selection_reads_verdicts_not_loss():
    from relate.evaluation import (
        PreservationProfile,
        PreservationVerdict,
        ScopeVerdict,
    )

    def profile(verdict):
        return PreservationProfile(
            source_space_hash="s", candidate_space_hash="c",
            target_space_hash="t",
            verdicts=(ScopeVerdict(scope="operator_fidelity", verdict=verdict,
                                   required_capabilities=("operator_fidelity",),
                                   failed_capabilities=()
                                   if verdict == PreservationVerdict.PASS
                                   else ("operator_fidelity",),
                                   rationale=""),))

    assert select_simplest_passing(
        [(2, "lin", profile(PreservationVerdict.PASS)),
         (0, "ident", profile(PreservationVerdict.PASS))],
        scope="operator_fidelity",
        policy_hash="p").selected_operator_id == "ident"
    none = select_simplest_passing(
        [(0, "ident", profile(PreservationVerdict.FAIL)),
         (3, "aff", profile(PreservationVerdict.FAIL))],
        scope="operator_fidelity",
        policy_hash="p")
    assert none.outcome == OperatorSelectionOutcome.NONE_PASS
    assert none.selected_operator_id is None
    assert none.evaluated_operator_ids == ("aff", "ident")
    assert none.policy_hash == "p"
    with pytest.raises(RelateError, match="scope"):
        select_simplest_passing([], scope="")


def test_unknown_scope_fails_closed_in_selection():
    from relate.evaluation import PreservationProfile

    empty = PreservationProfile(source_space_hash="s", candidate_space_hash="c",
                                target_space_hash="t")
    selection = select_simplest_passing([(0, "ident", empty)], scope="nope")
    assert selection.outcome == OperatorSelectionOutcome.NONE_PASS


def test_fit_contract_fails_early():
    source, target = _pairs()
    with pytest.raises(RelateError, match="aligned pairs"):
        fit_constant_delta(source[:1], target[:1], relation="r",
                           source_space_hash="s")
    with pytest.raises(RelateError, match="contract"):
        fit_constant_delta(source, target, relation="r",
                           source_space_hash="s").transform(np.ones((4, 3)))
    assert isinstance(identity_map(relation="r", source_space_hash="s"),
                      IdentityMap)


def test_operator_module_calls_no_models():
    text = (Path(__file__).resolve().parent.parent / "src" / "relate"
            / "transformations" / "operators.py").read_text()
    for forbidden in ("compare_neighborhoods", "linear_cka", "roc_curve",
                      "evaluate_hard_negatives", "calibrate(", "def usable_for",
                      "openai", "anthropic", "transformers", "requests",
                      "huggingface", "llm"):
        assert forbidden not in text, forbidden
