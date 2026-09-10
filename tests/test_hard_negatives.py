"""Hard-negative evaluator tests: generic, corpus-agnostic, scorer-injected."""

import numpy as np
import pytest

from relate import RelateError, RelationProjection
from relate.evaluation import (
    HardNegativeCase,
    compare_reports,
    cosine_scorer,
    euclidean_scorer,
    evaluate_hard_negatives,
    hard_negative_card,
    random_scorer,
    relation_scorer,
)


def _cases():
    return [
        HardNegativeCase(
            case_id="c1", anchor_id="a", positive_id="p1", negative_id="n1",
            relation="r1", group="g1",
        ),
        HardNegativeCase(
            case_id="c2", anchor_id="a", positive_id="p2", negative_id="n2",
            relation="r1", group="g2",
        ),
        HardNegativeCase(
            case_id="c3", anchor_id="a", positive_id="p3", negative_id="n3",
            relation="r2", group="g1",
        ),
    ]


def _vectors():
    a = np.array([1.0, 0.0])
    return {
        # c1: cosine prefers p1 (WIN with margin 1.0 - 0.0 = 1.0)
        "a": a,
        "p1": np.array([1.0, 0.0]),
        "n1": np.array([0.0, 1.0]),
        # c2: exact tie (identical candidates)
        "p2": np.array([0.0, 1.0]),
        "n2": np.array([0.0, 1.0]),
        # c3: cosine prefers n3 (LOSS, margin -1.0)
        "p3": np.array([0.0, 1.0]),
        "n3": np.array([1.0, 0.0]),
    }


def test_win_tie_loss_are_explicit():
    report = evaluate_hard_negatives(_cases(), _vectors(), cosine_scorer())
    assert (report.wins, report.ties, report.losses) == (1, 1, 1)
    assert report.total == 3
    assert report.accuracy == pytest.approx(1 / 3)
    assert report.accuracy_excluding_ties == pytest.approx(0.5)
    outcomes = {o.case_id: o.outcome for o in report.observations}
    assert outcomes == {"c1": "WIN", "c2": "TIE", "c3": "LOSS"}
    assert report.observations[0].margin == pytest.approx(1.0)


def test_grouped_results_and_worst_groups():
    report = evaluate_hard_negatives(_cases(), _vectors(), cosine_scorer())
    assert report.by_relation["r1"].accuracy == pytest.approx(0.5)
    assert report.by_relation["r2"].accuracy == pytest.approx(0.0)
    assert report.by_group["g1"].total == 2
    # r2 and g2 tie at 0.0 accuracy; keys break the tie alphabetically.
    assert report.worst_groups[:2] == ("g2", "r2")


def test_same_evaluator_different_scorer():
    vectors = _vectors()
    cosine_report = evaluate_hard_negatives(_cases(), vectors, cosine_scorer())
    euclidean_report = evaluate_hard_negatives(_cases(), vectors, euclidean_scorer())
    # negated Euclidean agrees with cosine on normalized vectors here
    assert euclidean_report.accuracy == pytest.approx(cosine_report.accuracy)
    flip = evaluate_hard_negatives(
        _cases(), vectors, lambda a, c: -float(cosine_scorer()(a, c))
    )
    assert flip.accuracy == pytest.approx(1 / 3)
    assert flip.losses == 1  # c1 flips to LOSS, c3 flips to WIN


def test_relation_scorer_flows_through_same_evaluator():
    embeddings = np.array([[2.0, 0.5], [2.5, -1.0], [-1.0, 3.0], [0.0, 0.0]])
    coords = embeddings[:, :1] * 3.0
    projection = RelationProjection.fit(
        embeddings, coords, alpha=1e-6, relation_names=("x",)
    )
    cases = [
        HardNegativeCase(
            case_id="q", anchor_id="a", positive_id="p", negative_id="n",
            relation="x",
        )
    ]
    vectors = {
        "a": np.array([1.0, 100.0]),
        "p": np.array([1.05, -100.0]),  # close in x, far in raw space
        "n": np.array([5.0, 100.0]),  # close in raw space, far in x
    }
    relation_report = evaluate_hard_negatives(cases, vectors, relation_scorer(projection))
    cosine_report = evaluate_hard_negatives(cases, vectors, cosine_scorer())
    assert relation_report.wins == 1
    assert cosine_report.losses == 1


def test_random_scorer_is_seeded():
    first = evaluate_hard_negatives(_cases(), _vectors(), random_scorer(seed=7))
    second = evaluate_hard_negatives(_cases(), _vectors(), random_scorer(seed=7))
    assert [o.margin for o in first.observations] == [
        o.margin for o in second.observations
    ]


def test_compare_reports_gives_deltas():
    vectors = _vectors()
    reference = evaluate_hard_negatives(_cases(), vectors, cosine_scorer())
    candidate = evaluate_hard_negatives(
        _cases(), vectors, lambda a, c: -float(cosine_scorer()(a, c))
    )
    delta = compare_reports(reference, candidate)
    assert delta.accuracy_delta == pytest.approx(0.0)
    assert set(delta.relation_deltas) == {"r1", "r2"}


def test_card_carries_report_without_second_dto():
    report = evaluate_hard_negatives(_cases(), _vectors(), cosine_scorer())
    card = hard_negative_card(
        report,
        evaluation_id="demo-v1",
        space_hash="abc123",
        corpus="demo",
        corpus_hash="def456",
        scorer=cosine_scorer(),
    )
    assert card.metrics["accuracy"] == pytest.approx(1 / 3)
    assert card.per_relation["r2"]["accuracy"] == pytest.approx(0.0)
    assert card.evaluation_id == "demo-v1"
    assert card.corpus_hash == "def456"
    assert card.scorer == "cosine_similarity"


def test_failures_are_loud():
    with pytest.raises(RelateError, match="at least one"):
        evaluate_hard_negatives([], {}, cosine_scorer())
    with pytest.raises(RelateError, match="missing vector"):
        evaluate_hard_negatives(_cases()[:1], {}, cosine_scorer())
    with pytest.raises(RelateError, match="duplicate case_id"):
        evaluate_hard_negatives(
            [_cases()[0], _cases()[0]], _vectors(), cosine_scorer()
        )
