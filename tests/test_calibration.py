"""Calibration tests: scorer-agnostic, three-way, provenance-bound."""

import numpy as np
import pytest

from relate import RelateError
from relate.evaluation import (
    CalibrationDecision,
    CalibrationScope,
    HardNegativeCase,
    NegativeSetDescriptor,
    calibrate,
    calibration_card,
    content_hash_of_scores,
    describe_scores,
    equal_error_point,
    evaluate_hard_negatives,
    extract_margin_distributions,
    roc_auc,
    roc_curve,
)
from relate.retrieval import CalibrationRecord


def _separated():
    return [3.0, 3.5, 4.0, 5.0], [0.0, 0.5, 1.0, 1.5]


def test_auc_marks_perfect_and_chance():
    positives, negatives = _separated()
    assert roc_auc(positives, negatives) == pytest.approx(1.0)
    assert roc_auc(negatives, positives) == pytest.approx(0.0)
    assert roc_auc([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.5)


def test_eer_lands_between_overlapping_modes():
    positives = [1.0, 1.0, 1.0, 1.0]
    negatives = [0.0, 0.0, 0.0, 0.0]
    thresholds, fars, frrs = roc_curve(positives, negatives)
    threshold, rate = equal_error_point(thresholds, fars, frrs)
    assert rate == pytest.approx(0.0)
    assert threshold == pytest.approx(1.0)


def test_direction_convention_is_universal():
    # Negated distances enter calibration; higher always means more positive.
    near = [-0.1 - 0.01 * i for i in range(12)]
    far = [-5.0 - 0.01 * i for i in range(12)]
    assert calibrate(far, near).auc == pytest.approx(0.0)
    assert calibrate(near, far).auc == pytest.approx(1.0)


def test_three_way_decision_keeps_ambiguity():
    positives = [1.0 + 0.5 * i for i in range(12)]
    negatives = [0.0 + 0.5 * i for i in range(12)]
    fit = calibrate(positives, negatives)
    record = fit.record
    assert record.escalate_high > record.escalate_low
    assert record.decide(100.0) is CalibrationDecision.ACCEPT
    assert record.decide(-100.0) is CalibrationDecision.REJECT
    middle = (record.escalate_low + record.escalate_high) / 2.0
    assert record.decide(middle) is CalibrationDecision.ESCALATE
    assert record.decide(middle) == "escalate"
    assert "is_match" not in dir(record)
    point = fit.operating_point
    assert 0.0 <= point.ambiguity_fraction <= 1.0
    assert point.false_accept_rate <= 0.10 + 1e-9
    assert point.false_reject_rate <= 0.10 + 1e-9


def test_single_threshold_record_without_band():
    record = CalibrationRecord(threshold=0.8, far=0.1, frr=0.1)
    assert record.decide(0.9) == "accept"
    assert record.decide(0.7) == "reject"


def test_infeasible_targets_fail_loudly():
    with pytest.raises(RelateError, match="infeasible"):
        calibrate([1.0, 2.0, 3.0], [1.5, 2.5, 3.5], far_target=1e-9, frr_target=1e-9)


def test_staleness_names_changed_dimensions():
    record = CalibrationRecord(
        threshold=0.8,
        far=0.1,
        frr=0.1,
        space_hash="s1",
        corpus="relate-0.1.0",
        corpus_hash="h1",
        task="retrieval",
        scorer="cosine_similarity",
        scope=CalibrationScope(task="retrieval", domain="finance"),
        negatives=NegativeSetDescriptor(name="hard", kind="mined", n=10),
    )
    assert record.staleness_against(space_hash="s1").stale is False
    assert record.staleness_against().stale is False
    staleness = record.staleness_against(space_hash="s2", corpus_hash="h2")
    assert staleness.stale is True
    assert staleness.changed == ("space_hash", "corpus_hash")
    assert record.is_stale_for(space_hash="s2") is True
    assert record.is_stale_for(space_hash="s1") is False


def test_scope_is_data_not_branching():
    base = dict(threshold=0.8, far=0.1, frr=0.1, space_hash="s")
    finance = CalibrationRecord(scope=CalibrationScope(domain="finance"), **base)
    legal = CalibrationRecord(scope=CalibrationScope(domain="legal"), **base)
    assert finance.staleness_against(domain="finance").stale is False
    assert finance.staleness_against(domain="legal").changed == ("domain",)
    assert legal.staleness_against(domain="finance").changed == ("domain",)


def test_3a_observations_feed_calibration():
    vectors = {
        "a": np.array([1.0, 0.0]),
        "p": np.array([1.0, 0.1]),
        "n": np.array([0.0, 1.0]),
    }
    cases = [
        HardNegativeCase(
            case_id=f"c{i}", anchor_id="a", positive_id="p", negative_id="n",
            relation="r",
        )
        for i in range(6)
    ]
    report = evaluate_hard_negatives(
        cases, vectors, lambda a, c: float(a @ c / (np.linalg.norm(a) * np.linalg.norm(c)))
    )
    paired = extract_margin_distributions(report)
    assert len(paired.positive) == 6 and len(paired.negative) == 6
    fit = calibrate(list(paired.positive), list(paired.negative))
    assert fit.auc == pytest.approx(1.0)
    card = calibration_card(
        fit,
        evaluation_id="seam-v1",
        corpus="demo",
        corpus_hash="abc",
        scorer="cosine_similarity",
    )
    assert card.metrics["auc"] == pytest.approx(1.0)
    assert card.evaluation_id == "seam-v1"


def test_negative_descriptor_hash_is_deterministic():
    scores = [0.1, 0.2, 0.3]
    assert content_hash_of_scores(scores) == content_hash_of_scores(scores)
    descriptor = NegativeSetDescriptor(name="n", kind="random", n=3)
    assert descriptor.content_hash is None
    with pytest.raises(RelateError, match="positive integer"):
        NegativeSetDescriptor(name="n", kind="random", n=0)


def test_distributions_are_explicit_objects():
    dist = describe_scores("neg", [0.0, 1.0, 2.0, 3.0])
    assert dist.n == 4
    assert dist.mean == pytest.approx(1.5)
    assert set(dist.quantiles) == {"p5", "p25", "p50", "p75", "p95"}
    with pytest.raises(RelateError, match="at least two"):
        describe_scores("neg", [1.0])
