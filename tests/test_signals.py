"""Signal-bundle tests: composition only, evidence never policy."""

import numpy as np
import pytest

from relate import RelateError, RetrievalPolicy
from relate.evaluation import (
    HardNegativeCase,
    Hubness,
    LocalDensity,
    NeighborhoodStability,
    calibrate,
    cosine_scorer,
    evaluate_hard_negatives,
)
from relate.retrieval import (
    CalibrationRecord,
    ExternalSignals,
    SignalBundle,
    build_signal_bundle,
)
from relate.retrieval.signals import SignalProvenance


def _observation():
    vectors = {
        "a": np.array([1.0, 0.0]),
        "p": np.array([1.0, 0.1]),
        "n": np.array([0.0, 1.0]),
    }
    cases = [
        HardNegativeCase(
            case_id="c0", anchor_id="a", positive_id="p", negative_id="n",
            relation="r",
        )
    ]
    report = evaluate_hard_negatives(cases, vectors, cosine_scorer())
    return report.observations[0]


def test_bundle_extracts_without_computing():
    observation = _observation()
    density = LocalDensity(value=0.8, k=10, metric="cosine")
    hubness = Hubness(count=3, normalized=0.03, k=10, corpus_size=101)
    stability = NeighborhoodStability(value=0.7, k=10, metric="cosine")
    calibration = CalibrationRecord(threshold=0.5, far=0.1, frr=0.1)
    external = ExternalSignals(source="generic-nli", signals={"support": 0.9},
                               verdict="SUPPORT")
    bundle = build_signal_bundle(
        score=0.9,
        observation=observation,
        density=density,
        hubness=hubness,
        stability=stability,
        calibration=calibration,
        external=external,
        scorer_id="cosine_similarity",
        space_hash="s",
        calibration_id="cal-v1",
    )
    assert bundle.margin == pytest.approx(observation.margin)
    assert bundle.local_density == pytest.approx(0.8)
    assert bundle.hubness == pytest.approx(0.03)
    assert bundle.neighborhood_stability == pytest.approx(0.7)
    assert bundle.calibration_decision == "accept"
    assert bundle.external.verdict == "SUPPORT"
    assert bundle.provenance.scorer_id == "cosine_similarity"
    assert bundle.provenance.density == density
    assert bundle.available_signals == (
        "score", "margin", "local_density", "hubness",
        "neighborhood_stability", "calibration_decision", "external",
    )


def test_missing_differs_from_bad():
    full = build_signal_bundle(score=0.1, margin=-0.5)
    assert full.available_signals == ("score", "margin")
    assert full.margin == pytest.approx(-0.5)
    empty = build_signal_bundle(score=0.1)
    assert empty.available_signals == ("score",)
    assert empty.margin is None


def test_margin_disagreement_fails_loudly():
    observation = _observation()
    with pytest.raises(RelateError, match="disagree"):
        build_signal_bundle(
            score=0.9, observation=observation, margin=observation.margin + 1.0
        )


def test_bundle_has_no_policy_surface():
    bundle = build_signal_bundle(score=0.5, margin=0.5)
    for name in ("route", "accept", "reject", "risk_score", "quality_score",
                 "safety_score", "compatibility_score", "is_safe", "decide"):
        assert not hasattr(bundle, name), name
    policy = RetrievalPolicy(name="demo")
    assert policy.route(bundle) == "accept"
    assert policy.route(build_signal_bundle(score=0.5)) == "verify"


def test_external_stays_typed_and_optional():
    geometric = build_signal_bundle(score=0.9, margin=0.4)
    assert geometric.external is None
    assert "external" not in geometric.available_signals
    assert geometric.provenance.external_source == ""
    with_external = build_signal_bundle(
        score=0.9, margin=0.4,
        external=ExternalSignals(source="nli", signals={"e": 0.2}, verdict="CONTRADICT"),
    )
    assert with_external.external.verdict == "CONTRADICT"
    assert with_external.margin == pytest.approx(0.4)


def test_density_hubness_stability_specs():
    vectors = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.5, 0.5]])
    from relate.evaluation import (
        hubness_counts,
        local_density,
        make_hubness,
        shared_neighborhood_stability,
    )

    density = local_density(vectors, 0, k=2)
    assert density.k == 2 and density.metric == "cosine"
    assert -1.0 <= density.value <= 1.0
    counts = hubness_counts(vectors, k=2)
    assert sum(counts) == 4 * 2
    hubness = make_hubness(counts[0], 4, 2)
    assert hubness.normalized == pytest.approx(counts[0] / 3)
    assert hubness.corpus_size == 4
    stability = shared_neighborhood_stability(vectors, 0, 1, k=2)
    assert 0.0 <= stability.value <= 1.0
    assert stability.value > 0.0


def test_calibrated_fit_feeds_bundle():
    fit = calibrate([2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2])
    low_bundle = build_signal_bundle(score=-5.0, calibration=fit.record)
    high_bundle = build_signal_bundle(score=50.0, calibration=fit.record)
    assert low_bundle.calibration_decision == "reject"
    assert high_bundle.calibration_decision == "accept"


def test_provenance_defaults_empty():
    bundle = build_signal_bundle(score=0.0)
    assert isinstance(bundle.provenance, SignalProvenance)
    assert bundle.provenance.scorer_id == ""
