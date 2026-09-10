"""Preservation tests: verdicts derive from declared policy, nothing else."""

from pathlib import Path

import pytest

from relate import RelateError
from relate.evaluation import (
    PreservationPolicy,
    ReferenceFrame,
    Requirement,
    ScopeVerdict,
    build_preservation_profile,
    calibrate,
    calibration_transfer_results,
)
from relate.evaluation.preservation import (
    DEFAULT_POLICIES,
    PreservationResult,
    PreservationVerdict,
)


def _result(capability, metric, *, value=None, delta=None, ratio=None,
            frame=ReferenceFrame.TARGET_NATIVE):
    return PreservationResult(
        capability=capability, metric=metric,
        value=value if value is not None else 0.0,
        reference_value=1.0, delta=delta, ratio=ratio,
        reference_frame=frame,
    )


def _profile(results, policies):
    return build_preservation_profile(
        source_space_hash="src", candidate_space_hash="cand",
        target_space_hash="tgt", results=results, policies=policies,
        bridge_id="b", evaluation_correspondence_hash="e",
    )


def test_no_global_score_and_frames_explicit():
    results = [
        _result("counterpart_recovery", "recall_at_1", value=0.94, ratio=0.94),
        _result("neighborhood_structure", "overlap_at_10", value=0.74, ratio=0.74),
    ]
    profile = _profile(results, [])
    assert profile.result("counterpart_recovery", "recall_at_1").value == pytest.approx(0.94)
    assert all(r.reference_frame == ReferenceFrame.TARGET_NATIVE for r in profile.results)
    assert not hasattr(profile, "compatibility_score")
    assert not hasattr(profile, "compatibility")
    with pytest.raises(RelateError, match="reference frame"):
        PreservationResult(capability="c", metric="m", value=1.0, reference_frame="vibes")


def test_verdicts_derive_from_declared_policy():
    results = [
        _result("counterpart_recovery", "recall_at_1", value=0.94, ratio=0.94),
        _result("neighborhood_structure", "overlap_at_10", value=0.74, ratio=0.74),
    ]
    strict = PreservationPolicy(scope="retrieval", requirements=(
        Requirement("counterpart_recovery", "recall_at_1", "min_value", 0.90),
        Requirement("neighborhood_structure", "overlap_at_10", "min_value", 0.90),
    ))
    lenient = PreservationPolicy(scope="retrieval", requirements=(
        Requirement("counterpart_recovery", "recall_at_1", "min_value", 0.90),
        Requirement("neighborhood_structure", "overlap_at_10", "min_value", 0.90,
                    advisory=True),
    ))
    assert _profile(results, [strict]).usable_for("retrieval") is False
    assert _profile(results, [strict]).verdict_for("retrieval").verdict == PreservationVerdict.FAIL
    warned = _profile(results, [lenient])
    assert warned.verdict_for("retrieval").verdict == PreservationVerdict.WARN
    assert warned.usable_for("retrieval") is False
    assert "advisory miss" in warned.verdict_for("retrieval").rationale


def test_unknown_scope_fails_closed():
    profile = _profile(
        [_result("counterpart_recovery", "recall_at_1", value=1.0, ratio=1.0),
         _result("neighborhood_structure", "overlap_at_10", value=0.9, ratio=0.9)],
        list(DEFAULT_POLICIES),
    )
    assert profile.usable_for("retrieval") is True
    assert profile.usable_for("never_measured") is False
    assert "never_measured" not in profile.usable_scopes
    assert "threshold_transfer" in profile.not_usable_for


def test_same_evidence_different_policies():
    results = [_result("counterpart_recovery", "recall_at_1", value=0.91, ratio=0.91)]
    picky = PreservationPolicy(scope="s", requirements=(
        Requirement("counterpart_recovery", "recall_at_1", "min_value", 0.95),))
    relaxed = PreservationPolicy(scope="s", requirements=(
        Requirement("counterpart_recovery", "recall_at_1", "min_value", 0.90),))
    assert _profile(results, [picky]).usable_for("s") is False
    assert _profile(results, [relaxed]).usable_for("s") is True


def test_explain_names_gates():
    profile = _profile(
        [_result("counterpart_recovery", "recall_at_1", value=0.5, ratio=0.5)],
        [PreservationPolicy(scope="retrieval", requirements=(
            Requirement("counterpart_recovery", "recall_at_1", "min_value", 0.90),))],
    )
    text = profile.explain("retrieval")
    assert text.startswith("FAIL retrieval:")
    assert "counterpart_recovery/recall_at_1" in text
    assert "unmeasured" not in text
    assert profile.explain("missing").startswith("UNKNOWN")


def test_per_relation_classes_stay_visible():
    results = [
        _result("hard_negative_ordering", "accuracy_delta", delta=-0.02),
        _result("hard_negative_ordering/negation", "accuracy_delta", delta=-0.21),
        _result("hard_negative_ordering/topic-related", "accuracy_delta", delta=0.0),
    ]
    policy = PreservationPolicy(scope="fine", requirements=(
        Requirement("hard_negative_ordering/negation", "accuracy_delta", "min_delta", -0.05),))
    profile = _profile(results, [policy])
    assert profile.usable_for("fine") is False
    assert profile.result("hard_negative_ordering/negation", "accuracy_delta").delta == pytest.approx(-0.21)
    assert profile.result("hard_negative_ordering", "accuracy_delta").delta == pytest.approx(-0.02)


def test_calibration_transfer_reuses_3c():
    native = calibrate([2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                       [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2])
    shifted_pos = [s - 0.5 for s in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]]
    shifted_neg = [s + 0.5 for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]]
    results = calibration_transfer_results(
        native_fit=native, candidate_positive_scores=shifted_pos,
        candidate_negative_scores=shifted_neg,
    )
    by_metric = {r.metric: r for r in results}
    assert set(by_metric) == {"far_increase", "frr_increase", "ambiguity_increase", "threshold_shift"}
    assert by_metric["far_increase"].delta > 0
    assert by_metric["threshold_shift"].delta != 0
    assert abs(by_metric["threshold_shift"].delta) > 0.05
    assert all(r.reference_frame == ReferenceFrame.TARGET_NATIVE for r in results)
    policy = PreservationPolicy(scope="threshold_transfer", requirements=(
        Requirement("calibration_transfer", "far_increase", "max_delta", 0.05),
        Requirement("calibration_transfer", "threshold_shift", "max_abs_delta", 0.05),
    ))
    assert _profile(results, [policy]).usable_for("threshold_transfer") is False


def test_policy_identity_stable_and_bound():
    policy = PreservationPolicy(scope="retrieval", requirements=(
        Requirement("counterpart_recovery", "recall_at_1", "min_value", 0.90),))
    assert policy.policy_id == PreservationPolicy(scope="retrieval", requirements=(
        Requirement("counterpart_recovery", "recall_at_1", "min_value", 0.90),)).policy_id
    assert len(policy.policy_id) == 16
    profile = _profile([], [policy])
    assert profile.provenance.policy_ids == (policy.policy_id,)
    with pytest.raises(RelateError, match="requirement kind"):
        Requirement("c", "m", "vibes", 0.5)


def test_round_trip_is_evidence_never_authority():
    results = [
        _result("round_trip", "mean_cosine", value=0.99, ratio=0.99,
                frame=ReferenceFrame.SOURCE_NATIVE),
        _result("counterpart_recovery", "recall_at_1", value=0.5, ratio=0.5),
    ]
    policy = PreservationPolicy(scope="retrieval", requirements=(
        Requirement("counterpart_recovery", "recall_at_1", "min_value", 0.90),
        Requirement("round_trip", "mean_cosine", "min_value", 0.90),
    ))
    profile = _profile(results, [policy])
    assert profile.usable_for("retrieval") is False
    assert profile.result("round_trip", "mean_cosine").reference_frame == ReferenceFrame.SOURCE_NATIVE


def test_profile_layer_contains_no_metric_code():
    text = (Path(__file__).resolve().parent.parent / "src" / "relate"
            / "evaluation" / "preservation.py").read_text()
    # Capability vocabulary ("counterpart_recovery" as data) is expected;
    # metric implementations and numeric machinery are not.
    for forbidden in ("compare_neighborhoods(", "compare_geometry(",
                      "counterpart_recovery(", "linear_cka(", "roc_curve(",
                      "roc_auc(", "evaluate_hard_negatives(", "twonn_estimate(",
                      "hubness_counts(", "local_density(", "import numpy", "np."):
        assert forbidden not in text, forbidden


def test_fit_diagnostics_stay_outside():
    with pytest.raises(RelateError, match="source, candidate"):
        build_preservation_profile(
            source_space_hash="", candidate_space_hash="c",
            target_space_hash="t", results=[], policies=[],
        )


def test_verdict_without_results_fails_closed():
    policy = PreservationPolicy(scope="retrieval", requirements=(
        Requirement("counterpart_recovery", "recall_at_1", "min_value", 0.90),))
    profile = _profile([], [policy])
    assert profile.usable_for("retrieval") is False
    assert isinstance(profile.verdict_for("retrieval"), ScopeVerdict)
