"""Observatory + bridges + calibration smoke tests (NumPy only)."""

import numpy as np

from relate import (
    BridgeSpec,
    CalibrationRecord,
    Observatory,
    RelationProjection,
    RetrievalPolicy,
    SignalBundle,
    SpaceIdentity,
    compare_spaces,
    fit_bridge,
)
from relate.bridges import bridge_output_space
from relate.evaluation import identity_correspondence
from relate.spaces.derivation import derive
from relate.transformations import check_compression


def test_backward_compat_relation_projection():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 8))
    y = x[:, :2] * 2.0
    model = RelationProjection.fit(x, y, relation_names=("a", "b"))
    hits = model.search(x[0], x, k=3)
    assert len(hits) == 3


def test_space_hash_stable_and_derived_differs():
    a = SpaceIdentity(model="m", revision="r1", dimensions=8)
    b = SpaceIdentity(model="m", revision="r1", dimensions=8)
    assert a.space_hash == b.space_hash
    c = derive(a, "pca", dimensions=4)
    assert c.space_hash != a.space_hash
    assert c.derived_from == a.space_hash


def test_compare_spaces_denies_mismatch():
    a = SpaceIdentity(model="m", dimensions=4)
    b = SpaceIdentity(model="n", dimensions=4)
    comp = compare_spaces(a, b)
    assert comp.verdict.startswith("DENIED")
    try:
        comp.require_compatible()
    except ValueError:
        pass
    else:
        raise AssertionError("expected denial")


def test_bridge_procrustes_recovers_rotation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    y = x @ q
    a = SpaceIdentity(model="a", dimensions=8)
    b = SpaceIdentity(model="b", dimensions=8)
    ids = [f"row-{i}" for i in range(30)]
    correspondence = identity_correspondence(ids)
    bridge = fit_bridge(
        source_vectors=x, target_vectors=y, correspondence=correspondence,
        spec=BridgeSpec(source_space_hash=a.space_hash,
                        target_space_hash=b.space_hash, method="procrustes"),
    )
    assert not hasattr(bridge, "usable_for")
    mapped = bridge.transform(x)
    cos = np.mean(np.sum(mapped * y, axis=1) / (
        np.linalg.norm(mapped, axis=1) * np.linalg.norm(y, axis=1)))
    assert cos > 0.99
    derived = bridge_output_space(a, bridge, b)
    assert derived.space_hash != b.space_hash
    assert derived.derived_from == a.space_hash


def test_observatory_evaluate_bridge_and_policy():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 8))
    y = x + 0.01 * rng.normal(size=(40, 8))
    runtime = Observatory()
    a = runtime.register_space(model="a", dimensions=8)
    b = runtime.register_space(model="b", dimensions=8)
    ids = [f"row-{i}" for i in range(40)]
    bridge = runtime.fit_bridge(
        x, y, source_space=a, target_space=b,
        correspondence=identity_correspondence(ids),
    )
    profile = runtime.evaluate_bridge(bridge, x, y)
    assert profile.usable_for("retrieval") is True
    assert "threshold_transfer" not in profile.usable_scopes
    assert profile.candidate_space_hash != b.space_hash
    assert "retrieval" in profile.explain("retrieval")


def test_calibration_and_signals():
    cal = CalibrationRecord(threshold=0.8, far=0.24, frr=0.24, auc=0.75,
                            escalate_low=0.7, escalate_high=0.9,
                            space_hash="s", corpus="relate-0.2.0", task="dup")
    assert cal.decide(0.75) == "escalate"
    assert cal.decide(0.95) == "accept"
    assert cal.is_stale_for(corpus="other") is True
    assert RetrievalPolicy(name="demo").route(SignalBundle(score=0.79, margin=0.02)) == "verify"


def test_compression_fail_closed():
    rec = check_compression(source_space_hash="a", derived_space_hash="b",
                            method="truncate-25%",
                            global_geometry_preserved=True)
    assert rec.global_geometry_preserved is True
    assert rec.faithfulness_verified is False
