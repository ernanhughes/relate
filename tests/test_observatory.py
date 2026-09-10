"""Observatory + bridges + calibration smoke tests (NumPy only)."""

import numpy as np
import pytest

from relate import (
    BridgeEvaluation,
    BridgeSpec,
    CalibrationRecord,
    Observatory,
    RelateError,
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
    evaluation = runtime.evaluate_bridge_full(
        bridge, x, y, correspondence=identity_correspondence(ids),
        source_space=a, target_space=b,
    )
    assert isinstance(evaluation, BridgeEvaluation)
    assert evaluation.comparison.neighborhood.mean_overlap > 0.95
    assert evaluation.candidate_space.space_hash != b.space_hash
    profile = evaluation.profile
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


def _canonical_flow(seed=0):
    import hashlib
    import json

    from relate.evaluation import DEFAULT_POLICIES

    rng = np.random.default_rng(seed)
    latent = rng.uniform(-3.0, 3.0, size=(120, 3))
    mixing = rng.normal(size=(3, 8))
    nuisance = rng.normal(size=(120, 8))
    nuisance *= 2.0 / np.linalg.norm(nuisance, axis=1, keepdims=True)
    signal = latent @ mixing
    signal *= 1.5 / np.linalg.norm(signal, axis=1, keepdims=True)
    target = signal + nuisance
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    source = target + 0.05 * rng.normal(size=target.shape)
    source /= np.linalg.norm(source, axis=1, keepdims=True)

    runtime = Observatory()
    space_a = runtime.register_space(model="canonical-a", dimensions=8)
    space_b = runtime.register_space(model="canonical-b", dimensions=8)
    source_vectors = runtime.attach(source, space=space_a)
    target_vectors = runtime.attach(target, space=space_b)
    ids = [f"item-{i}" for i in range(120)]
    native = runtime.compare_spaces(
        source_vectors, target_vectors,
        correspondence=identity_correspondence(ids),
        source_space=space_a, target_space=space_b,
    )
    assert native.source_space_hash == space_a.space_hash
    assert not hasattr(native, "usable_for")
    bridge = runtime.fit_bridge(
        source_vectors, target_vectors, source_space=space_a,
        target_space=space_b, correspondence=identity_correspondence(ids),
        method="ridge", params={"alpha": 1.0},
    )
    assert runtime.bridges.get(bridge.bridge_id) is bridge
    derived = runtime.bridge_space(space_a, bridge, space_b)
    evaluation = runtime.evaluate_bridge_full(
        bridge, source_vectors, target_vectors,
        correspondence=identity_correspondence(ids),
        source_space=space_a, target_space=space_b,
        policies=list(DEFAULT_POLICIES),
    )
    assert evaluation.candidate_space.space_hash == derived.space_hash
    replay = {
        "bridge_id": bridge.bridge_id,
        "parameter_hash": bridge.parameter_hash,
        "derived_space_hash": derived.space_hash,
        "cka": native.geometry.cka,
        "counterpart": evaluation.comparison.counterpart.top1,
        "verdicts": [(v.scope, v.verdict.value) for v in evaluation.profile.verdicts],
        "policies": [p.policy_id for p in DEFAULT_POLICIES],
    }
    digest = hashlib.sha256(
        json.dumps(replay, sort_keys=True).encode()).hexdigest()[:16]
    return replay, digest


def test_canonical_flow_replays_deterministically():
    first, first_hash = _canonical_flow()
    second, second_hash = _canonical_flow()
    assert first == second
    assert first_hash == second_hash
    assert len(first["bridge_id"]) == 16


def test_evaluation_refuses_identity_mistakes():
    from relate.bridges import BridgeMismatchError

    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 6))
    y = rng.normal(size=(20, 6))
    runtime = Observatory()
    a = runtime.register_space(model="a", dimensions=6)
    b = runtime.register_space(model="b", dimensions=6)
    c = runtime.register_space(model="c", dimensions=6)
    ids = [f"row-{i}" for i in range(20)]
    bridge = runtime.fit_bridge(
        x, y, source_space=a, target_space=b,
        correspondence=identity_correspondence(ids),
    )
    with pytest.raises(BridgeMismatchError, match="BRIDGE SOURCE MISMATCH"):
        runtime.evaluate_bridge(
            bridge, x, y, correspondence=identity_correspondence(ids),
            source_space=c, target_space=b,
        )
    with pytest.raises(BridgeMismatchError, match="BRIDGE TARGET MISMATCH"):
        runtime.evaluate_bridge(
            bridge, x, y, correspondence=identity_correspondence(ids),
            source_space=a, target_space=c,
        )
    with pytest.raises(TypeError, match="correspondence"):
        runtime.evaluate_bridge(bridge, x, y)


def test_fit_refuses_unregistered_spaces():
    from relate import SpaceIdentity

    runtime = Observatory()
    ghost_a = SpaceIdentity(model="ghost-a", dimensions=4)
    ghost_b = SpaceIdentity(model="ghost-b", dimensions=4)
    rng = np.random.default_rng(5)
    with pytest.raises(RelateError, match="unknown space_hash"):
        runtime.fit_bridge(
            rng.normal(size=(8, 4)), rng.normal(size=(8, 4)),
            source_space=ghost_a, target_space=ghost_b,
            correspondence=identity_correspondence([f"g{i}" for i in range(8)]),
        )
