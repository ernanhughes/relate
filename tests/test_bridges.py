"""Bridge producer tests: one contract, correspondence-driven, no judgment."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from relate import RelateError, SpaceIdentity
from relate.bridges import (
    Bridge,
    BridgeRegistry,
    BridgeSpec,
    bridge_output_space,
    constant_centroid_bridge,
    fit_bridge,
    fit_linear,
    fit_procrustes,
    fit_ridge,
    identity_bridge,
    random_map_bridge,
)
from relate.evaluation import CorrespondenceSet, identity_correspondence


def _anchors(seed=0, n=40, d_in=16, d_out=12):
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(n, d_in))
    mixing = rng.normal(size=(d_in, d_out))
    target = source @ mixing + 0.01 * rng.normal(size=(n, d_out))
    return source, target


def _spec(a="srchash", b="tgthash", method="procrustes", params=None):
    return BridgeSpec(source_space_hash=a, target_space_hash=b, method=method,
                      params=dict(params or {}))


def _correspondence(n):
    return identity_correspondence([f"anchor-{i}" for i in range(n)])


def test_shared_producer_contract():
    source, target = _anchors()
    correspondence = _correspondence(len(source))
    producers = [
        fit_bridge(source_vectors=source, target_vectors=target,
                   correspondence=correspondence,
                   spec=_spec(method="procrustes")),
        fit_bridge(source_vectors=source, target_vectors=target,
                   correspondence=correspondence,
                   spec=_spec(method="linear")),
        fit_bridge(source_vectors=source, target_vectors=target,
                   correspondence=correspondence,
                   spec=_spec(method="ridge", params={"alpha": 0.5})),
    ]
    for bridge in producers:
        assert bridge.direction == "srchash->tgthash"
        out = bridge.transform(source)
        assert out.shape == target.shape
        assert np.isfinite(out).all()
        assert bridge.fit_provenance.n_anchors == len(source)
        assert len(bridge.bridge_id) == 16
        assert len(bridge.parameter_hash) == 16
        assert len(bridge.anchor_set_hash) == 16


def test_fitting_follows_correspondence_not_row_order():
    source, target = _anchors()
    n = len(source)
    correspondence = _correspondence(n)
    plain = fit_procrustes(
        source_vectors=source, target_vectors=target,
        correspondence=correspondence, spec=_spec(),
    )
    permutation = np.random.default_rng(7).permutation(n)
    shuffled_target = target[permutation]
    shuffled_ids = [f"anchor-{i}" for i in range(n)]
    # Original target row i sits at shuffled position inv[i]: pair source
    # row i with exactly that position, not with permutation[i].
    inverse = np.argsort(permutation)
    remapped = CorrespondenceSet(
        ids=tuple(shuffled_ids),
        source_rows=tuple(range(n)),
        target_rows=tuple(int(inverse[i]) for i in range(n)),
    )
    # Same logical anchors through shuffled rows: identical parameters.
    # The artifact ids still differ: provenance binds the specified
    # correspondence object, not just the resolved values.
    matched = fit_procrustes(
        source_vectors=source, target_vectors=shuffled_target,
        correspondence=remapped, spec=_spec(),
    )
    assert matched.parameter_hash == plain.parameter_hash
    assert matched.bridge_id != plain.bridge_id
    assert matched.anchor_set_hash == remapped.content_hash


def test_deterministic_artifact_identity():
    source, target = _anchors()
    correspondence = _correspondence(len(source))
    first = fit_ridge(source_vectors=source, target_vectors=target,
                      correspondence=correspondence, spec=_spec(method="ridge"))
    second = fit_ridge(source_vectors=source, target_vectors=target,
                       correspondence=correspondence, spec=_spec(method="ridge"))
    assert first.bridge_id == second.bridge_id
    assert first.parameter_hash == second.parameter_hash


def test_procrustes_stays_orthogonal():
    source, target = _anchors(d_in=12, d_out=12)
    bridge = fit_procrustes(
        source_vectors=source, target_vectors=target,
        correspondence=_correspondence(len(source)), spec=_spec(),
    )
    gram = bridge.mapping.T @ bridge.mapping
    np.testing.assert_allclose(gram, np.eye(12), atol=1e-9)


def test_controls_share_the_path():
    source, target = _anchors(d_in=10, d_out=10)
    a = SpaceIdentity(model="a", dimensions=10)
    b = SpaceIdentity(model="b", dimensions=10)
    identity = identity_bridge(10, source_space_hash=a.space_hash,
                               target_space_hash=b.space_hash)
    np.testing.assert_allclose(identity.transform(source), source)
    centroid = constant_centroid_bridge(
        target_anchors=target, source_dimensions=10,
        source_space_hash=a.space_hash, target_space_hash=b.space_hash,
    )
    np.testing.assert_allclose(
        centroid.transform(source[:5]), np.tile(target.mean(axis=0), (5, 1))
    )
    first = random_map_bridge(source_dimensions=10, target_dimensions=10, seed=3,
                              source_space_hash=a.space_hash,
                              target_space_hash=b.space_hash)
    second = random_map_bridge(source_dimensions=10, target_dimensions=10, seed=3,
                               source_space_hash=a.space_hash,
                               target_space_hash=b.space_hash)
    assert first.parameter_hash == second.parameter_hash
    assert first.bridge_id == second.bridge_id
    assert first.fit_provenance.seed == 3
    with pytest.raises(RelateError, match="positive integer"):
        identity_bridge(0, source_space_hash="a", target_space_hash="b")


def test_transform_contract_fails_early():
    source, target = _anchors()
    bridge = fit_linear(
        source_vectors=source, target_vectors=target,
        correspondence=_correspondence(len(source)), spec=_spec(method="linear"),
    )
    assert bridge.transform(source[0]).shape == (target.shape[1],)
    with pytest.raises(RelateError, match="contract"):
        bridge.transform(np.ones((4, 3)))
    with pytest.raises(RelateError, match="finite"):
        bad = source.copy()
        bad[0, 0] = np.inf
        bridge.transform(bad)
    with pytest.raises(RelateError, match="vector or matrix"):
        bridge.transform(np.ones((2, 2, 2)))


def test_direction_is_immutable_evidence():
    source, target = _anchors()
    bridge = fit_linear(
        source_vectors=source, target_vectors=target,
        correspondence=_correspondence(len(source)), spec=_spec(method="linear"),
    )
    with pytest.raises(FrozenInstanceError):
        bridge.direction = "other"
    with pytest.raises(RelateError, match="direction"):
        Bridge(
            bridge_id="x", source_space_hash="a", target_space_hash="b",
            direction="b->a", method="linear",
            mapping=np.eye(2), bias=np.zeros(2),
        )


def test_derived_identity_never_native():
    a = SpaceIdentity(model="a", dimensions=16)
    b = SpaceIdentity(model="b", dimensions=12)
    source, target = _anchors()
    bridge = fit_ridge(
        source_vectors=source, target_vectors=target,
        correspondence=_correspondence(len(source)),
        spec=BridgeSpec(source_space_hash=a.space_hash,
                        target_space_hash=b.space_hash, method="ridge"),
    )
    derived = bridge_output_space(a, bridge, b)
    assert derived.space_hash != b.space_hash
    assert derived.space_hash != a.space_hash
    assert derived.derived_from == a.space_hash
    assert bridge.bridge_id in derived.derivation
    assert b.space_hash in derived.derivation
    assert derived.dimensions == 12
    with pytest.raises(RelateError, match="does not match"):
        bridge_output_space(b, bridge, b)


def test_producer_has_no_judgment():
    source, target = _anchors()
    bridge = fit_procrustes(
        source_vectors=source, target_vectors=target,
        correspondence=_correspondence(len(source)), spec=_spec(),
    )
    for name in ("usable_for", "preservation", "not_usable_for",
                 "compatibility_score", "evaluate", "PASS"):
        assert not hasattr(bridge, name), name
    assert bridge.status == "EXPERIMENTAL"
    source_dir = Path(__file__).resolve().parent.parent / "src" / "relate" / "bridges"
    text = "\n".join(
        (source_dir / leaf).read_text()
        for leaf in ("base.py", "linear.py", "procrustes.py", "ridge.py",
                     "controls.py", "fit.py", "registry.py")
    )
    for forbidden in ("relate.evaluation.hard_negatives",
                      "relate.evaluation.neighborhoods",
                      "relate.evaluation.geometry",
                      "relate.evaluation.calibration",
                      "relate.evaluation.baselines",
                      "relate.evaluation.preservation",
                      "usable_for", "PASS", "not_usable_for"):
        assert forbidden not in text, forbidden


def test_registry_lists_without_ranking():
    registry = BridgeRegistry()
    source, target = _anchors()
    correspondence = _correspondence(len(source))
    proc = fit_procrustes(source_vectors=source, target_vectors=target,
                          correspondence=correspondence, spec=_spec())
    lin = fit_linear(source_vectors=source, target_vectors=target,
                     correspondence=correspondence, spec=_spec(method="linear"))
    registry.register(proc)
    registry.register(lin)
    assert registry.register(proc) is proc
    found = registry.find_all("srchash", "tgthash")
    assert {b.method for b in found} == {"procrustes", "linear"}
    assert registry.lookup("srchash", "tgthash") in found
    with pytest.raises(RelateError, match="DENIED"):
        registry.require("srchash", "missing")
    assert not hasattr(registry, "best_for")
    assert registry.get(proc.bridge_id) is proc


def test_unknown_method_rejected():
    with pytest.raises(RelateError, match="unknown bridge method"):
        BridgeSpec(source_space_hash="a", target_space_hash="b", method="mlp")
    with pytest.raises(RelateError, match="unknown fitted bridge method"):
        fit_bridge(source_vectors=np.ones((3, 2)), target_vectors=np.ones((3, 2)),
                   correspondence=_correspondence(3), spec=_spec(method="identity"))
