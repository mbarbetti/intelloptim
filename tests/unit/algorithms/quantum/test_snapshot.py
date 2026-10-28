"""Persisted snapshots: shape-preserving JSON and the model digest."""

from __future__ import annotations

import json

import numpy as np
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution

from intelloptim.algorithms.backend.quantum.commons.config import QuantumPolicyConfig
from intelloptim.algorithms.backend.quantum.commons.learner import initial_snapshot
from intelloptim.algorithms.backend.quantum.commons.policy import QuantumPolicy
from intelloptim.algorithms.backend.quantum.commons.ppo import Adam
from intelloptim.algorithms.backend.quantum.commons.serialization import (
    array_from_json,
    array_to_json,
)
from intelloptim.algorithms.backend.quantum.commons.snapshot import (
    Snapshot,
    shape_digest,
)
from intelloptim.algorithms.backend.quantum.commons.space import SpaceLayout
from tests.support.numpy_executor import NumpyExecutor


@pytest.mark.parametrize("shape", [(3,), (2, 4), (0, 5), (0,), (3, 0)])
def test_arrays_keep_their_shape_even_when_empty(shape: tuple) -> None:
    array = np.arange(float(np.prod(shape))).reshape(shape)
    back = array_from_json(json.loads(json.dumps(array_to_json(array))))
    assert back.shape == shape and np.array_equal(back, array)


def test_adam_state_round_trips_with_empty_parameters() -> None:
    adam = Adam(lr=0.1)
    params = {"w": np.zeros((0, 3)), "b": np.ones(2)}
    params = adam.step(params, {"w": np.zeros((0, 3)), "b": np.ones(2)})
    again = Adam(lr=0.1)
    again.load_json(json.loads(json.dumps(adam.to_json())))
    assert again.t == 1 and again.m["w"].shape == (0, 3)
    again.step(params, {"w": np.zeros((0, 3)), "b": np.ones(2)})  # must not raise


def test_snapshot_round_trip_for_a_space_with_only_categoricals() -> None:
    layout = SpaceLayout.from_space(
        {"c": CategoricalDistribution(["a", "b", "c"])}, "head"
    )
    config = QuantumPolicyConfig(
        algorithm="mixture",
        n_mode_qubits=1,
        n_feature_qubits=2,
        categorical_encoding="head",
    )
    policy = QuantumPolicy(config, layout)
    snapshot = initial_snapshot(
        config, policy, NumpyExecutor(), np.random.default_rng(0)
    )
    back = Snapshot.from_json(json.loads(json.dumps(snapshot.to_json())))
    assert back.phi["Wm"].shape == (0, policy.n_features)
    assert np.allclose(back.dist.cats[0], snapshot.dist.cats[0])
    assert back.digest == snapshot.digest and back.names == ["c"]


def test_digest_depends_on_the_shape_and_the_space() -> None:
    config = QuantumPolicyConfig(algorithm="gaus", n_feature_qubits=4)
    a = SpaceLayout.from_space(
        {"x": FloatDistribution(0, 1)}, "register"
    ).digest_payload()
    b = SpaceLayout.from_space(
        {"x": FloatDistribution(0, 2)}, "register"
    ).digest_payload()
    assert shape_digest(config.model_shape(), a) != shape_digest(
        config.model_shape(), b
    )
    other = QuantumPolicyConfig(algorithm="gaus", n_feature_qubits=5)
    assert shape_digest(config.model_shape(), a) != shape_digest(other.model_shape(), a)


def test_unknown_format_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported snapshot format"):
        Snapshot.from_json({"format": 99})
