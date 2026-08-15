"""Unit tests for sampler plugin discovery via entry points."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import optuna
import pytest

from intelloptim.algorithms.samplers import (
    _RESOLVERS,
    SamplerNotImplementedError,
    load_sampler_plugins,
    resolve_sampler,
)
from intelloptim.core.samplers.quantum._stubs import QuantumQAOA, QuantumVQE


def _make_ep(name: str, func: Any) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = func
    return ep


@pytest.fixture(autouse=True)
def _restore_resolvers() -> Any:
    """Restore _RESOLVERS after each test so plugin-loading side-effects don't leak."""
    snapshot = dict(_RESOLVERS)
    yield
    _RESOLVERS.clear()
    _RESOLVERS.update(snapshot)


class TestLoadSamplerPlugins:
    def test_registers_resolver_for_known_kind(self) -> None:
        def fake_resolve(spec: Any) -> optuna.samplers.BaseSampler:
            return optuna.samplers.RandomSampler(seed=99)

        with patch("intelloptim.algorithms.samplers.entry_points") as mock_eps:
            mock_eps.return_value = [_make_ep("quantum_qaoa", fake_resolve)]
            load_sampler_plugins()

        result = resolve_sampler(QuantumQAOA())
        assert isinstance(result, optuna.samplers.RandomSampler)

    def test_replaces_unimplemented_stub_for_vqe(self) -> None:
        with pytest.raises(SamplerNotImplementedError):
            resolve_sampler(QuantumVQE())

        def fake_resolve(spec: Any) -> optuna.samplers.BaseSampler:
            return optuna.samplers.RandomSampler()

        with patch("intelloptim.algorithms.samplers.entry_points") as mock_eps:
            mock_eps.return_value = [_make_ep("quantum_vqe", fake_resolve)]
            load_sampler_plugins()

        assert isinstance(resolve_sampler(QuantumVQE()), optuna.samplers.RandomSampler)

    def test_spec_object_forwarded_to_resolver(self) -> None:
        received: list[Any] = []

        def capturing_resolve(spec: Any) -> optuna.samplers.BaseSampler:
            received.append(spec)
            return optuna.samplers.RandomSampler(seed=spec.seed)

        with patch("intelloptim.algorithms.samplers.entry_points") as mock_eps:
            mock_eps.return_value = [_make_ep("quantum_qaoa", capturing_resolve)]
            load_sampler_plugins()

        spec = QuantumQAOA(reps=3, seed=7)
        resolve_sampler(spec)

        assert len(received) == 1
        assert received[0].reps == 3
        assert received[0].seed == 7

    def test_skips_unknown_kind(self) -> None:
        original = dict(_RESOLVERS)

        with patch("intelloptim.algorithms.samplers.entry_points") as mock_eps:
            mock_eps.return_value = [_make_ep("unknown_kind_xyz", lambda s: None)]
            load_sampler_plugins()

        assert original == _RESOLVERS

    def test_skips_broken_plugin_on_load_error(self) -> None:
        original = dict(_RESOLVERS)
        broken_ep = MagicMock()
        broken_ep.name = "quantum_qaoa"
        broken_ep.load.side_effect = ImportError("qiskit not installed")

        with patch("intelloptim.algorithms.samplers.entry_points") as mock_eps:
            mock_eps.return_value = [broken_ep]
            load_sampler_plugins()

        assert original == _RESOLVERS
        with pytest.raises(SamplerNotImplementedError):
            resolve_sampler(QuantumQAOA())

    def test_empty_entry_points_leaves_resolvers_unchanged(self) -> None:
        original = dict(_RESOLVERS)

        with patch("intelloptim.algorithms.samplers.entry_points") as mock_eps:
            mock_eps.return_value = []
            load_sampler_plugins()

        assert original == _RESOLVERS

    def test_idempotent_when_called_twice(self) -> None:
        def fake_resolve(spec: Any) -> optuna.samplers.BaseSampler:
            return optuna.samplers.RandomSampler(seed=0)

        ep = _make_ep("quantum_qaoa", fake_resolve)
        with patch("intelloptim.algorithms.samplers.entry_points") as mock_eps:
            mock_eps.return_value = [ep]
            load_sampler_plugins()
            load_sampler_plugins()

        assert isinstance(resolve_sampler(QuantumQAOA()), optuna.samplers.RandomSampler)
        assert ep.load.call_count == 2
