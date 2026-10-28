"""Unit tests for sampler plugin discovery via entry points."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import optuna
import pytest

from intelloptim.algorithms.samplers import (
    _RESOLVERS,
    load_sampler_plugins,
    resolve_sampler,
)
from intelloptim.core.samplers import RandomSamplerSpec, SASamplerSpec


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
            mock_eps.return_value = [_make_ep("sa", fake_resolve)]
            load_sampler_plugins()

        assert isinstance(
            resolve_sampler(SASamplerSpec()), optuna.samplers.RandomSampler
        )

    def test_replaces_the_builtin_resolver_of_that_kind_only(self) -> None:
        def fake_resolve(spec: Any) -> optuna.samplers.BaseSampler:
            return optuna.samplers.TPESampler()

        with patch("intelloptim.algorithms.samplers.entry_points") as mock_eps:
            mock_eps.return_value = [_make_ep("random", fake_resolve)]
            load_sampler_plugins()

        assert isinstance(
            resolve_sampler(RandomSamplerSpec()), optuna.samplers.TPESampler
        )
        assert not isinstance(
            resolve_sampler(SASamplerSpec()), optuna.samplers.TPESampler
        )

    def test_spec_object_forwarded_to_resolver(self) -> None:
        received: list[Any] = []

        def capturing_resolve(spec: Any) -> optuna.samplers.BaseSampler:
            received.append(spec)
            return optuna.samplers.RandomSampler(seed=spec.seed)

        with patch("intelloptim.algorithms.samplers.entry_points") as mock_eps:
            mock_eps.return_value = [_make_ep("random", capturing_resolve)]
            load_sampler_plugins()

        resolve_sampler(RandomSamplerSpec(seed=7))

        assert len(received) == 1
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
        broken_ep.name = "random"
        broken_ep.load.side_effect = ImportError("dependency not installed")

        with patch("intelloptim.algorithms.samplers.entry_points") as mock_eps:
            mock_eps.return_value = [broken_ep]
            load_sampler_plugins()

        assert original == _RESOLVERS
        assert isinstance(
            resolve_sampler(RandomSamplerSpec()), optuna.samplers.RandomSampler
        )

    def test_empty_entry_points_leaves_resolvers_unchanged(self) -> None:
        original = dict(_RESOLVERS)

        with patch("intelloptim.algorithms.samplers.entry_points") as mock_eps:
            mock_eps.return_value = []
            load_sampler_plugins()

        assert original == _RESOLVERS

    def test_idempotent_when_called_twice(self) -> None:
        def fake_resolve(spec: Any) -> optuna.samplers.BaseSampler:
            return optuna.samplers.TPESampler(seed=0)

        ep = _make_ep("random", fake_resolve)
        with patch("intelloptim.algorithms.samplers.entry_points") as mock_eps:
            mock_eps.return_value = [ep]
            load_sampler_plugins()
            load_sampler_plugins()

        assert isinstance(
            resolve_sampler(RandomSamplerSpec()), optuna.samplers.TPESampler
        )
        assert ep.load.call_count == 2
