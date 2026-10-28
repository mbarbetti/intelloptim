"""Sampler spec construction — classical and quantum."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intelloptim.core.samplers import SamplerKind
from intelloptim.core.samplers.optuna import (
    GPSamplerSpec,
    NSGAIISamplerSpec,
    TPESamplerSpec,
)


class TestTPESamplerSpec:
    def test_defaults(self) -> None:
        s = TPESamplerSpec()
        assert s.kind is SamplerKind.TPE
        assert s.runs_remote is False
        assert s.rebuild_policy is None
        assert s.seed is None
        assert s.n_startup_trials == 10

    def test_with_kwargs(self) -> None:
        s = TPESamplerSpec(seed=42, multivariate=True, n_startup_trials=20)
        assert s.seed == 42
        assert s.multivariate is True
        assert s.n_startup_trials == 20


class TestFieldValidation:
    @pytest.mark.parametrize(
        "spec,kwargs",
        [
            (TPESamplerSpec, {"n_startup_trials": -1}),
            (TPESamplerSpec, {"n_ei_candidates": 0}),
            (TPESamplerSpec, {"multivariate": False, "group": True}),
            (GPSamplerSpec, {"n_startup_trials": -1}),
            (NSGAIISamplerSpec, {"population_size": 1}),
            (NSGAIISamplerSpec, {"mutation_prob": 1.5}),
            (NSGAIISamplerSpec, {"crossover_prob": 3.0}),
            (NSGAIISamplerSpec, {"swapping_prob": -0.1}),
        ],
    )
    def test_rejects_invalid_values(self, spec: type, kwargs: dict) -> None:  # type: ignore[type-arg]
        with pytest.raises(ValidationError):
            spec(**kwargs)

    @pytest.mark.parametrize(
        "spec,kwargs",
        [
            (TPESamplerSpec, {"n_startup_trials": 0}),
            (TPESamplerSpec, {"multivariate": True, "group": True}),
            (GPSamplerSpec, {"n_startup_trials": 0}),
            (NSGAIISamplerSpec, {"population_size": 2, "mutation_prob": 0.0}),
            (NSGAIISamplerSpec, {"crossover_prob": 1.0, "swapping_prob": 0.0}),
        ],
    )
    def test_accepts_boundary_values(self, spec: type, kwargs: dict) -> None:  # type: ignore[type-arg]
        spec(**kwargs)
