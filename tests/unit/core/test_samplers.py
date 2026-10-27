"""Sampler spec construction — classical and quantum."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intelloptim.core.samplers import SamplerKind
from intelloptim.core.samplers.base import RebuildPolicy
from intelloptim.core.samplers.optuna import (
    GPSamplerSpec,
    NSGAIISamplerSpec,
    TPESamplerSpec,
)
from intelloptim.core.samplers.quantum import QuantumQAOA, QuantumVQE


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


class TestQuantumStubs:
    def test_qaoa_constructible(self) -> None:
        s = QuantumQAOA(reps=3, optimizer="SPSA", shots=2048, seed=1)
        assert s.kind is SamplerKind.QUANTUM_QAOA
        assert s.runs_remote is True
        assert isinstance(s.rebuild_policy, RebuildPolicy)
        assert s.rebuild_policy.every_n_tells == 10
        assert s.reps == 3
        assert s.optimizer == "SPSA"

    def test_vqe_constructible(self) -> None:
        s = QuantumVQE(ansatz="EfficientSU2")
        assert s.kind is SamplerKind.QUANTUM_VQE
        assert s.runs_remote is True
        assert s.ansatz == "EfficientSU2"

    def test_quantum_specs_serialize(self) -> None:
        # The whole point of constructible stubs is that they round-trip
        # the wire so a study spec can be hashed before any plugin is
        # available.
        s = QuantumQAOA(reps=2)
        dumped = s.model_dump(mode="json")
        assert dumped["kind"] == "quantum_qaoa"
        assert dumped["reps"] == 2


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
            (QuantumQAOA, {"reps": 0}),
            (QuantumQAOA, {"shots": 0}),
            (QuantumVQE, {"shots": -5}),
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
            (QuantumQAOA, {"reps": 1, "shots": 1}),
        ],
    )
    def test_accepts_boundary_values(self, spec: type, kwargs: dict) -> None:  # type: ignore[type-arg]
        spec(**kwargs)
