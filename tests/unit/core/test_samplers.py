"""Sampler spec construction — classical and quantum."""

from __future__ import annotations

from intelloptim.core.samplers import SamplerKind
from intelloptim.core.samplers.base import RebuildPolicy
from intelloptim.core.samplers.optuna import TPESampler as TPE
from intelloptim.core.samplers.quantum import QuantumQAOA, QuantumVQE


class TestTPE:
    def test_defaults(self) -> None:
        s = TPE()
        assert s.kind is SamplerKind.TPE
        assert s.runs_remote is False
        assert s.rebuild_policy is None
        assert s.seed is None
        assert s.n_startup_trials == 10

    def test_with_kwargs(self) -> None:
        s = TPE(seed=42, multivariate=True, n_startup_trials=20)
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
