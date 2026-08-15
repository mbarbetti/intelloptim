"""Resolver functions for the dummy quantum plugin.

Each function matches the plugin contract:

    (spec: <SpecClass>) -> optuna.samplers.BaseSampler

The entry-point name in pyproject.toml must equal the SamplerKind value
(e.g. ``"quantum_qaoa"``); the server maps it to the right resolver at
startup via ``load_sampler_plugins()``.

A real quantum implementation would replace RandomSampler with a
Qiskit-, PennyLane-, or D-Wave-based sampler that encodes the search
space as a QUBO or ansatz circuit and decodes bitstring outputs into
parameter assignments.
"""

from __future__ import annotations

import optuna

from intelloptim.core.samplers.quantum._stubs import QuantumQAOA, QuantumVQE


def resolve_qaoa(spec: QuantumQAOA) -> optuna.samplers.BaseSampler:
    """Resolve a QuantumQAOA spec to a sampler instance.

    Dummy implementation: returns a seeded RandomSampler so the study
    behaves deterministically in tests without requiring a quantum backend.
    """
    return optuna.samplers.RandomSampler(seed=spec.seed)


def resolve_vqe(spec: QuantumVQE) -> optuna.samplers.BaseSampler:
    """Resolve a QuantumVQE spec to a sampler instance."""
    return optuna.samplers.RandomSampler(seed=spec.seed)
