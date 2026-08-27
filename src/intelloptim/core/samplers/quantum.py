"""Quantum sampler spec stubs.

These spec classes are constructible — they are real pydantic models that
serialise correctly and participate in the canonical study hash — so user
code can write::

    sampler = intopt.samplers.QuantumQAOA(reps=3)

from day one. Resolving the spec to a runtime sampler requires an external
plugin registered under the ``intelloptim.samplers`` entry-point group; if
no plugin matches, the orchestrator raises :class:`NotImplementedError`
with a pointer to ``docs/samplers/quantum_plugin.md``.
"""

from __future__ import annotations

from typing import Literal, Optional

from intelloptim.core.samplers.base import SamplerKind, BaseRebuildSamplerSpec


class QuantumQAOA(BaseRebuildSamplerSpec):
    """Quantum Approximate Optimization Algorithm sampler spec.

    Reference: Farhi, Goldstone & Gutmann, *A Quantum Approximate
    Optimization Algorithm*, 2014 (arXiv:1411.4028). The search space is
    encoded as a cost Hamiltonian; QAOA returns bitstring samples that
    are decoded back into parameter assignments.

    Args:
        reps: QAOA depth (commonly written ``p``). Larger ``reps`` give
            better approximation ratios at exponential simulation cost.
        optimizer: Classical optimizer driving the variational angles
            (e.g. ``"COBYLA"``, ``"SPSA"``).
        shots: Number of measurement shots per circuit execution.
        seed: Random seed for the variational optimizer.
    """

    kind: Literal[SamplerKind.QUANTUM_QAOA] = SamplerKind.QUANTUM_QAOA

    reps: int = 1
    optimizer: str = "COBYLA"
    shots: int = 1024
    seed: Optional[int] = None


class QuantumVQE(BaseRebuildSamplerSpec):
    """Variational Quantum Eigensolver sampler spec.

    Used as a sampler over discrete search spaces by encoding parameter
    choices into the eigenstates of a problem Hamiltonian.

    Args:
        ansatz: Name of the variational circuit ansatz (e.g.
            ``"RealAmplitudes"``, ``"EfficientSU2"``).
        optimizer: Classical optimizer for the variational parameters.
        shots: Measurement shots per circuit execution.
        seed: Random seed for the variational optimizer.
    """

    kind: Literal[SamplerKind.QUANTUM_VQE] = SamplerKind.QUANTUM_VQE

    ansatz: str = "RealAmplitudes"
    optimizer: str = "COBYLA"
    shots: int = 1024
    seed: Optional[int] = None


__all__ = ["QuantumQAOA", "QuantumVQE"]
