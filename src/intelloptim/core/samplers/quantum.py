"""Quantum sampler spec stubs.

These spec classes are constructible — they are real pydantic models that
serialise correctly and participate in the canonical study hash — so user
code can write::

    sampler = intopt.samplers.QuantumQAOA(reps=3)

from day one. Resolving the spec to a runtime sampler requires an external
plugin registered under the ``intelloptim.samplers`` entry-point group; if
no plugin matches, resolution raises
:class:`~intelloptim.algorithms.errors.SamplerNotImplementedError`.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from intelloptim.core.samplers.base import BaseRebuildSamplerSpec, SamplerKind


class QuantumQAOA(BaseRebuildSamplerSpec):
    """Quantum Approximate Optimization Algorithm sampler spec.

    Reference: Farhi, Goldstone & Gutmann, *A Quantum Approximate
    Optimization Algorithm*, 2014 (arXiv:1411.4028). The search space is
    encoded as a cost Hamiltonian; QAOA returns bitstring samples that
    are decoded back into parameter assignments.

    Args:
        reps (int): QAOA depth (commonly written ``p``), at least ``1``. Larger
            ``reps`` give better approximation ratios at exponential
            simulation cost. Defaults to ``1``.
        optimizer (str): Classical optimizer driving the variational angles
            (e.g. ``"COBYLA"``, ``"SPSA"``). Defaults to ``"COBYLA"``.
        shots (int): Number of measurement shots per circuit execution, at
            least ``1``. Defaults to ``1024``.
        seed (Optional[int]): Random seed for the variational optimizer.
            Defaults to ``None``.
    """

    kind: Literal[SamplerKind.QUANTUM_QAOA] = SamplerKind.QUANTUM_QAOA

    reps: int = Field(default=1, ge=1)
    optimizer: str = "COBYLA"
    shots: int = Field(default=1024, ge=1)
    seed: Optional[int] = None


class QuantumVQE(BaseRebuildSamplerSpec):
    """Variational Quantum Eigensolver sampler spec.

    Used as a sampler over discrete search spaces by encoding parameter
    choices into the eigenstates of a problem Hamiltonian.

    Args:
        ansatz (str): Name of the variational circuit ansatz (e.g.
            ``"RealAmplitudes"``, ``"EfficientSU2"``). Defaults to
            ``"RealAmplitudes"``.
        optimizer (str): Classical optimizer for the variational parameters.
            Defaults to ``"COBYLA"``.
        shots (int): Measurement shots per circuit execution, at least ``1``.
            Defaults to ``1024``.
        seed (Optional[int]): Random seed for the variational optimizer.
            Defaults to ``None``.
    """

    kind: Literal[SamplerKind.QUANTUM_VQE] = SamplerKind.QUANTUM_VQE

    ansatz: str = "RealAmplitudes"
    optimizer: str = "COBYLA"
    shots: int = Field(default=1024, ge=1)
    seed: Optional[int] = None


__all__ = ["QuantumQAOA", "QuantumVQE"]
