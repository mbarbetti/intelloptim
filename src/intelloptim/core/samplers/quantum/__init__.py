"""Quantum sampler namespace.

The vendor-neutral plugin contract is :class:`QuantumSampler` in
:mod:`intelloptim.core.samplers.quantum.base`. Concrete spec stubs
(:class:`QuantumQAOA`, :class:`QuantumVQE`) are constructible so user code
can write ``intopt.samplers.QuantumQAOA(reps=3)`` from day one; resolving
them to a runtime implementation requires registering an external plugin
under the ``intelloptim.samplers`` entry-point group.

See ``docs/samplers/quantum_plugin.md`` for the plugin contract.
"""

from intelloptim.core.samplers.quantum._stubs import QuantumQAOA, QuantumVQE
from intelloptim.core.samplers.quantum.base import QuantumSampler

__all__ = ["QuantumQAOA", "QuantumSampler", "QuantumVQE"]
