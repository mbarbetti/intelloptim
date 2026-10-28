"""Everything powered by Qiskit: circuit construction, simulation and IBM hardware.

This is the only package of the quantum samplers that imports ``qiskit``, so the
rest of them (:mod:`~intelloptim.algorithms.backend.quantum.commons`) works
without it. A provider that is not based on Qiskit gets a sibling package that
implements the same two pieces: build the circuit, and a
:class:`~intelloptim.algorithms.backend.quantum.commons.executor.QuantumExecutor`.

- :mod:`~intelloptim.algorithms.backend.quantum.qiskit.circuit` — the ansatz as
  a Qiskit circuit.
- :mod:`~intelloptim.algorithms.backend.quantum.qiskit.simulator` — ideal
  statevector simulation, used by the ``Sim`` samplers.
- :mod:`~intelloptim.algorithms.backend.quantum.qiskit.ibm_runtime` — IBM
  Quantum hardware through Qiskit Runtime.
"""
