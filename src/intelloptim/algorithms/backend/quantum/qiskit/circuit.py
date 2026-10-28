"""The ansatz of the quantum policies as a Qiskit circuit."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from intelloptim.algorithms.backend.quantum.commons.executor import Ansatz


def build_circuit(ansatz: Ansatz, theta: np.ndarray) -> QuantumCircuit:
    """Return the circuit for the angles *theta* (no measurements).

    Args:
        ansatz (Ansatz): Circuit shape.
        theta (np.ndarray): One angle per ``RY`` gate, layer by layer.

    Raises:
        ValueError: If *theta* does not have ``ansatz.n_params`` entries.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (ansatz.n_params,):
        raise ValueError(f"expected {ansatz.n_params} angles, got shape {theta.shape}")
    circuit = QuantumCircuit(ansatz.n_qubits)
    for layer in range(ansatz.n_layers):
        for qubit in range(ansatz.n_qubits):
            circuit.ry(float(theta[layer * ansatz.n_qubits + qubit]), qubit)
        for a, b in ansatz.entangler_pairs():
            circuit.cz(a, b)
    return circuit


__all__ = ["build_circuit"]
