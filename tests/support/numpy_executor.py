"""An independent, Qiskit-free simulator of the quantum policies' ansatz.

Unit tests run the sampler logic against this executor, and integration tests use
it as the oracle for Qiskit's output. It applies ``RY`` rotations and the ring of
``CZ`` gates to a real statevector with plain numpy (bit ``q`` of an outcome index
is qubit ``q``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from intelloptim.algorithms.backend.quantum.commons.executor import Ansatz
from intelloptim.algorithms.backend.quantum.commons.resources import ResourceLedger


def reference_probabilities(ansatz: Ansatz, theta: np.ndarray) -> np.ndarray:
    """Return the outcome probabilities of the ansatz for the angles *theta*."""
    n = ansatz.n_qubits
    state = np.zeros(1 << n)
    state[0] = 1.0
    index = np.arange(1 << n)
    for layer in range(ansatz.n_layers):
        for q in range(n):
            c, s = np.cos(theta[layer * n + q] / 2), np.sin(theta[layer * n + q] / 2)
            bit = (index >> q) & 1
            partner = state[index ^ (1 << q)]
            state = np.where(bit == 0, c * state - s * partner, s * partner + c * state)
        for a, b in ansatz.entangler_pairs():
            both = ((index >> a) & 1) & ((index >> b) & 1)
            state = np.where(both == 1, -state, state)
    return state**2


class NumpyExecutor:
    """A :class:`QuantumExecutor` backed by :func:`reference_probabilities`.

    Args:
        shots (Optional[int]): Shots per circuit, ``None`` for exact output.
        rng (Optional[np.random.Generator]): Source of the shot draws.
        ledger (Optional[ResourceLedger]): Ledger to update.
    """

    def __init__(
        self,
        shots: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        ledger: Optional[ResourceLedger] = None,
    ) -> None:
        if shots is not None and rng is None:
            raise ValueError("a shot-based simulation needs an rng")
        self._shots = shots
        self._rng = rng
        self.ledger = ledger if ledger is not None else ResourceLedger()

    @property
    def shots(self) -> Optional[int]:
        return self._shots

    def probabilities(self, ansatz: Ansatz, thetas: np.ndarray) -> np.ndarray:
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        self.ledger.circuits += thetas.shape[0]
        out = np.empty((thetas.shape[0], ansatz.dim))
        for row, theta in enumerate(thetas):
            probs = reference_probabilities(ansatz, theta)
            if self._shots is not None:
                assert self._rng is not None
                self.ledger.shots += self._shots
                probs = (
                    self._rng.multinomial(self._shots, probs / probs.sum())
                    / self._shots
                )
            out[row] = probs
        return out
