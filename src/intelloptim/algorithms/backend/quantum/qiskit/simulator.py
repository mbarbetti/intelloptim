"""Ideal (noiseless) simulation with Qiskit's statevector."""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit.quantum_info import Statevector

from intelloptim.algorithms.backend.quantum.commons.executor import Ansatz
from intelloptim.algorithms.backend.quantum.commons.resources import ResourceLedger
from intelloptim.algorithms.backend.quantum.qiskit.circuit import build_circuit


class StatevectorExecutor:
    """Exact simulator, optionally with finite-shot sampling noise.

    Args:
        shots (Optional[int]): Shots per circuit. ``None`` returns the exact
            probabilities; otherwise each row is the empirical frequency of
            ``shots`` draws, which models statistical (not hardware) noise.
        rng (Optional[np.random.Generator]): Source of the shot draws; required
            when *shots* is given, so that runs are reproducible.
        ledger (Optional[ResourceLedger]): Ledger to update. A new one is
            created when omitted.

    Raises:
        ValueError: If *shots* is given without *rng*, or is not positive.
    """

    def __init__(
        self,
        shots: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        ledger: Optional[ResourceLedger] = None,
    ) -> None:
        if shots is not None:
            if shots < 1:
                raise ValueError(f"shots must be positive: {shots}")
            if rng is None:
                raise ValueError("a shot-based simulation needs an rng")
        self.shots = shots
        self._rng = rng
        self.ledger = ledger if ledger is not None else ResourceLedger()

    def probabilities(self, ansatz: Ansatz, thetas: np.ndarray) -> np.ndarray:
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        self.ledger.circuits += thetas.shape[0]
        out = np.empty((thetas.shape[0], ansatz.dim))
        for row, theta in enumerate(thetas):
            state = Statevector.from_instruction(build_circuit(ansatz, theta))
            probs = np.clip(state.probabilities(), 0.0, None)
            probs = probs / probs.sum()
            if self.shots is not None:
                assert self._rng is not None
                self.ledger.shots += self.shots
                probs = self._rng.multinomial(self.shots, probs) / self.shots
            out[row] = probs
        return out


__all__ = ["StatevectorExecutor"]
