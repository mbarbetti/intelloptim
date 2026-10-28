"""Observables and derivatives of the measured distribution.

Everything the policies need is a *linear* function of the outcome probability
vector ``p`` (Z-basis moments, mode probabilities, register masses and their
conditioned numerators), so the derivative of each of them with respect to an
angle follows from the derivative of ``p`` itself. For ``RY`` the parameter-shift
rule gives that exactly: ``dp/dtheta_j = (p(theta_j + pi/2) - p(theta_j - pi/2)) / 2``.
One batch of ``2P + 1`` circuits therefore yields the probabilities and the full
Jacobian, shared by every trial of a cohort.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

import numpy as np

from intelloptim.algorithms.backend.quantum.commons.executor import (
    Ansatz,
    QuantumExecutor,
)


def z_signs(n_qubits: int) -> np.ndarray:
    """Return ``Z`` eigenvalues, shape ``(n_qubits, 2 ** n_qubits)``."""
    index = np.arange(1 << n_qubits)
    return np.asarray(
        [1.0 - 2.0 * ((index >> q) & 1) for q in range(n_qubits)], dtype=float
    )


def ring_pairs(n: int) -> List[Tuple[int, int]]:
    """Neighbouring pairs of a ring of *n* qubits (one pair if ``n == 2``)."""
    if n < 2:
        return []
    if n == 2:
        return [(0, 1)]
    return [(q, (q + 1) % n) for q in range(n)]


def all_pairs(n: int) -> List[Tuple[int, int]]:
    """All unordered pairs of *n* qubits."""
    return list(combinations(range(n), 2))


def moment_matrix(n_qubits: int, pairs: List[Tuple[int, int]]) -> np.ndarray:
    """Return the matrix mapping ``p`` to ``[<Z_q>..., <Z_i Z_j>...]``.

    Shape ``(n_qubits + len(pairs), 2 ** n_qubits)``.
    """
    z = z_signs(n_qubits)
    rows = [z[q] for q in range(n_qubits)] + [z[i] * z[j] for i, j in pairs]
    return np.asarray(rows)


def parameter_shift(
    executor: QuantumExecutor, ansatz: Ansatz, theta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``p(theta)`` and its Jacobian with one batch of ``2P + 1`` circuits.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``p`` with shape ``(2 ** n,)`` and
        ``dp/dtheta`` with shape ``(P, 2 ** n)``.
    """
    n = ansatz.n_params
    shifts = (np.pi / 2.0) * np.eye(n)
    batch = np.vstack([theta[None, :], theta + shifts, theta - shifts])
    probs = executor.probabilities(ansatz, batch)
    base, plus, minus = probs[0], probs[1 : 1 + n], probs[1 + n :]
    return base, 0.5 * (plus - minus)


def circuits_per_gradient(ansatz: Ansatz) -> int:
    """Number of circuits :func:`parameter_shift` executes."""
    return 2 * ansatz.n_params + 1


__all__ = [
    "all_pairs",
    "circuits_per_gradient",
    "moment_matrix",
    "parameter_shift",
    "ring_pairs",
    "z_signs",
]
