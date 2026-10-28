"""Opt-in smoke test on a real IBM Quantum device. Never part of a default run.

Enable it deliberately, on a machine whose environment holds valid credentials::

    INTELLOPTIM_ALLOW_QUANTUM_HARDWARE=1 INTELLOPTIM_RUN_HARDWARE_TESTS=1 \\
        pytest tests/integration/quantum/test_hardware_smoke.py -m hardware

It submits one small batch (a few circuits, few shots) and checks the output
shape; it makes no claim about result quality.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("qiskit_ibm_runtime")
pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        os.environ.get("INTELLOPTIM_RUN_HARDWARE_TESTS") != "1"
        or os.environ.get("INTELLOPTIM_ALLOW_QUANTUM_HARDWARE") != "1",
        reason="hardware tests are opt-in",
    ),
]


def test_one_small_batch_on_the_least_busy_device() -> None:
    from intelloptim.algorithms.backend.quantum.commons.executor import Ansatz
    from intelloptim.algorithms.backend.quantum.commons.resources import ResourceLedger
    from intelloptim.algorithms.backend.quantum.qiskit.ibm_runtime import (
        IBMRuntimeExecutor,
    )

    ledger = ResourceLedger(max_shots=2000)
    executor = IBMRuntimeExecutor(shots=100, ledger=ledger)
    probs = executor.probabilities(Ansatz(3, 1), np.zeros((2, 3)))
    assert probs.shape == (2, 8) and np.allclose(probs.sum(axis=1), 1.0)
    assert ledger.shots == 200
