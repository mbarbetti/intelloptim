"""Qiskit circuit construction and the statevector executor against a numpy oracle."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytestmark = pytest.mark.qiskit

from intelloptim.algorithms.backend.quantum.commons.executor import Ansatz  # noqa: E402
from intelloptim.algorithms.backend.quantum.commons.measurement import (  # noqa: E402
    circuits_per_gradient,
    parameter_shift,
)
from intelloptim.algorithms.backend.quantum.commons.resources import (
    ResourceLedger,
)  # noqa: E402
from intelloptim.algorithms.backend.quantum.qiskit.circuit import (
    build_circuit,
)  # noqa: E402
from intelloptim.algorithms.backend.quantum.qiskit.simulator import (  # noqa: E402
    StatevectorExecutor,
)
from tests.support.numpy_executor import (
    NumpyExecutor,
    reference_probabilities,
)  # noqa: E402


class TestCircuit:
    def test_has_the_expected_gates(self) -> None:
        ops = build_circuit(Ansatz(3, 2), np.zeros(6)).count_ops()
        assert ops["ry"] == 6 and ops["cz"] == 6

    def test_rejects_wrong_angle_count(self) -> None:
        with pytest.raises(ValueError, match="expected 6 angles"):
            build_circuit(Ansatz(3, 2), np.zeros(5))


class TestStatevectorExecutor:
    @pytest.mark.parametrize("shape", [(2, 1), (4, 2), (6, 3), (10, 2)])
    def test_matches_the_independent_numpy_simulation(self, shape: tuple) -> None:
        ansatz = Ansatz(*shape)
        theta = np.random.default_rng(0).normal(size=ansatz.n_params)
        probs = StatevectorExecutor().probabilities(ansatz, theta[None, :])[0]
        assert np.allclose(probs, reference_probabilities(ansatz, theta), atol=1e-12)
        assert probs.sum() == pytest.approx(1.0)

    def test_zero_angles_give_the_ground_state(self) -> None:
        probs = StatevectorExecutor().probabilities(Ansatz(3, 2), np.zeros((1, 6)))[0]
        assert probs[0] == pytest.approx(1.0)

    def test_batches_rows_and_counts_circuits_but_no_shots_when_exact(self) -> None:
        executor = StatevectorExecutor()
        probs = executor.probabilities(
            Ansatz(3, 1), np.random.default_rng(1).normal(size=(5, 3))
        )
        assert probs.shape == (5, 8) and np.allclose(probs.sum(axis=1), 1.0)
        assert (executor.ledger.circuits, executor.ledger.shots) == (5, 0)

    def test_shots_are_reproducible_unbiased_and_counted(self) -> None:
        ansatz, theta = Ansatz(3, 1), np.array([[0.7, 1.1, -0.4]])
        exact = StatevectorExecutor().probabilities(ansatz, theta)[0]
        a = StatevectorExecutor(2000, np.random.default_rng(3))
        b = StatevectorExecutor(2000, np.random.default_rng(3))
        pa = a.probabilities(ansatz, theta)[0]
        assert np.array_equal(pa, b.probabilities(ansatz, theta)[0])
        assert pa.sum() == pytest.approx(1.0) and np.abs(pa - exact).max() < 0.05
        assert (a.ledger.circuits, a.ledger.shots) == (1, 2000)
        many = StatevectorExecutor(200, np.random.default_rng(4))
        mean = many.probabilities(ansatz, np.repeat(theta, 400, axis=0)).mean(axis=0)
        assert np.abs(mean - exact).max() < 0.01

    def test_shot_mode_needs_an_rng_and_positive_shots(self) -> None:
        with pytest.raises(ValueError, match="rng"):
            StatevectorExecutor(100)
        with pytest.raises(ValueError, match="positive"):
            StatevectorExecutor(0, np.random.default_rng(0))

    def test_parameter_shift_agrees_with_the_numpy_executor(self) -> None:
        ansatz = Ansatz(5, 2)
        theta = np.random.default_rng(2).normal(size=ansatz.n_params)
        qiskit_exec, numpy_exec = StatevectorExecutor(), NumpyExecutor()
        p_q, jac_q = parameter_shift(qiskit_exec, ansatz, theta)
        p_n, jac_n = parameter_shift(numpy_exec, ansatz, theta)
        assert np.allclose(p_q, p_n, atol=1e-12) and np.allclose(
            jac_q, jac_n, atol=1e-12
        )
        assert qiskit_exec.ledger.circuits == circuits_per_gradient(ansatz)

    def test_uses_the_given_ledger(self) -> None:
        ledger = ResourceLedger()
        StatevectorExecutor(ledger=ledger).probabilities(Ansatz(2, 1), np.zeros((3, 2)))
        assert ledger.circuits == 3
