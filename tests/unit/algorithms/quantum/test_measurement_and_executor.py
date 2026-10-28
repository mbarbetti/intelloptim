"""Moments, parameter-shift Jacobians, the ledger and the executor registry."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from intelloptim.algorithms.backend.quantum.commons import executor as executor_module
from intelloptim.algorithms.backend.quantum.commons.executor import (
    ENTRY_POINT_GROUP,
    Ansatz,
    executor_for,
    register_executor,
)
from intelloptim.algorithms.backend.quantum.commons.measurement import (
    all_pairs,
    circuits_per_gradient,
    moment_matrix,
    parameter_shift,
    ring_pairs,
    z_signs,
)
from intelloptim.algorithms.backend.quantum.commons.resources import (
    QuantumBudgetExceededError,
    ResourceLedger,
)
from tests.support.numpy_executor import NumpyExecutor, reference_probabilities


class TestAnsatz:
    def test_parameter_count_is_one_ry_per_qubit_per_layer(self) -> None:
        assert Ansatz(6, 3).n_params == 18 and Ansatz(6, 3).dim == 64

    def test_ring_of_cz_gates(self) -> None:
        assert Ansatz(4, 1).entangler_pairs() == [(0, 1), (1, 2), (2, 3), (3, 0)]
        assert Ansatz(2, 1).entangler_pairs() == [(0, 1)]

    @pytest.mark.parametrize("args", [(1, 1), (3, 0)])
    def test_rejects_degenerate_shapes(self, args: Any) -> None:
        with pytest.raises(ValueError):
            Ansatz(*args)


class TestMoments:
    def test_z_signs(self) -> None:
        z = z_signs(2)
        assert z[0].tolist() == [1, -1, 1, -1] and z[1].tolist() == [1, 1, -1, -1]

    def test_moments_of_a_basis_state(self) -> None:
        p = np.zeros(8)
        p[0b101] = 1.0  # qubits 0 and 2 are |1>
        assert (moment_matrix(3, [(0, 1), (0, 2)]) @ p).tolist() == [-1, 1, -1, -1, 1]

    def test_pair_helpers(self) -> None:
        assert len(all_pairs(4)) == 6
        assert ring_pairs(4) == [(0, 1), (1, 2), (2, 3), (3, 0)] and ring_pairs(2) == [
            (0, 1)
        ]


class TestParameterShift:
    def test_jacobian_matches_finite_differences(self) -> None:
        ansatz = Ansatz(4, 3)
        theta = np.random.default_rng(0).normal(size=ansatz.n_params)
        executor = NumpyExecutor()
        p, jac = parameter_shift(executor, ansatz, theta)
        assert jac.shape == (ansatz.n_params, ansatz.dim)
        for j in range(ansatz.n_params):
            e = np.zeros_like(theta)
            e[j] = 1e-6
            numeric = (
                reference_probabilities(ansatz, theta + e)
                - reference_probabilities(ansatz, theta - e)
            ) / 2e-6
            assert np.allclose(jac[j], numeric, atol=1e-8)
        assert p.sum() == pytest.approx(1.0)
        assert np.allclose(jac.sum(axis=1), 0.0, atol=1e-12)

    def test_one_batch_of_two_p_plus_one_circuits(self) -> None:
        ansatz = Ansatz(6, 3)
        executor = NumpyExecutor()
        parameter_shift(executor, ansatz, np.zeros(ansatz.n_params))
        assert executor.ledger.circuits == circuits_per_gradient(ansatz) == 37


class TestLedger:
    def test_reserve_records_and_enforces_the_budget(self) -> None:
        ledger = ResourceLedger(max_shots=1000)
        ledger.reserve(3, 100)
        assert (ledger.circuits, ledger.shots) == (3, 300)
        with pytest.raises(QuantumBudgetExceededError):
            ledger.reserve(8, 100)
        assert (ledger.circuits, ledger.shots) == (
            3,
            300,
        )  # a refused batch records nothing

    def test_unlimited_by_default_and_would_fit(self) -> None:
        ledger = ResourceLedger()
        ledger.reserve(10**6, 10**6)
        assert ledger.would_fit(1, 1)
        assert ResourceLedger(shots=900, max_shots=1000).would_fit(1, 100)
        assert not ResourceLedger(shots=900, max_shots=1000).would_fit(1, 101)


class TestExecutorRegistry:
    def test_registered_provider_is_built_with_the_given_arguments(self) -> None:
        register_executor("unit_test_provider", lambda **kw: ("built", kw))  # type: ignore[arg-type,return-value]
        assert executor_for("unit_test_provider", shots=5) == ("built", {"shots": 5})

    def test_unknown_provider_has_a_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(executor_module, "entry_points", lambda group: [])
        with pytest.raises(KeyError, match="no quantum executor for provider 'nope'"):
            executor_for("nope")

    def test_plugins_are_discovered_through_entry_points(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Entry:
            name = "plugin_provider"

            def load(self) -> Any:
                return lambda **kw: ("from_plugin", kw)

        seen = {}

        def fake_entry_points(group: str) -> list:
            seen["group"] = group
            return [Entry()]

        monkeypatch.setattr(executor_module, "entry_points", fake_entry_points)
        executor_module._PROVIDERS.pop("plugin_provider", None)
        assert executor_for("plugin_provider", shots=3) == ("from_plugin", {"shots": 3})
        assert seen["group"] == ENTRY_POINT_GROUP == "intelloptim.quantum_executors"
        # The second call needs no discovery.
        monkeypatch.setattr(executor_module, "entry_points", lambda group: [])
        assert executor_for("plugin_provider")[0] == "from_plugin"
        executor_module._PROVIDERS.pop("plugin_provider", None)

    def test_ibm_runtime_is_registered_lazily(self) -> None:
        assert "ibm_runtime" in executor_module._PROVIDERS
