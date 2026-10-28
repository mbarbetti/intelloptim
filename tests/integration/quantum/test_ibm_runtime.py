"""The IBM Runtime executor, against a stubbed ``qiskit_ibm_runtime`` (no credentials)."""

from __future__ import annotations

import logging
import sys
import types
from typing import Any, Dict, List

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytestmark = pytest.mark.qiskit

from intelloptim.algorithms.backend.quantum.commons.executor import (  # noqa: E402
    Ansatz,
    executor_for,
)
from intelloptim.algorithms.backend.quantum.commons.resources import (  # noqa: E402
    QuantumBudgetExceededError,
    ResourceLedger,
)
from intelloptim.algorithms.backend.quantum.qiskit.ibm_runtime import (  # noqa: E402
    IBMRuntimeExecutor,
)

SECRET = "super-secret-token"


class _Device:
    def __init__(self, name: str) -> None:
        self.name = name


class _Result:
    """One pub result whose counts put almost everything on ``|0..0>``."""

    def __init__(self, n_qubits: int, shots: int) -> None:
        counts = {"0" * n_qubits: shots - 1, "1" * n_qubits: 1}
        self.data = types.SimpleNamespace(
            meas=types.SimpleNamespace(get_counts=lambda: counts)
        )


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Install a fake ``qiskit_ibm_runtime`` and a pass-through transpiler."""
    state: Dict[str, Any] = {"runs": [], "devices": []}

    class Service:
        def backend(self, name: str) -> _Device:
            state["devices"].append(name)
            return _Device(name)

        def least_busy(self, **kwargs: Any) -> _Device:
            state["devices"].append("least_busy")
            return _Device("ibm_least")

    class Sampler:
        def __init__(self, mode: Any) -> None:
            state["mode"] = mode

        def run(self, pubs: List[Any], shots: int) -> Any:
            state["runs"].append((len(pubs), shots))
            n = pubs[0][0].num_qubits
            return types.SimpleNamespace(
                result=lambda: [_Result(n, shots) for _ in pubs]
            )

    module = types.ModuleType("qiskit_ibm_runtime")
    module.QiskitRuntimeService = Service  # type: ignore[attr-defined]
    module.SamplerV2 = Sampler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", module)

    class PassManager:
        def run(self, circuit: Any) -> Any:
            return circuit

    monkeypatch.setattr(
        "qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager",
        lambda **kwargs: PassManager(),
    )
    return state


def test_returns_normalised_frequencies_and_accounts_shots(
    runtime: Dict[str, Any],
) -> None:
    ledger = ResourceLedger()
    executor = IBMRuntimeExecutor(200, "ibm_x", ledger)
    probs = executor.probabilities(Ansatz(3, 1), np.zeros((2, 3)))
    assert probs.shape == (2, 8) and probs.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert probs[0, 0] == pytest.approx(199 / 200) and probs[0, 7] == pytest.approx(
        1 / 200
    )
    assert (ledger.circuits, ledger.shots) == (2, 400)
    assert runtime["runs"] == [(2, 200)] and runtime["devices"] == ["ibm_x"]
    assert executor.shots == 200


def test_picks_the_least_busy_device_by_default(runtime: Dict[str, Any]) -> None:
    IBMRuntimeExecutor(100)
    assert runtime["devices"] == ["least_busy"]


def test_over_budget_batch_is_refused_before_submitting(
    runtime: Dict[str, Any],
) -> None:
    ledger = ResourceLedger(max_shots=500)
    executor = IBMRuntimeExecutor(100, None, ledger)
    with pytest.raises(QuantumBudgetExceededError):
        executor.probabilities(Ansatz(3, 1), np.zeros((6, 3)))
    assert runtime["runs"] == [] and ledger.shots == 0


def test_rejects_non_positive_shots(runtime: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        IBMRuntimeExecutor(0)


def test_credentials_never_reach_logs_or_the_executor(
    runtime: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("QISKIT_IBM_TOKEN", SECRET)
    package_logger = logging.getLogger("intelloptim")
    package_logger.addHandler(caplog.handler)
    package_logger.setLevel(logging.INFO)
    try:
        executor = IBMRuntimeExecutor(100)
        executor.probabilities(Ansatz(3, 1), np.zeros((1, 3)))
    finally:
        package_logger.removeHandler(caplog.handler)
    assert SECRET not in caplog.text and SECRET not in repr(vars(executor))


def test_the_provider_registry_builds_it(runtime: Dict[str, Any]) -> None:
    executor = executor_for("ibm_runtime", shots=50, backend_name="ibm_x")
    assert isinstance(executor, IBMRuntimeExecutor) and runtime["devices"] == ["ibm_x"]
