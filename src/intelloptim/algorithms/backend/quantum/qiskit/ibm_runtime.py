"""IBM Quantum hardware through Qiskit Runtime.

Running on a device costs money and queue time, so this executor is guarded: every
batch is checked against the shot budget **before** it is submitted. The opt-in
environment flag is enforced one level up, by the hardware sampler. Credentials
are never taken from, or written to, the sampler spec, the study or the logs:
:class:`~qiskit_ibm_runtime.QiskitRuntimeService` finds them in the worker's
environment or saved account.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from intelloptim._internal.logging import get_logger
from intelloptim.algorithms.backend.quantum.commons.executor import Ansatz
from intelloptim.algorithms.backend.quantum.commons.resources import ResourceLedger
from intelloptim.algorithms.backend.quantum.qiskit.circuit import build_circuit

_log = get_logger(__name__)


class IBMRuntimeExecutor:
    """Executes the ansatz on an IBM Quantum device and returns frequencies.

    Args:
        shots (int): Shots per circuit.
        backend_name (Optional[str]): Device name, or ``None`` for the least
            busy operational device. Defaults to ``None``.
        ledger (Optional[ResourceLedger]): Ledger to update and to enforce the
            shot budget with.

    Raises:
        ValueError: If *shots* is not positive.
        ImportError: If ``qiskit-ibm-runtime`` is not installed.
    """

    def __init__(
        self,
        shots: int,
        backend_name: Optional[str] = None,
        ledger: Optional[ResourceLedger] = None,
    ) -> None:
        if shots < 1:
            raise ValueError(f"shots must be positive: {shots}")
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        self.shots = shots
        self.ledger = ledger if ledger is not None else ResourceLedger()
        service = QiskitRuntimeService()
        device: Any = (
            service.backend(backend_name)
            if backend_name
            else service.least_busy(operational=True, simulator=False)
        )
        self._pass_manager = generate_preset_pass_manager(
            optimization_level=1, backend=device
        )
        self._sampler = SamplerV2(mode=device)
        _log.info("quantum hardware executor ready", extra={"device": device.name})

    def probabilities(self, ansatz: Ansatz, thetas: np.ndarray) -> np.ndarray:
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        # Raises before anything is submitted if the batch exceeds the budget.
        self.ledger.reserve(thetas.shape[0], self.shots)
        circuits = []
        for theta in thetas:
            circuit = build_circuit(ansatz, theta)
            circuit.measure_all()
            circuits.append(self._pass_manager.run(circuit))
        _log.info(
            "submitting quantum job",
            extra={"circuits": len(circuits), "shots": self.shots},
        )
        result = self._sampler.run([(c,) for c in circuits], shots=self.shots).result()
        out = np.zeros((thetas.shape[0], ansatz.dim))
        for row in range(thetas.shape[0]):
            counts = result[row].data.meas.get_counts()
            total = float(sum(counts.values()))
            for bits, count in counts.items():
                out[row, int(bits, 2)] = count / total
        return out


__all__ = ["IBMRuntimeExecutor"]
