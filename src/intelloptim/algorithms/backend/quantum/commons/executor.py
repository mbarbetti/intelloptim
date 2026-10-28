"""What executes the circuits of a quantum policy, and how a provider is found.

A :class:`QuantumExecutor` turns circuit angles into outcome probabilities, on a
simulator or on a device. Samplers talk to this protocol only, so a new provider
(AWS, Quantinuum, ...) is one module that implements it. Providers are looked up
by name: built-in ones are registered here, and others are discovered on demand
from the ``intelloptim.quantum_executors`` entry-point group, so a plugin works
inside the optimizer worker without any start-up call.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from intelloptim.algorithms.backend.quantum.commons.resources import ResourceLedger

ALLOW_HARDWARE_ENV = "INTELLOPTIM_ALLOW_QUANTUM_HARDWARE"
"""Worker environment variable that must be ``"1"`` to run on real hardware."""

ENTRY_POINT_GROUP = "intelloptim.quantum_executors"


@dataclass(frozen=True)
class Ansatz:
    """Shape of the layered ``RY`` + ``CZ`` ansatz.

    Each layer applies one ``RY`` rotation per qubit, then a ring of ``CZ``
    gates. With one ``RY`` per qubit per layer there are
    ``P = n_qubits * n_layers`` trainable angles, and each angle appears in
    exactly one gate, so the two-term parameter-shift rule is exact for it.
    The angle of qubit ``q`` in layer ``l`` is ``theta[l * n_qubits + q]``, and
    bit ``q`` of an outcome index is qubit ``q``.

    Attributes:
        n_qubits (int): Number of qubits, at least ``2``.
        n_layers (int): Number of layers, at least ``1``.
    """

    n_qubits: int
    n_layers: int

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValueError(f"n_qubits must be >= 2: {self.n_qubits}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1: {self.n_layers}")

    @property
    def n_params(self) -> int:
        return self.n_qubits * self.n_layers

    @property
    def dim(self) -> int:
        """Number of measurement outcomes, ``2 ** n_qubits``."""
        return 1 << self.n_qubits

    def entangler_pairs(self) -> List[Tuple[int, int]]:
        """Return the ``CZ`` pairs of one layer (a ring, or one pair for 2 qubits)."""
        if self.n_qubits == 2:
            return [(0, 1)]
        return [(q, (q + 1) % self.n_qubits) for q in range(self.n_qubits)]


class QuantumExecutor(Protocol):
    """Runs the ansatz and returns outcome probabilities.

    Attributes:
        ledger (ResourceLedger): Circuits and shots used so far.
        shots (Optional[int]): Shots per circuit, ``None`` for exact output.
    """

    ledger: ResourceLedger

    @property
    def shots(self) -> Optional[int]: ...

    def probabilities(self, ansatz: Ansatz, thetas: np.ndarray) -> np.ndarray:
        """Return outcome probabilities for each row of *thetas*.

        Args:
            ansatz (Ansatz): Circuit shape.
            thetas (np.ndarray): Angles, shape ``(M, ansatz.n_params)``.

        Returns:
            np.ndarray: Shape ``(M, 2 ** n_qubits)``; exact probabilities or
            empirical frequencies, each row summing to one.
        """
        ...


ExecutorFactory = Callable[..., QuantumExecutor]

_PROVIDERS: Dict[str, ExecutorFactory] = {}


def register_executor(provider: str, factory: ExecutorFactory) -> None:
    """Register *factory* as the executor of ``provider``."""
    _PROVIDERS[provider] = factory


def executor_for(provider: str, **kwargs: Any) -> QuantumExecutor:
    """Build the executor of *provider*, discovering plugins if needed.

    Args:
        provider (str): Provider name from the sampler spec.
        **kwargs: Passed to the factory (``shots``, ``ledger``, and the
            provider's own options).

    Raises:
        KeyError: If no executor is registered or installed for *provider*.
    """
    if provider not in _PROVIDERS:
        for entry in entry_points(group=ENTRY_POINT_GROUP):
            if entry.name == provider:
                _PROVIDERS[provider] = entry.load()
                break
    try:
        factory = _PROVIDERS[provider]
    except KeyError as exc:
        raise KeyError(
            f"no quantum executor for provider {provider!r}; install the plugin "
            f"that registers it in the {ENTRY_POINT_GROUP!r} entry-point group"
        ) from exc
    return factory(**kwargs)


def _ibm_runtime_factory(**kwargs: Any) -> QuantumExecutor:
    from intelloptim.algorithms.backend.quantum.qiskit.ibm_runtime import (
        IBMRuntimeExecutor,
    )

    return IBMRuntimeExecutor(**kwargs)


register_executor("ibm_runtime", _ibm_runtime_factory)

__all__ = [
    "ALLOW_HARDWARE_ENV",
    "ENTRY_POINT_GROUP",
    "Ansatz",
    "ExecutorFactory",
    "QuantumExecutor",
    "executor_for",
    "register_executor",
]
