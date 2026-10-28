"""Accounting of the quantum resources a sampler uses.

Black-box evaluations and quantum executions are different budgets: the first
is the sample-efficiency budget of the study, the second is what costs money
and queue time on hardware. The ledger counts the second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


class QuantumBudgetExceededError(RuntimeError):
    """Raised before submitting circuits that would exceed the shot budget."""


@dataclass
class ResourceLedger:
    """Running totals of circuits executed and shots spent.

    Attributes:
        circuits (int): Number of circuit executions requested.
        shots (int): Number of measurement shots spent (zero for exact
            simulation).
        max_shots (Optional[int]): Hard budget on :attr:`shots`; ``None`` means
            unlimited.
    """

    circuits: int = 0
    shots: int = 0
    max_shots: Optional[int] = None

    def reserve(self, n_circuits: int, shots_per_circuit: int) -> None:
        """Check the budget for a planned batch, then record it.

        Raises:
            QuantumBudgetExceededError: If the batch would exceed ``max_shots``.
                Nothing is recorded in that case.
        """
        needed = n_circuits * shots_per_circuit
        if self.max_shots is not None and self.shots + needed > self.max_shots:
            raise QuantumBudgetExceededError(
                f"{needed} shots requested but only {self.max_shots - self.shots} "
                f"of the {self.max_shots} budgeted remain"
            )
        self.circuits += n_circuits
        self.shots += needed

    def would_fit(self, n_circuits: int, shots_per_circuit: int) -> bool:
        """Return whether a planned batch fits in the remaining budget."""
        if self.max_shots is None:
            return True
        return self.shots + n_circuits * shots_per_circuit <= self.max_shots

    def to_json(self) -> Dict[str, int]:
        return {"circuits": self.circuits, "shots": self.shots}


__all__ = ["QuantumBudgetExceededError", "ResourceLedger"]
