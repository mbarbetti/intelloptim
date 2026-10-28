"""How a declared search space maps onto the latent space of a quantum policy.

The policy works in an unconstrained latent space ``u`` and squashes it with
``s = sigmoid(u)`` into ``(0, 1)``. Each parameter reads ``s`` differently:

- a **continuous float** maps ``s`` linearly (or in log space) onto its range;
- a **discrete numeric** parameter (integer, or float with a step) has ``n``
  values and splits ``(0, 1)`` into ``n`` bins; the value is the bin that ``s``
  lands in, and the action the policy records is the bin index. Its probability
  is the exact mass of the bin under the latent Gaussian, so it is normalised by
  construction and never a rounded continuous density. Log-scaled integers use
  the half-integer edges in log space, like Optuna;
- a **categorical** parameter is not numeric: it has a probability per choice,
  produced by the policy (see :mod:`~intelloptim.algorithms.backend.quantum.commons.policy`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from optuna import distributions

from intelloptim.core.samplers.quantum import MAX_QUBITS

__all__ = [
    "CategoricalDim",
    "MAX_QUBITS",
    "NumericDim",
    "SpaceLayout",
]


def sigmoid(u: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * np.asarray(u, dtype=float))))


@dataclass(frozen=True)
class NumericDim:
    """A numeric parameter, continuous or discrete.

    Attributes:
        name (str): Parameter name.
        low (float): Lower end of the range.
        high (float): Upper end of the range.
        log (bool): Whether the scale is logarithmic.
        n_bins (int): Number of values for a discrete parameter, ``0`` for a
            continuous one.
        step (float): Distance between discrete values (``1`` for log integers).
        is_int (bool): Whether the value is returned as an ``int``.
    """

    name: str
    low: float
    high: float
    log: bool
    n_bins: int
    step: float
    is_int: bool

    @property
    def discrete(self) -> bool:
        return self.n_bins > 0

    @classmethod
    def from_distribution(
        cls, name: str, dist: distributions.BaseDistribution
    ) -> "NumericDim":
        if isinstance(dist, distributions.IntDistribution):
            n = int((dist.high - dist.low) // dist.step) + 1
            return cls(name, dist.low, dist.high, dist.log, n, float(dist.step), True)
        assert isinstance(dist, distributions.FloatDistribution)
        if dist.step is None:
            return cls(name, dist.low, dist.high, dist.log, 0, 0.0, False)
        n = int(round((dist.high - dist.low) / dist.step)) + 1
        return cls(name, dist.low, dist.high, False, n, float(dist.step), False)

    # -- discrete: bins of s = sigmoid(u) ------------------------------------

    def _log_edges(self) -> Tuple[float, float]:
        return math.log(self.low - 0.5), math.log(self.high + 0.5)

    def boundary(self, j: np.ndarray) -> np.ndarray:
        """Return the edge ``b_j`` of bin ``j`` in ``s`` space, ``j`` in ``0..n``."""
        j = np.asarray(j, dtype=float)
        if self.log:
            lo, hi = self._log_edges()
            return np.asarray((np.log(self.low + j - 0.5) - lo) / (hi - lo))
        return np.asarray(j / self.n_bins)

    def logit_boundary(self, j: np.ndarray) -> np.ndarray:
        """Return ``logit(b_j)``: ``-inf`` for the first edge, ``+inf`` for the last."""
        b = self.boundary(j)
        with np.errstate(divide="ignore"):
            return np.asarray(np.log(b) - np.log1p(-b))

    def bin_of(self, s: np.ndarray) -> np.ndarray:
        """Return the bin index of each ``s`` in ``(0, 1)``."""
        s = np.asarray(s, dtype=float)
        if self.log:
            lo, hi = self._log_edges()
            j = np.floor(np.exp(lo + s * (hi - lo)) - (self.low - 0.5))
        else:
            j = np.floor(s * self.n_bins)
        return np.clip(j, 0, self.n_bins - 1).astype(int)

    def value_of_bin(self, j: int) -> Any:
        value = self.low + self.step * int(j)
        return int(round(value)) if self.is_int else float(value)

    # -- continuous -----------------------------------------------------------

    def value_of_latent(self, u: float) -> float:
        s = float(sigmoid(np.asarray(u)))
        if self.log:
            value = math.exp(
                math.log(self.low) + s * (math.log(self.high) - math.log(self.low))
            )
        else:
            value = self.low + s * (self.high - self.low)
        return float(min(max(value, self.low), self.high))

    def value(self, action: float) -> Any:
        """Return the parameter value for a policy action (latent or bin index)."""
        return (
            self.value_of_bin(int(action))
            if self.discrete
            else self.value_of_latent(action)
        )


@dataclass(frozen=True)
class CategoricalDim:
    """A categorical parameter.

    Attributes:
        name (str): Parameter name.
        choices (Tuple[Any, ...]): The choices, in Optuna's order.
        n_register_qubits (int): Qubits of its register; ``0`` when the choices
            are encoded by the head, or when there is a single choice.
    """

    name: str
    choices: Tuple[Any, ...]
    n_register_qubits: int

    @property
    def n_choices(self) -> int:
        return len(self.choices)


@dataclass(frozen=True)
class SpaceLayout:
    """The numeric and categorical parameters of a study, in a fixed order.

    Parameters are sorted by name within each group, so the layout of the stored
    model is a function of the declared space only.

    Attributes:
        numeric (Tuple[NumericDim, ...]): Numeric parameters.
        categorical (Tuple[CategoricalDim, ...]): Categorical parameters.
        encoding (str): ``"register"`` or ``"head"``.
        dists (Dict[str, BaseDistribution]): The declared distributions.
    """

    numeric: Tuple[NumericDim, ...]
    categorical: Tuple[CategoricalDim, ...]
    encoding: str
    dists: Dict[str, distributions.BaseDistribution]

    @classmethod
    def from_space(
        cls, space: Dict[str, distributions.BaseDistribution], encoding: str
    ) -> "SpaceLayout":
        """Build the layout of *space*.

        Raises:
            ValueError: If *space* is empty, *encoding* is unknown, or a
                distribution is not a float, integer or categorical one.
        """
        if not space:
            raise ValueError("the search space is empty")
        if encoding not in ("register", "head"):
            raise ValueError(f"unknown categorical encoding: {encoding!r}")
        numeric, categorical = [], []
        for name in sorted(space):
            dist = space[name]
            if isinstance(dist, distributions.CategoricalDistribution):
                n = len(dist.choices)
                qubits = (
                    math.ceil(math.log2(n)) if encoding == "register" and n > 1 else 0
                )
                categorical.append(CategoricalDim(name, tuple(dist.choices), qubits))
            elif isinstance(
                dist, (distributions.FloatDistribution, distributions.IntDistribution)
            ):
                numeric.append(NumericDim.from_distribution(name, dist))
            else:
                raise ValueError(f"unsupported distribution for {name!r}: {dist!r}")
        return cls(tuple(numeric), tuple(categorical), encoding, dict(space))

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self.dists))

    @property
    def n_register_qubits(self) -> int:
        return sum(c.n_register_qubits for c in self.categorical)

    def params(self, actions: np.ndarray, cats: np.ndarray) -> Dict[str, Any]:
        """Return the parameter values for one trial's actions."""
        out: Dict[str, Any] = {}
        for dim, action in zip(self.numeric, actions, strict=True):
            out[dim.name] = dim.value(action)
        for cat, index in zip(self.categorical, cats, strict=True):
            out[cat.name] = cat.choices[int(index)]
        return out

    def digest_payload(self) -> Dict[str, Any]:
        """Return what fixes the shape of the stored model, as plain JSON."""
        return {
            "encoding": self.encoding,
            "space": {
                name: distributions.distribution_to_json(self.dists[name])
                for name in self.names
            },
        }
