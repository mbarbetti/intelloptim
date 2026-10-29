"""PPO ingredients: advantages, the clipped surrogate, Adam and a KL estimate.

Nothing here depends on the policy or the quantum backend; the learner combines
these with the policy's analytic gradients.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from intelloptim.algorithms.backend.quantum.commons.serialization import (
    array_from_json,
    array_to_json,
)


def advantages(rewards: np.ndarray, mode: str) -> np.ndarray:
    """Turn rewards into advantages, using the cohort mean as the baseline.

    Args:
        rewards (np.ndarray): One reward per trial (larger is better).
        mode (str): ``"mean"`` for ``R - mean(R)``, ``"standardize"`` for the
            same divided by the standard deviation, ``"rank"`` for centred and
            scaled ranks (robust to outliers in the objective).

    Returns:
        np.ndarray: Advantages with zero mean (all zeros if the rewards are
        identical).

    Raises:
        ValueError: If *mode* is unknown.
    """
    r = np.asarray(rewards, dtype=float)
    if mode == "mean":
        return r - float(r.mean())
    if mode == "standardize":
        std = float(r.std())
        return (r - float(r.mean())) / std if std > 1e-12 else np.zeros_like(r)
    if mode == "rank":
        if r.size < 2:
            return np.zeros_like(r)
        return (_average_ranks(r) - (r.size - 1) / 2.0) / (r.size - 1)
    raise ValueError(f"unknown advantage mode: {mode!r}")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 0-based ranks, with tied values sharing their average rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size)
    ordered = values[order]
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0
        start = stop + 1
    return ranks


def clipped_surrogate(
    logp_new: np.ndarray, logp_old: np.ndarray, adv: np.ndarray, clip_eps: float
) -> Tuple[float, np.ndarray]:
    """Return the PPO objective (to maximise) and its gradient w.r.t. ``logp_new``.

    The objective is ``mean(min(r * A, clip(r, 1 - e, 1 + e) * A))`` with
    ``r = exp(logp_new - logp_old)``. The gradient follows the branch that the
    ``min`` selects: it is ``r * A / B`` where the unclipped term is active and
    exactly zero where the clipped constant is.
    """
    ratio = np.exp(logp_new - logp_old)
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    unclipped_term = ratio * adv
    clipped_term = clipped * adv
    value = float(np.minimum(unclipped_term, clipped_term).mean())
    active = unclipped_term <= clipped_term
    grad = np.where(active, ratio * adv, 0.0) / adv.shape[0]
    return value, np.asarray(grad)


def kl_estimate(logp_new: np.ndarray, logp_old: np.ndarray) -> float:
    """Estimate ``KL(old || new)`` from samples of the old policy (``k3``)."""
    log_ratio = logp_new - logp_old
    return float(np.mean(np.expm1(log_ratio) - log_ratio))


class Adam:
    """Adam over a dictionary of arrays, with state that can be serialised."""

    def __init__(
        self,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(
        self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Return updated parameters for a gradient-descent step on *grads*."""
        self.t += 1
        out: Dict[str, np.ndarray] = {}
        for name, value in params.items():
            g = grads[name]
            m = self.m.get(name, np.zeros_like(value))
            v = self.v.get(name, np.zeros_like(value))
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g**2
            self.m[name], self.v[name] = m, v
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            out[name] = value - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return out

    def to_json(self) -> Dict[str, object]:
        return {
            "t": self.t,
            "m": {k: array_to_json(v) for k, v in self.m.items()},
            "v": {k: array_to_json(v) for k, v in self.v.items()},
        }

    def load_json(self, raw: Dict[str, object]) -> None:
        self.t = int(raw["t"])  # type: ignore[call-overload]
        self.m = {k: array_from_json(v) for k, v in raw["m"].items()}  # type: ignore[attr-defined]
        self.v = {k: array_from_json(v) for k, v in raw["v"].items()}  # type: ignore[attr-defined]


__all__ = ["Adam", "advantages", "clipped_surrogate", "kl_estimate"]
