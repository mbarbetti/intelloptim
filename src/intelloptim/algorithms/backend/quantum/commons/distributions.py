"""Probability model of the quantum policies: a mixture over mixed parameters.

The policy is a mixture of ``K`` components (``K = 1`` for the Gaus policy). In
component ``k`` every numeric parameter has a latent Gaussian
``N(m_ki, s_ki^2)`` and every categorical parameter a probability vector
``P_ki``. The *action* of the policy is the tuple ``(k, a, c)``: the component,
the numeric actions (the latent draw ``u`` of a continuous parameter, the bin
index ``j`` of a discrete one) and the category indices. Every log-probability
here is the **joint** one,

``log w_k + sum_i log p(a_i | k) + sum_i log P_ki(c_i)``,

with ``p`` a Gaussian density for continuous parameters and an exact bin mass for
discrete ones. The Jacobian of the latent-to-physical map does not depend on the
policy parameters, so it cancels in PPO ratios; :func:`log_prob_x` includes it
for continuous floats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from intelloptim.algorithms.backend.quantum.commons.space import (
    NumericDim,
    SpaceLayout,
    sigmoid,
)

_LOG_2PI = math.log(2.0 * math.pi)
_FLOOR = 1e-300
_SQRT2 = math.sqrt(2.0)
_erfc = np.vectorize(math.erfc, otypes=[float])


def softplus(a: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(1 + exp(a))``."""
    return np.asarray(np.logaddexp(0.0, np.asarray(a, dtype=float)))


def inverse_softplus(y: float) -> float:
    """Return ``a`` such that ``softplus(a) == y`` (``y`` must be positive)."""
    return float(math.log(math.expm1(y))) if y < 30.0 else float(y)


def to_x(u: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Map latent draws to the box ``[low, high]`` (linear floats)."""
    x = low + (high - low) * sigmoid(u)
    return np.asarray(np.minimum(np.maximum(x, low), high))


def to_u(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Map points of the open box to latent space (the logit)."""
    s = np.clip((np.asarray(x, dtype=float) - low) / (high - low), 1e-15, 1 - 1e-15)
    return np.asarray(np.log(s) - np.log1p(-s))


def log_jacobian(u: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Return ``log |dx/du|`` summed over the last axis (linear floats)."""
    s = sigmoid(u)
    per_dim = np.log(high - low) + np.log(np.maximum(s, _FLOOR))
    per_dim = per_dim + np.log(np.maximum(1.0 - s, _FLOOR))
    return np.asarray(per_dim.sum(axis=-1))


@dataclass(frozen=True)
class DistState:
    """The sampling distribution of one policy version.

    Attributes:
        weights (np.ndarray): Mixture weights, shape ``(K,)``, summing to one.
        means (np.ndarray): Latent means of the numeric parameters, ``(K, d)``.
        sigmas (np.ndarray): Latent standard deviations, ``(K, d)``, positive.
        cats (List[np.ndarray]): Per categorical parameter, the probability of
            each choice in each component, ``(K, C_i)``.
    """

    weights: np.ndarray
    means: np.ndarray
    sigmas: np.ndarray
    cats: List[np.ndarray] = field(default_factory=list)

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[0])

    def to_json(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "sigmas": self.sigmas.tolist(),
            "cats": [c.tolist() for c in self.cats],
        }

    @classmethod
    def from_json(cls, raw: Dict[str, Any]) -> "DistState":
        k = len(raw["weights"])
        return cls(
            weights=np.asarray(raw["weights"], dtype=float),
            means=np.asarray(raw["means"], dtype=float).reshape(k, -1),
            sigmas=np.asarray(raw["sigmas"], dtype=float).reshape(k, -1),
            cats=[np.asarray(c, dtype=float) for c in raw.get("cats", [])],
        )


@dataclass
class DistGrads:
    """Gradients of a scalar with respect to the parameters of a :class:`DistState`."""

    weights: np.ndarray
    means: np.ndarray
    sigmas: np.ndarray
    cats: List[np.ndarray]

    @classmethod
    def zeros(cls, dist: DistState) -> "DistGrads":
        return cls(
            np.zeros_like(dist.weights),
            np.zeros_like(dist.means),
            np.zeros_like(dist.sigmas),
            [np.zeros_like(c) for c in dist.cats],
        )


# -- Gaussian bins -------------------------------------------------------------


def _cdf(z: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * _erfc(-z / _SQRT2))


def _sf(z: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * _erfc(z / _SQRT2))


def _pdf(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi))


def interval_mass(z_lo: np.ndarray, z_hi: np.ndarray) -> np.ndarray:
    """Return ``P(z_lo < Z < z_hi)`` for a standard normal, stable in both tails."""
    upper_tail = z_lo > 0
    return np.asarray(
        np.where(upper_tail, _sf(z_lo) - _sf(z_hi), _cdf(z_hi) - _cdf(z_lo))
    )


def bin_log_prob(
    dim: NumericDim, bins: np.ndarray, mean: np.ndarray, sigma: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``log P(bin)`` and its derivatives wrt the latent mean and std.

    The mass is that of the interval of the latent Gaussian that falls in the
    bin. A bin whose mass underflows is floored and gets a zero gradient, which
    is the gradient of the floored function.
    """
    j = np.asarray(bins, dtype=int)
    z_lo = (dim.logit_boundary(j) - mean) / sigma
    z_hi = (dim.logit_boundary(j + 1) - mean) / sigma
    mass = interval_mass(z_lo, z_hi)
    pdf_lo, pdf_hi = _pdf(z_lo), _pdf(z_hi)
    zp_lo = np.where(np.isfinite(z_lo), z_lo, 0.0) * pdf_lo
    zp_hi = np.where(np.isfinite(z_hi), z_hi, 0.0) * pdf_hi
    d_mean = -(pdf_hi - pdf_lo) / sigma
    d_sigma = -(zp_hi - zp_lo) / sigma
    ok = mass > _FLOOR
    safe = np.where(ok, mass, 1.0)
    return (
        np.log(np.maximum(mass, _FLOOR)),
        np.where(ok, d_mean / safe, 0.0),
        np.where(ok, d_sigma / safe, 0.0),
    )


# -- sampling and log-probabilities ---------------------------------------------


def sample_joint(
    dist: DistState, layout: SpaceLayout, rng: np.random.Generator
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Draw one action ``(k, numeric actions, category indices)``."""
    k = int(rng.choice(dist.n_components, p=dist.weights / dist.weights.sum()))
    u = dist.means[k] + dist.sigmas[k] * rng.standard_normal(dist.means.shape[1])
    actions = u.copy()
    for i, dim in enumerate(layout.numeric):
        if dim.discrete:
            actions[i] = dim.bin_of(sigmoid(u[i]))
    cats = np.asarray(
        [int(rng.choice(len(p[k]), p=p[k] / p[k].sum())) for p in dist.cats],
        dtype=int,
    )
    return k, actions, cats


def joint_log_prob(
    dist: DistState,
    layout: SpaceLayout,
    ks: np.ndarray,
    actions: np.ndarray,
    cats: np.ndarray,
) -> np.ndarray:
    """Log-probability of each action ``(k_i, a_i, c_i)``, shape ``(B,)``."""
    ks = np.asarray(ks, dtype=int)
    logp = np.log(np.maximum(dist.weights[ks], _FLOOR))
    for i, dim in enumerate(layout.numeric):
        m, s = dist.means[ks, i], dist.sigmas[ks, i]
        if dim.discrete:
            logp = logp + bin_log_prob(dim, actions[:, i], m, s)[0]
        else:
            z = (actions[:, i] - m) / s
            logp = logp - 0.5 * z * z - np.log(s) - 0.5 * _LOG_2PI
    for i, probs in enumerate(dist.cats):
        logp = logp + np.log(np.maximum(probs[ks, cats[:, i]], _FLOOR))
    return np.asarray(logp)


def joint_log_prob_grads(
    dist: DistState,
    layout: SpaceLayout,
    ks: np.ndarray,
    actions: np.ndarray,
    cats: np.ndarray,
    coef: np.ndarray,
) -> DistGrads:
    """Return ``sum_i coef_i * d log pi(action_i) / d (w, m, s, P)``."""
    ks = np.asarray(ks, dtype=int)
    grads = DistGrads.zeros(dist)
    np.add.at(grads.weights, ks, coef / np.maximum(dist.weights[ks], _FLOOR))
    for i, dim in enumerate(layout.numeric):
        m, s = dist.means[ks, i], dist.sigmas[ks, i]
        if dim.discrete:
            _, d_m, d_s = bin_log_prob(dim, actions[:, i], m, s)
        else:
            diff = actions[:, i] - m
            d_m = diff / s**2
            d_s = diff**2 / s**3 - 1.0 / s
        np.add.at(grads.means[:, i], ks, coef * d_m)
        np.add.at(grads.sigmas[:, i], ks, coef * d_s)
    for i, probs in enumerate(dist.cats):
        picked = np.maximum(probs[ks, cats[:, i]], _FLOOR)
        np.add.at(grads.cats[i], (ks, cats[:, i]), coef / picked)
    return grads


def marginal_log_prob(dist: DistState, us: np.ndarray) -> np.ndarray:
    """Log-density of continuous latents ``u`` with the component integrated out."""
    diff = us[:, None, :] - dist.means[None, :, :]
    z = diff / dist.sigmas[None, :, :]
    log_normal = -0.5 * (z**2).sum(axis=2) - np.log(dist.sigmas).sum(axis=1)[None, :]
    log_normal = log_normal - 0.5 * dist.means.shape[1] * _LOG_2PI
    log_w = np.log(np.maximum(dist.weights, _FLOOR))[None, :]
    joint = log_w + log_normal
    peak = joint.max(axis=1)
    return np.asarray(peak + np.log(np.exp(joint - peak[:, None]).sum(axis=1)))


def log_prob_x(
    dist: DistState, xs: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """Density of continuous linear floats ``x`` on the box (marginal over ``k``)."""
    us = to_u(xs, low, high)
    return np.asarray(marginal_log_prob(dist, us) - log_jacobian(us, low, high))


# -- entropy ----------------------------------------------------------------------


def _component_entropy(dist: DistState) -> np.ndarray:
    """Entropy per component: latent Gaussians plus categorical distributions."""
    h = (np.log(dist.sigmas) + 0.5 * (_LOG_2PI + 1.0)).sum(axis=1)
    for probs in dist.cats:
        h = h - (probs * np.log(np.maximum(probs, _FLOOR))).sum(axis=1)
    return np.asarray(h)


def entropy(dist: DistState) -> float:
    """Entropy of the mixture: ``H(k) + sum_k w_k H(component k)``.

    Numeric parameters contribute the entropy of their *latent* Gaussian, also
    when discretised, which is a smooth regulariser of the spread.
    """
    w = dist.weights
    h_k = -(w * np.log(np.maximum(w, _FLOOR))).sum()
    return float(h_k + (w * _component_entropy(dist)).sum())


def entropy_grads(dist: DistState) -> DistGrads:
    """Gradient of :func:`entropy` with respect to weights, sigmas and category probabilities."""
    w = dist.weights
    grads = DistGrads.zeros(dist)
    grads.weights = -np.log(np.maximum(w, _FLOOR)) - 1.0 + _component_entropy(dist)
    grads.sigmas = w[:, None] / dist.sigmas
    for i, probs in enumerate(dist.cats):
        grads.cats[i] = w[:, None] * (-np.log(np.maximum(probs, _FLOOR)) - 1.0)
    return grads


__all__ = [
    "DistGrads",
    "DistState",
    "bin_log_prob",
    "entropy",
    "entropy_grads",
    "interval_mass",
    "inverse_softplus",
    "joint_log_prob",
    "joint_log_prob_grads",
    "log_jacobian",
    "log_prob_x",
    "marginal_log_prob",
    "sample_joint",
    "softplus",
    "to_u",
    "to_x",
]
