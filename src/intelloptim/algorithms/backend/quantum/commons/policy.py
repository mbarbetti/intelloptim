"""The quantum policy: circuit outcome probabilities -> a mixed-variable mixture.

A policy maps the measured probability vector ``p`` of the circuit, through a
small classical affine head and a few linear read-outs, to a
:class:`~intelloptim.algorithms.backend.quantum.commons.distributions.DistState`.
It also provides the analytic backward pass from gradients of that distribution
to gradients of ``p`` and of the head. Chaining ``dL/dp`` with the
parameter-shift Jacobian of ``p`` gives the gradient of the angles.

The circuit's qubits are laid out as ``[mode | feature | category registers]``:

- the **mode** register (mixture only) has ``2 ** n`` outcomes whose
  probabilities are the mixture weights; the Gaus policy has none and a single
  component;
- the **feature** qubits give the ``Z`` moments and ``ZZ`` moments (ring pairs
  for Gaus, all pairs for the mixture) that feed the head, *conditioned on the
  mode* for the mixture. Conditioned moments are ratios ``n_k / (w_k + eps)``,
  shrunk so that a rare mode does not divide by a near-zero probability; their
  derivative uses the quotient rule on the linear numerators and weights;
- the optional **category registers** (register encoding) hold one categorical
  parameter each; the probability of a choice is its Born mass in the mode,
  ``(m_kc + eps) / sum_c' (m_kc' + eps)``, over the valid codewords.

With the ``"head"`` encoding a categorical parameter instead gets a softmax over
an affine map of the same features, and uses no extra qubits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from intelloptim.algorithms.backend.quantum.commons.config import QuantumPolicyConfig
from intelloptim.algorithms.backend.quantum.commons.distributions import (
    DistGrads,
    DistState,
    inverse_softplus,
    softplus,
)
from intelloptim.algorithms.backend.quantum.commons.executor import Ansatz
from intelloptim.algorithms.backend.quantum.commons.measurement import (
    all_pairs,
    moment_matrix,
    ring_pairs,
)
from intelloptim.algorithms.backend.quantum.commons.space import (
    MAX_QUBITS,
    SpaceLayout,
    sigmoid,
)

Phi = Dict[str, np.ndarray]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return np.asarray(e / e.sum(axis=1, keepdims=True))


class QuantumPolicy:
    """Policy of the Gaus and mixture quantum samplers.

    Args:
        config (QuantumPolicyConfig): Hyperparameters and circuit shape.
        layout (SpaceLayout): The parameters to sample.

    Raises:
        ValueError: If the circuit, including category registers, would need
            more than ``MAX_QUBITS`` qubits.
    """

    def __init__(self, config: QuantumPolicyConfig, layout: SpaceLayout) -> None:
        self.config = config
        self.layout = layout
        n_mode, n_feat = config.n_mode_qubits, config.n_feature_qubits
        total = n_mode + n_feat + layout.n_register_qubits
        if total > MAX_QUBITS:
            raise ValueError(
                f"this search space needs {total} qubits ({n_mode} mode, {n_feat} "
                f"feature, {layout.n_register_qubits} category-register); at most "
                f"{MAX_QUBITS} are supported. Use categorical_encoding='head' or "
                "fewer qubits."
            )
        self.ansatz = Ansatz(total, config.n_layers)
        self._k = 1 << n_mode
        self._n_mode = n_mode
        index = np.arange(self.ansatz.dim)
        mode_of = index & (self._k - 1)
        feat_of = (index >> n_mode) & ((1 << n_feat) - 1)
        self._w_mat = (mode_of[None, :] == np.arange(self._k)[:, None]).astype(float)
        pairs = ring_pairs(n_feat) if n_mode == 0 else all_pairs(n_feat)
        obs = moment_matrix(n_feat, pairs)
        self._n_features = int(obs.shape[0])
        self._n_mat = self._w_mat[:, None, :] * obs[:, feat_of][None, :, :]
        self._eps_mode = config.min_mode_weight if self._k > 1 else 0.0

        self._registers: List[np.ndarray] = []  # (K, C, dim) per register categorical
        shift = n_mode + n_feat
        for cat in layout.categorical:
            if cat.n_register_qubits == 0:
                self._registers.append(np.empty((0, 0, 0)))
                continue
            code = (index >> shift) & ((1 << cat.n_register_qubits) - 1)
            valid = code[None, :] == np.arange(cat.n_choices)[:, None]
            self._registers.append(self._w_mat[:, None, :] * valid[None, :, :])
            shift += cat.n_register_qubits
        self._use_head_cats = layout.encoding == "head"
        self._cat_sizes = [c.n_choices for c in layout.categorical]
        self._d = len(layout.numeric)
        self._n_logits = sum(self._cat_sizes) if self._use_head_cats else 0

    @property
    def n_components(self) -> int:
        return self._k

    @property
    def n_features(self) -> int:
        return self._n_features

    # -- initial parameters ---------------------------------------------------

    def init_theta(self, rng: np.random.Generator) -> np.ndarray:
        """Angles giving a broad, near-uniform first policy.

        Mode and register qubits start in an even superposition (``RY(pi/2)`` in
        the first layer, small angles after), so the first mode weights and
        category probabilities are roughly even.
        """
        cfg, n = self.config, self.ansatz.n_qubits
        theta = rng.normal(0.0, cfg.theta_init_std, self.ansatz.n_params)
        even = list(range(self._n_mode)) + list(
            range(cfg.n_mode_qubits + cfg.n_feature_qubits, n)
        )
        for layer in range(self.ansatz.n_layers):
            for q in even:
                theta[layer * n + q] = (
                    np.pi / 2.0 if layer == 0 else rng.normal(0.0, 0.1)
                )
        return theta

    def init_phi(self, rng: np.random.Generator) -> Phi:
        cfg, nf, d = self.config, self._n_features, self._d
        scale = 1.0 / np.sqrt(nf)
        phi: Phi = {
            "Wm": rng.normal(0.0, 0.5 * scale, (d, nf)),
            "bm": np.zeros(d),
            "Ws": rng.normal(0.0, 0.1 * scale, (d, nf)),
            "bs": np.full(
                d, inverse_softplus(max(cfg.init_scale - cfg.sigma_min, 1e-3))
            ),
        }
        if self._use_head_cats:
            phi["Wc"] = np.zeros((self._n_logits, nf))
            phi["bc"] = np.zeros(self._n_logits)
        return phi

    # -- forward ----------------------------------------------------------------

    def _forward(self, p: np.ndarray, phi: Phi) -> Tuple[DistState, Dict[str, Any]]:
        w = self._w_mat @ p
        num = self._n_mat @ p
        z = num / (w + self._eps_mode)[:, None] if self._k > 1 else num
        means = z @ phi["Wm"].T + phi["bm"]
        pre = z @ phi["Ws"].T + phi["bs"]
        sigmas = softplus(pre) + self.config.sigma_min
        cats: List[np.ndarray] = []
        cache: Dict[str, Any] = {"w": w, "num": num, "z": z, "pre": pre}
        if self._use_head_cats:
            logits = z @ phi["Wc"].T + phi["bc"]
            start = 0
            for size in self._cat_sizes:
                cats.append(_softmax(logits[:, start : start + size]))
                start += size
        else:
            raw: List[np.ndarray] = []
            for cat, reg in zip(self.layout.categorical, self._registers, strict=True):
                if cat.n_register_qubits == 0:
                    raw.append(np.ones((self._k, cat.n_choices)))
                    cats.append(np.ones((self._k, cat.n_choices)) / cat.n_choices)
                    continue
                mass = reg @ p
                a = mass + self.config.min_category_weight
                raw.append(a)
                cats.append(a / a.sum(axis=1, keepdims=True))
            cache["raw"] = raw
        dist = DistState(w / w.sum(), means, sigmas, cats)
        return dist, cache

    def distribution(self, p: np.ndarray, phi: Phi) -> DistState:
        """Return the sampling distribution for outcome probabilities ``p``."""
        return self._forward(p, phi)[0]

    # -- backward ---------------------------------------------------------------

    def backward(self, p: np.ndarray, phi: Phi, g: DistGrads) -> Tuple[np.ndarray, Phi]:
        """Back-propagate ``g = dL/d(distribution)`` to ``p`` and the head.

        Returns:
            Tuple[np.ndarray, Phi]: ``dL/dp`` (treating ``p`` as free, valid
            along directions that keep it normalised) and the head gradients.
        """
        dist, cache = self._forward(p, phi)
        z, pre = cache["z"], cache["pre"]
        d_pre = g.sigmas * sigmoid(pre)
        grad: Phi = {
            "Wm": g.means.T @ z,
            "bm": g.means.sum(axis=0),
            "Ws": d_pre.T @ z,
            "bs": d_pre.sum(axis=0),
        }
        d_z = g.means @ phi["Wm"] + d_pre @ phi["Ws"]
        grad_p = np.zeros_like(p)

        if self._use_head_cats:
            d_logits = np.zeros((self._k, self._n_logits))
            start = 0
            for probs, g_c, size in zip(
                dist.cats, g.cats, self._cat_sizes, strict=True
            ):
                inner = (g_c * probs).sum(axis=1, keepdims=True)
                d_logits[:, start : start + size] = probs * (g_c - inner)
                start += size
            grad["Wc"] = d_logits.T @ z
            grad["bc"] = d_logits.sum(axis=0)
            d_z = d_z + d_logits @ phi["Wc"]
        else:
            for reg, a, probs, g_c in zip(
                self._registers, cache["raw"], dist.cats, g.cats, strict=True
            ):
                if reg.size == 0:
                    continue
                total = a.sum(axis=1, keepdims=True)
                inner = (g_c * probs).sum(axis=1, keepdims=True)
                grad_p += np.einsum("kc,kci->i", (g_c - inner) / total, reg)

        if self._k > 1:
            denom = (cache["w"] + self._eps_mode)[:, None]
            d_num = d_z / denom
            d_w = g.weights - (d_z * cache["num"] / denom**2).sum(axis=1)
        else:
            d_num, d_w = d_z, g.weights
        grad_p += self._w_mat.T @ d_w + np.einsum("kf,kfi->i", d_num, self._n_mat)
        return grad_p, grad


__all__ = ["Phi", "QuantumPolicy"]
