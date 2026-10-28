"""The quantum policy: layout of qubits, read-outs and analytic backward passes."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from intelloptim.algorithms.backend.quantum.commons.config import QuantumPolicyConfig
from intelloptim.algorithms.backend.quantum.commons.distributions import (
    joint_log_prob,
    joint_log_prob_grads,
    sample_joint,
)
from intelloptim.algorithms.backend.quantum.commons.policy import QuantumPolicy
from intelloptim.algorithms.backend.quantum.commons.space import SpaceLayout
from tests.support.numpy_executor import NumpyExecutor

MIXED: Dict[str, BaseDistribution] = {
    "x": FloatDistribution(-1.0, 1.0),
    "lr": FloatDistribution(1e-3, 1.0, log=True),
    "n": IntDistribution(1, 16),
    "act": CategoricalDistribution(["relu", "tanh", "gelu"]),
    "flag": CategoricalDistribution([True, False]),
}
CONTINUOUS: Dict[str, BaseDistribution] = {
    f"x{i}": FloatDistribution(-1.0, 1.0) for i in range(3)
}


def _policy(
    algorithm: str, encoding: str, space: Dict[str, BaseDistribution]
) -> QuantumPolicy:
    if algorithm == "gaus":
        config = QuantumPolicyConfig(
            algorithm="gaus", n_feature_qubits=4, categorical_encoding=encoding
        )
    else:
        config = QuantumPolicyConfig(
            algorithm="mixture",
            n_mode_qubits=2,
            n_feature_qubits=3,
            categorical_encoding=encoding,
        )
    return QuantumPolicy(config, SpaceLayout.from_space(space, encoding))


def _tangent(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=n)
    return v - v.mean()  # probabilities stay normalised along the direction


CASES = [
    ("gaus", "register", MIXED),
    ("gaus", "head", MIXED),
    ("mixture", "register", MIXED),
    ("mixture", "head", MIXED),
    ("gaus", "register", CONTINUOUS),
    ("mixture", "head", CONTINUOUS),
]
IDS = [f"{a}-{e}-{'mixed' if s is MIXED else 'float'}" for a, e, s in CASES]


@pytest.mark.parametrize("algorithm, encoding, space", CASES, ids=IDS)
class TestBackward:
    def test_gradient_wrt_probabilities_and_head(
        self, algorithm: str, encoding: str, space: Dict[str, BaseDistribution]
    ) -> None:
        rng = np.random.default_rng(7)
        policy = _policy(algorithm, encoding, space)
        phi = policy.init_phi(rng)
        # Non-trivial head: random logits so the softmax gradients are exercised.
        for key in ("Wc", "bc"):
            if key in phi:
                phi[key] = rng.normal(0, 0.5, phi[key].shape)
        p = rng.dirichlet(np.ones(policy.ansatz.dim))
        dist = policy.distribution(p, phi)
        draws = [sample_joint(dist, policy.layout, rng) for _ in range(10)]
        ks = np.array([d[0] for d in draws])
        acts = np.array([d[1] for d in draws])
        cats = np.array([d[2] for d in draws])
        coef = rng.normal(size=10)

        def f(p_: np.ndarray, phi_: dict) -> float:
            d = policy.distribution(p_, phi_)
            return float(
                (coef * joint_log_prob(d, policy.layout, ks, acts, cats)).sum()
            )

        grad_p, grad_phi = policy.backward(
            p, phi, joint_log_prob_grads(dist, policy.layout, ks, acts, cats, coef)
        )
        for _ in range(3):
            v = _tangent(rng, p.size)
            num = (f(p + 1e-6 * v, phi) - f(p - 1e-6 * v, phi)) / 2e-6
            assert num == pytest.approx(float(grad_p @ v), abs=1e-4)
        for name, g in grad_phi.items():
            for idx in np.ndindex(g.shape):
                hi = {k: v.copy() for k, v in phi.items()}
                lo = {k: v.copy() for k, v in phi.items()}
                hi[name][idx] += 1e-6
                lo[name][idx] -= 1e-6
                assert (f(p, hi) - f(p, lo)) / 2e-6 == pytest.approx(
                    g[idx], abs=1e-4
                ), name

    def test_initial_policy_is_broad_and_valid(
        self, algorithm: str, encoding: str, space: Dict[str, BaseDistribution]
    ) -> None:
        policy = _policy(algorithm, encoding, space)
        rng = np.random.default_rng(0)
        theta, phi = policy.init_theta(rng), policy.init_phi(rng)
        p = NumpyExecutor().probabilities(policy.ansatz, theta[None])[0]
        dist = policy.distribution(p, phi)
        assert dist.weights.sum() == pytest.approx(1.0)
        assert np.all(dist.sigmas > 0.5)
        for probs, cat in zip(dist.cats, policy.layout.categorical, strict=True):
            assert probs.sum(axis=1) == pytest.approx(np.ones(policy.n_components))
            assert probs.shape == (policy.n_components, cat.n_choices)
            assert probs.min() > 0.05  # near uniform at the start


class TestQubitLayout:
    def test_register_encoding_adds_register_qubits(self) -> None:
        reg = _policy("gaus", "register", MIXED)
        head = _policy("gaus", "head", MIXED)
        assert head.ansatz.n_qubits == 4
        assert reg.ansatz.n_qubits == 4 + 2 + 1  # 3 choices -> 2 qubits, 2 choices -> 1

    def test_mixture_qubits_are_mode_then_features_then_registers(self) -> None:
        policy = _policy("mixture", "register", MIXED)
        assert policy.ansatz.n_qubits == 2 + 3 + 3 and policy.n_components == 4

    def test_too_many_qubits_is_refused_with_advice(self) -> None:
        space = {f"c{i}": CategoricalDistribution(list(range(16))) for i in range(5)}
        with pytest.raises(ValueError, match="at most 16 are supported"):
            _policy("gaus", "register", space)
        _policy("gaus", "head", space)  # the head encoding has no such limit


class TestCategoricalProbabilities:
    def test_register_probabilities_are_the_masked_born_marginal(self) -> None:
        space = {
            "c": CategoricalDistribution(["a", "b", "c"])
        }  # 2 qubits, codeword 3 unused
        policy = _policy("gaus", "register", space)
        p = np.random.default_rng(1).dirichlet(np.ones(policy.ansatz.dim))
        phi = policy.init_phi(np.random.default_rng(0))
        got = policy.distribution(p, phi).cats[0][0]
        register = (np.arange(policy.ansatz.dim) >> 4) & 3  # after 4 feature qubits
        mass = np.array([p[register == c].sum() for c in range(3)]) + 1e-3
        assert got == pytest.approx(mass / mass.sum())

    def test_head_probabilities_are_a_softmax_of_the_features(self) -> None:
        space = {"c": CategoricalDistribution(["a", "b", "c"])}
        policy = _policy("gaus", "head", space)
        rng = np.random.default_rng(2)
        phi = policy.init_phi(rng)
        phi["Wc"] = rng.normal(size=phi["Wc"].shape)
        p = rng.dirichlet(np.ones(policy.ansatz.dim))
        z = policy._n_mat[0] @ p
        logits = phi["Wc"] @ z + phi["bc"]
        expected = np.exp(logits) / np.exp(logits).sum()
        assert policy.distribution(p, phi).cats[0][0] == pytest.approx(expected)

    def test_head_starts_uniform(self) -> None:
        space = {"c": CategoricalDistribution(["a", "b", "c", "d"])}
        policy = _policy("mixture", "head", space)
        p = np.random.default_rng(3).dirichlet(np.ones(policy.ansatz.dim))
        dist = policy.distribution(p, policy.init_phi(np.random.default_rng(0)))
        assert np.allclose(dist.cats[0], 0.25)

    def test_a_single_choice_always_has_probability_one(self) -> None:
        space = {"c": CategoricalDistribution(["only"]), "x": FloatDistribution(0, 1)}
        for encoding in ("register", "head"):
            policy = _policy("gaus", encoding, space)
            p = np.random.default_rng(4).dirichlet(np.ones(policy.ansatz.dim))
            dist = policy.distribution(p, policy.init_phi(np.random.default_rng(0)))
            assert dist.cats[0].tolist() == [[1.0]]


class TestMixtureReadOut:
    def test_weights_are_the_mode_register_probabilities(self) -> None:
        policy = _policy("mixture", "head", CONTINUOUS)
        p = np.random.default_rng(1).dirichlet(np.ones(policy.ansatz.dim))
        w = policy.distribution(p, policy.init_phi(np.random.default_rng(0))).weights
        expected = [p[np.arange(policy.ansatz.dim) % 4 == k].sum() for k in range(4)]
        assert np.allclose(w, expected)

    def test_a_rare_mode_does_not_blow_up(self) -> None:
        policy = _policy("mixture", "head", CONTINUOUS)
        p = np.zeros(policy.ansatz.dim)
        p[0] = 1.0  # all mass on mode 0
        dist = policy.distribution(p, policy.init_phi(np.random.default_rng(0)))
        assert np.all(np.isfinite(dist.means)) and np.all(np.isfinite(dist.sigmas))

    @pytest.mark.parametrize("seed", range(6))
    def test_initial_modes_are_roughly_balanced(self, seed: int) -> None:
        policy = _policy("mixture", "register", MIXED)
        rng = np.random.default_rng(seed)
        theta, phi = policy.init_theta(rng), policy.init_phi(rng)
        p = NumpyExecutor().probabilities(policy.ansatz, theta[None])[0]
        assert policy.distribution(p, phi).weights.min() > 0.05
