"""The mixed-variable mixture: bin masses, joint log-probabilities and gradients."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from intelloptim.algorithms.backend.quantum.commons.distributions import (
    DistState,
    bin_log_prob,
    entropy,
    entropy_grads,
    interval_mass,
    inverse_softplus,
    joint_log_prob,
    joint_log_prob_grads,
    log_prob_x,
    marginal_log_prob,
    sample_joint,
    softplus,
    to_u,
    to_x,
)
from intelloptim.algorithms.backend.quantum.commons.space import SpaceLayout, sigmoid

LOW, HIGH = np.array([-2.0, 0.0]), np.array([3.0, 10.0])

SPACE = {
    "a": FloatDistribution(-1.0, 1.0),
    "b": IntDistribution(1, 12),
    "c": IntDistribution(1, 40, log=True),
    "d": FloatDistribution(0.0, 1.0, step=0.1),
    "e": CategoricalDistribution(["u", "v", "w"]),
    "f": CategoricalDistribution([True, False]),
}


def _layout() -> SpaceLayout:
    return SpaceLayout.from_space(SPACE, "register")


def _dist(rng: np.random.Generator, k: int = 3) -> DistState:
    layout = _layout()
    d = len(layout.numeric)
    cats = [rng.dirichlet(np.ones(c.n_choices), size=k) for c in layout.categorical]
    return DistState(
        rng.dirichlet(np.ones(k)),
        rng.normal(size=(k, d)),
        rng.uniform(0.4, 1.5, (k, d)),
        cats,
    )


def _draws(dist: DistState, layout: SpaceLayout, rng: np.random.Generator, n: int):
    draws = [sample_joint(dist, layout, rng) for _ in range(n)]
    return (
        np.array([k for k, _, _ in draws]),
        np.array([a for _, a, _ in draws]),
        np.array([c for _, _, c in draws]),
    )


class TestBinMass:
    def test_masses_of_all_bins_sum_to_one(self) -> None:
        layout = _layout()
        rng = np.random.default_rng(0)
        for dim in layout.numeric:
            if not dim.discrete:
                continue
            for m, s in [(0.0, 1.0), (1.5, 0.3), (-2.0, 2.5)]:
                bins = np.arange(dim.n_bins)
                logp, _, _ = bin_log_prob(
                    dim, bins, np.full(dim.n_bins, m), np.full(dim.n_bins, s)
                )
                assert np.exp(logp).sum() == pytest.approx(1.0, abs=1e-9), (
                    dim.name,
                    m,
                    s,
                )
        assert rng is not None

    def test_masses_match_the_frequencies_of_sampled_latents(self) -> None:
        layout = _layout()
        dim = layout.numeric[1]  # b: Int(1, 12)
        m, s, n = 0.4, 0.9, 40000
        u = m + s * np.random.default_rng(1).standard_normal(n)
        freq = np.bincount(dim.bin_of(sigmoid(u)), minlength=dim.n_bins) / n
        logp, _, _ = bin_log_prob(
            dim, np.arange(dim.n_bins), np.full(dim.n_bins, m), np.full(dim.n_bins, s)
        )
        assert np.abs(np.exp(logp) - freq).max() < 0.01

    def test_a_single_bin_has_all_the_mass(self) -> None:
        dim = SpaceLayout.from_space({"x": IntDistribution(5, 5)}, "head").numeric[0]
        logp, d_m, d_s = bin_log_prob(
            dim, np.array([0]), np.array([2.0]), np.array([0.7])
        )
        assert logp[0] == pytest.approx(0.0) and d_m[0] == pytest.approx(0.0, abs=1e-12)
        assert d_s[0] == pytest.approx(0.0, abs=1e-12)

    def test_gradients_match_finite_differences(self) -> None:
        dim = _layout().numeric[2]  # c: log int
        bins = np.array([0, 3, 10, 25, 39])
        m = np.array([0.2, -0.5, 1.0, 2.0, -1.5])
        s = np.array([0.8, 1.2, 0.6, 1.0, 0.9])
        _, d_m, d_s = bin_log_prob(dim, bins, m, s)
        h = 1e-6
        num_m = (
            bin_log_prob(dim, bins, m + h, s)[0] - bin_log_prob(dim, bins, m - h, s)[0]
        ) / (2 * h)
        num_s = (
            bin_log_prob(dim, bins, m, s + h)[0] - bin_log_prob(dim, bins, m, s - h)[0]
        ) / (2 * h)
        assert np.allclose(d_m, num_m, atol=1e-5) and np.allclose(d_s, num_s, atol=1e-5)

    def test_far_tail_bins_are_floored_not_nan(self) -> None:
        dim = _layout().numeric[1]
        logp, d_m, d_s = bin_log_prob(
            dim, np.array([0]), np.array([60.0]), np.array([0.1])
        )
        assert np.isfinite(logp[0]) and d_m[0] == 0.0 and d_s[0] == 0.0

    def test_interval_mass_is_stable_in_the_upper_tail(self) -> None:
        mass = interval_mass(np.array([8.0]), np.array([8.5]))
        assert 0.0 < mass[0] < 1e-14


class TestJoint:
    def test_gradients_match_finite_differences(self) -> None:
        layout, rng = _layout(), np.random.default_rng(2)
        dist = _dist(rng)
        ks, acts, cats = _draws(dist, layout, rng, 12)
        coef = rng.normal(size=12)
        grads = joint_log_prob_grads(dist, layout, ks, acts, cats, coef)

        def f(d: DistState) -> float:
            return float((coef * joint_log_prob(d, layout, ks, acts, cats)).sum())

        def check(
            name: str, base: np.ndarray, grad: np.ndarray, build: Callable
        ) -> None:
            for idx in np.ndindex(grad.shape):
                hi, lo = base.copy(), base.copy()
                hi[idx] += 1e-6
                lo[idx] -= 1e-6
                num = (f(build(hi)) - f(build(lo))) / 2e-6
                assert num == pytest.approx(grad[idx], abs=1e-5), (name, idx)

        check(
            "w",
            dist.weights,
            grads.weights,
            lambda x: DistState(x, dist.means, dist.sigmas, dist.cats),
        )
        check(
            "m",
            dist.means,
            grads.means,
            lambda x: DistState(dist.weights, x, dist.sigmas, dist.cats),
        )
        check(
            "s",
            dist.sigmas,
            grads.sigmas,
            lambda x: DistState(dist.weights, dist.means, x, dist.cats),
        )
        for i, probs in enumerate(dist.cats):

            def build(x: np.ndarray, i: int = i) -> DistState:
                cats_ = list(dist.cats)
                cats_[i] = x
                return DistState(dist.weights, dist.means, dist.sigmas, cats_)

            check(f"cat{i}", probs, grads.cats[i], build)

    def test_probabilities_of_all_discrete_actions_sum_to_one(self) -> None:
        # Enumerate every (k, int bin, category) combination of a tiny space.
        space = {
            "n": IntDistribution(1, 4),
            "c": CategoricalDistribution(["a", "b", "c"]),
        }
        layout = SpaceLayout.from_space(space, "register")
        rng = np.random.default_rng(3)
        dist = DistState(
            rng.dirichlet(np.ones(2)),
            rng.normal(size=(2, 1)),
            rng.uniform(0.5, 1.5, (2, 1)),
            [rng.dirichlet(np.ones(3), size=2)],
        )
        ks = np.repeat(np.arange(2), 12)
        bins = np.tile(np.repeat(np.arange(4), 3), 2)[:, None].astype(float)
        cats = np.tile(np.arange(3), 8)[:, None]
        total = np.exp(joint_log_prob(dist, layout, ks, bins, cats)).sum()
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_sampling_is_reproducible_and_in_range(self) -> None:
        layout, dist = _layout(), _dist(np.random.default_rng(4))
        a = sample_joint(dist, layout, np.random.default_rng(9))
        b = sample_joint(dist, layout, np.random.default_rng(9))
        assert (
            a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
        )
        _, acts, cats = _draws(dist, layout, np.random.default_rng(5), 300)
        for i, dim in enumerate(layout.numeric):
            if dim.discrete:
                assert acts[:, i].min() >= 0 and acts[:, i].max() <= dim.n_bins - 1
        assert cats[:, 0].max() <= 2 and cats[:, 1].max() <= 1

    def test_sampled_frequencies_follow_the_distribution(self) -> None:
        layout = SpaceLayout.from_space(
            {"c": CategoricalDistribution(["a", "b", "c"])}, "head"
        )
        dist = DistState(
            np.array([0.25, 0.75]),
            np.zeros((2, 0)),
            np.ones((2, 0)),
            [np.array([[1.0, 0.0, 0.0], [0.1, 0.2, 0.7]])],
        )
        ks, _, cats = _draws(dist, layout, np.random.default_rng(6), 8000)
        assert np.mean(ks == 1) == pytest.approx(0.75, abs=0.03)
        assert np.all(cats[ks == 0, 0] == 0)
        assert np.mean(cats[ks == 1, 0] == 2) == pytest.approx(0.7, abs=0.03)

    def test_a_space_with_only_categoricals_has_no_numeric_dims(self) -> None:
        layout = SpaceLayout.from_space(
            {"c": CategoricalDistribution(["a", "b"])}, "head"
        )
        dist = DistState(
            np.ones(1), np.zeros((1, 0)), np.ones((1, 0)), [np.array([[0.3, 0.7]])]
        )
        k, acts, cats = sample_joint(dist, layout, np.random.default_rng(0))
        assert acts.shape == (0,)
        logp = joint_log_prob(dist, layout, np.array([k]), acts[None, :], cats[None, :])
        assert logp[0] == pytest.approx(np.log([0.3, 0.7][cats[0]]))

    def test_json_round_trip(self) -> None:
        dist = _dist(np.random.default_rng(7))
        back = DistState.from_json(dist.to_json())
        assert np.array_equal(back.means, dist.means)
        assert all(
            np.array_equal(a, b) for a, b in zip(back.cats, dist.cats, strict=True)
        )

    def test_json_round_trip_without_numeric_dims(self) -> None:
        dist = DistState(
            np.ones(2) / 2, np.zeros((2, 0)), np.ones((2, 0)), [np.full((2, 2), 0.5)]
        )
        back = DistState.from_json(dist.to_json())
        assert back.means.shape == (2, 0) and back.sigmas.shape == (2, 0)


class TestEntropy:
    def test_gradients_match_finite_differences(self) -> None:
        dist = _dist(np.random.default_rng(8))
        grads = entropy_grads(dist)
        for idx in np.ndindex(dist.sigmas.shape):
            hi, lo = dist.sigmas.copy(), dist.sigmas.copy()
            hi[idx] += 1e-6
            lo[idx] -= 1e-6
            num = (
                entropy(DistState(dist.weights, dist.means, hi, dist.cats))
                - entropy(DistState(dist.weights, dist.means, lo, dist.cats))
            ) / 2e-6
            assert num == pytest.approx(grads.sigmas[idx], abs=1e-5)
        probs = dist.cats[0]
        idx = (1, 2)
        hi, lo = probs.copy(), probs.copy()
        hi[idx] += 1e-6
        lo[idx] -= 1e-6
        num = (
            entropy(
                DistState(dist.weights, dist.means, dist.sigmas, [hi, *dist.cats[1:]])
            )
            - entropy(
                DistState(dist.weights, dist.means, dist.sigmas, [lo, *dist.cats[1:]])
            )
        ) / 2e-6
        assert num == pytest.approx(grads.cats[0][idx], abs=1e-5)


class TestContinuousHelpers:
    def test_density_on_the_box_integrates_to_one(self) -> None:
        dist = DistState(
            np.array([0.3, 0.7]), np.array([[-1.0], [0.8]]), np.array([[0.6], [1.1]])
        )
        low, high = np.array([2.0]), np.array([5.0])
        grid = np.linspace(2.0, 5.0, 200001)[1:-1, None]
        density = np.exp(log_prob_x(dist, grid, low, high))
        assert np.trapezoid(density, grid[:, 0]) == pytest.approx(1.0, abs=1e-4)

    def test_transform_round_trip_and_marginal(self) -> None:
        u = np.array([[-3.0, 0.5], [2.0, -1.0]])
        assert np.allclose(to_u(to_x(u, LOW, HIGH), LOW, HIGH), u)
        dist = DistState(np.array([0.5, 0.5]), np.zeros((2, 2)), np.ones((2, 2)))
        assert marginal_log_prob(dist, u).shape == (2,)

    def test_softplus_helpers(self) -> None:
        assert softplus(np.array([0.0]))[0] == pytest.approx(np.log(2.0))
        assert softplus(np.array([800.0]))[0] == pytest.approx(800.0)
        assert softplus(np.array([inverse_softplus(0.7)]))[0] == pytest.approx(0.7)
