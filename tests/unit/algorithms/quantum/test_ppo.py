"""PPO ingredients."""

from __future__ import annotations

import numpy as np
import pytest

from intelloptim.algorithms.backend.quantum.commons.ppo import (
    Adam,
    advantages,
    clipped_surrogate,
    kl_estimate,
)


class TestAdvantages:
    def test_mean_baseline_is_centred(self) -> None:
        a = advantages(np.array([1.0, 2.0, 6.0]), "mean")
        assert a.sum() == pytest.approx(0.0)
        assert a[2] > 0 > a[0]

    def test_standardize_has_unit_scale(self) -> None:
        a = advantages(np.array([1.0, 2.0, 6.0, 9.0]), "standardize")
        assert a.std() == pytest.approx(1.0)

    def test_rank_ignores_outliers(self) -> None:
        a = advantages(np.array([1.0, 2.0, 3.0]), "rank")
        b = advantages(np.array([1.0, 2.0, 1e9]), "rank")
        assert np.allclose(a, b)

    def test_rank_ties_share_a_rank(self) -> None:
        a = advantages(np.array([1.0, 5.0, 5.0, 9.0]), "rank")
        assert a[1] == a[2]
        assert a[0] < a[1] < a[3]
        assert a.sum() == pytest.approx(0.0)

    @pytest.mark.parametrize("mode", ["mean", "standardize", "rank"])
    def test_identical_rewards_give_zero_advantage(self, mode: str) -> None:
        assert np.all(advantages(np.full(5, 3.0), mode) == 0.0)

    def test_unknown_mode(self) -> None:
        with pytest.raises(ValueError):
            advantages(np.ones(3), "nope")


class TestClippedSurrogate:
    EPS = 0.2

    @staticmethod
    def _grad(logp_new: float, adv: float) -> float:
        _, g = clipped_surrogate(
            np.array([logp_new]),
            np.array([0.0]),
            np.array([adv]),
            TestClippedSurrogate.EPS,
        )
        return float(g[0])

    def test_unclipped_inside_the_range(self) -> None:
        assert self._grad(0.05, 1.0) == pytest.approx(np.exp(0.05))

    def test_positive_advantage_above_the_range_is_clipped(self) -> None:
        assert self._grad(np.log(1.5), 1.0) == 0.0

    def test_positive_advantage_below_the_range_is_not_clipped(self) -> None:
        assert self._grad(np.log(0.5), 1.0) != 0.0

    def test_negative_advantage_below_the_range_is_clipped(self) -> None:
        assert self._grad(np.log(0.5), -1.0) == 0.0

    def test_negative_advantage_above_the_range_is_not_clipped(self) -> None:
        assert self._grad(np.log(1.5), -1.0) != 0.0

    def test_gradient_matches_finite_differences_of_the_value(self) -> None:
        rng = np.random.default_rng(0)
        logp_old = rng.normal(size=6)
        logp_new = logp_old + rng.normal(0, 0.3, 6)
        adv = rng.normal(size=6)
        value, grad = clipped_surrogate(logp_new, logp_old, adv, self.EPS)
        for i in range(6):
            hi, lo = logp_new.copy(), logp_new.copy()
            hi[i] += 1e-6
            lo[i] -= 1e-6
            num = (
                clipped_surrogate(hi, logp_old, adv, self.EPS)[0]
                - clipped_surrogate(lo, logp_old, adv, self.EPS)[0]
            ) / 2e-6
            assert num == pytest.approx(grad[i], abs=1e-6)
        assert value == pytest.approx(
            np.minimum(
                np.exp(logp_new - logp_old) * adv,
                np.clip(np.exp(logp_new - logp_old), 0.8, 1.2) * adv,
            ).mean()
        )

    def test_equal_policies_give_ratio_one(self) -> None:
        logp = np.array([-1.0, -2.0])
        adv = np.array([1.0, -1.0])
        value, grad = clipped_surrogate(logp, logp, adv, self.EPS)
        assert value == pytest.approx(0.0)
        assert np.allclose(grad, adv / 2)


def test_kl_estimate_is_zero_for_equal_policies_and_positive_otherwise() -> None:
    logp = np.array([-1.0, -2.0, -0.5])
    assert kl_estimate(logp, logp) == 0.0
    assert kl_estimate(logp + np.array([0.3, -0.2, 0.1]), logp) > 0.0


class TestAdam:
    def test_minimises_a_quadratic(self) -> None:
        adam = Adam(lr=0.1)
        params = {"x": np.array([3.0, -2.0])}
        for _ in range(400):
            params = adam.step(params, {"x": 2 * params["x"]})
        assert np.allclose(params["x"], 0.0, atol=1e-2)

    def test_state_round_trip_continues_identically(self) -> None:
        a, b = Adam(lr=0.1), Adam(lr=0.1)
        pa = {"x": np.array([1.0])}
        for _ in range(3):
            pa = a.step(pa, {"x": 2 * pa["x"]})
        b.load_json(a.to_json())
        pb = {"x": pa["x"].copy()}
        assert np.allclose(
            a.step(pa, {"x": 2 * pa["x"]})["x"], b.step(pb, {"x": 2 * pb["x"]})["x"]
        )
