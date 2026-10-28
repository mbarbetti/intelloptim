"""One PPO update: direction of the step, trust region and bookkeeping."""

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
    sample_joint,
)
from intelloptim.algorithms.backend.quantum.commons.learner import (
    Cohort,
    initial_snapshot,
    update,
)
from intelloptim.algorithms.backend.quantum.commons.policy import QuantumPolicy
from intelloptim.algorithms.backend.quantum.commons.ppo import (
    advantages,
    clipped_surrogate,
    kl_estimate,
)
from intelloptim.algorithms.backend.quantum.commons.resources import (
    QuantumBudgetExceededError,
    ResourceLedger,
)
from intelloptim.algorithms.backend.quantum.commons.space import SpaceLayout
from tests.support.numpy_executor import NumpyExecutor

SPACE: Dict[str, BaseDistribution] = {
    "a": FloatDistribution(-1.0, 1.0),
    "b": IntDistribution(1, 8),
    "c": CategoricalDistribution(["u", "v", "w"]),
}


def _config(algorithm: str, **kw: object) -> QuantumPolicyConfig:
    if algorithm == "gaus":
        return QuantumPolicyConfig(algorithm="gaus", n_feature_qubits=4, **kw)  # type: ignore[arg-type]
    return QuantumPolicyConfig(
        algorithm="mixture", n_mode_qubits=2, n_feature_qubits=3, **kw  # type: ignore[arg-type]
    )


def _start(config: QuantumPolicyConfig):
    policy = QuantumPolicy(
        config, SpaceLayout.from_space(SPACE, config.categorical_encoding)
    )
    snapshot = initial_snapshot(
        config, policy, NumpyExecutor(), np.random.default_rng(0)
    )
    return policy, snapshot


def _cohort(policy: QuantumPolicy, snapshot, size: int, seed: int) -> Cohort:
    rng = np.random.default_rng(seed)
    draws = [sample_joint(snapshot.dist, policy.layout, rng) for _ in range(size)]
    ks = np.array([k for k, _, _ in draws])
    acts = np.array([a for _, a, _ in draws])
    cats = np.array([c for _, _, c in draws])
    # Better when the float is high, the int is high and the category is "w".
    rewards = acts[:, 0] + acts[:, 1] / 4.0 + (cats[:, 0] == 2)
    return Cohort(ks, acts, cats, rewards)


def _gain(config, policy, snapshot, cohort) -> float:
    """Surrogate value of the updated policy against the behaviour policy."""
    new, _ = update(
        config, policy, NumpyExecutor(ledger=ResourceLedger()), snapshot, cohort
    )
    layout = policy.layout
    logp_old = joint_log_prob(
        snapshot.dist, layout, cohort.ks, cohort.actions, cohort.cats
    )
    logp_new = joint_log_prob(new.dist, layout, cohort.ks, cohort.actions, cohort.cats)
    adv = advantages(cohort.rewards, config.advantage)
    return clipped_surrogate(logp_new, logp_old, adv, config.clip_eps)[0]


@pytest.mark.parametrize("encoding", ["register", "head"])
@pytest.mark.parametrize("algorithm", ["gaus", "mixture"])
@pytest.mark.parametrize("seed", range(2))
def test_the_circuit_angles_alone_move_the_policy_towards_better_trials(
    algorithm: str, encoding: str, seed: int
) -> None:
    # The head is frozen (tiny learning rate), so any gain comes from theta: a
    # wrong sign in the parameter-shift chain would make the gain negative.
    config = _config(
        algorithm,
        categorical_encoding=encoding,
        lr_phi=1e-12,
        lr_theta=0.05,
        ppo_epochs=2,
        kl_stop=10.0,
    )
    policy, snapshot = _start(config)
    assert _gain(config, policy, snapshot, _cohort(policy, snapshot, 40, seed)) > 0.0


@pytest.mark.parametrize("encoding", ["register", "head"])
@pytest.mark.parametrize("algorithm", ["gaus", "mixture"])
def test_the_head_alone_also_improves_the_surrogate(
    algorithm: str, encoding: str
) -> None:
    config = _config(
        algorithm,
        categorical_encoding=encoding,
        lr_theta=1e-12,
        lr_phi=0.02,
        ppo_epochs=2,
        kl_stop=10.0,
    )
    policy, snapshot = _start(config)
    assert _gain(config, policy, snapshot, _cohort(policy, snapshot, 40, 1)) > 0.0


def test_the_update_stays_inside_the_trust_region() -> None:
    config = _config("gaus", lr_theta=0.3, lr_phi=0.3, kl_stop=0.02)
    policy, snapshot = _start(config)
    cohort = _cohort(policy, snapshot, 10, 0)
    new, summary = update(config, policy, NumpyExecutor(), snapshot, cohort)
    layout = policy.layout
    kl = kl_estimate(
        joint_log_prob(new.dist, layout, cohort.ks, cohort.actions, cohort.cats),
        joint_log_prob(snapshot.dist, layout, cohort.ks, cohort.actions, cohort.cats),
    )
    assert kl <= 0.02 + 1e-12
    assert summary["stopped_early"] is True and summary["backtracks"] >= 1


def test_publishes_the_next_version_and_keeps_the_layout() -> None:
    config = _config("mixture")
    policy, snapshot = _start(config)
    new, summary = update(
        config, policy, NumpyExecutor(), snapshot, _cohort(policy, snapshot, 10, 0)
    )
    assert (new.version, new.n_updates) == (1, 1)
    assert (snapshot.version, snapshot.n_updates) == (0, 0)  # the input is untouched
    assert new.digest == snapshot.digest and new.names == ["a", "b", "c"]
    assert not np.array_equal(new.theta, snapshot.theta)
    assert new.dist.weights.sum() == pytest.approx(1.0)
    assert len(new.dist.cats) == 1 and new.dist.cats[0].shape == (4, 3)
    assert summary["cohort_size"] == 10 and summary["version"] == 1


def test_optimizer_state_is_carried_to_the_next_update() -> None:
    config = _config("gaus")
    policy, snapshot = _start(config)
    first, _ = update(
        config, policy, NumpyExecutor(), snapshot, _cohort(policy, snapshot, 10, 0)
    )
    assert first.adam_theta["t"] >= 1 and first.adam_phi["t"] >= 1
    second, _ = update(
        config, policy, NumpyExecutor(), first, _cohort(policy, first, 10, 1)
    )
    assert second.adam_theta["t"] > first.adam_theta["t"]


def test_circuit_cost_is_two_p_plus_one_per_epoch() -> None:
    config = _config("gaus", ppo_epochs=2, kl_stop=10.0)
    policy, snapshot = _start(config)
    executor = NumpyExecutor()
    update(config, policy, executor, snapshot, _cohort(policy, snapshot, 10, 0))
    p = policy.ansatz.n_params
    assert (
        executor.ledger.circuits == 2 * (2 * p + 1) + 1
    )  # two gradients + the final policy


def test_budget_is_checked_before_any_circuit_runs() -> None:
    config = _config("gaus", ppo_epochs=2)
    policy, snapshot = _start(config)
    ledger = ResourceLedger(max_shots=1000)
    executor = NumpyExecutor(100, np.random.default_rng(0), ledger)
    with pytest.raises(QuantumBudgetExceededError):
        update(config, policy, executor, snapshot, _cohort(policy, snapshot, 10, 0))
    assert ledger.circuits == 0 and ledger.shots == 0


def test_identical_rewards_leave_the_policy_unchanged() -> None:
    config = _config("gaus")
    policy, snapshot = _start(config)
    cohort = _cohort(policy, snapshot, 10, 0)
    flat = Cohort(cohort.ks, cohort.actions, cohort.cats, np.zeros(10))
    new, _ = update(config, policy, NumpyExecutor(), snapshot, flat)
    assert np.allclose(new.theta, snapshot.theta)
    assert np.allclose(new.dist.means, snapshot.dist.means)


def test_entropy_bonus_is_supported_with_categoricals() -> None:
    config = _config("mixture", entropy_coef=0.1, categorical_encoding="head")
    policy, snapshot = _start(config)
    new, _ = update(
        config, policy, NumpyExecutor(), snapshot, _cohort(policy, snapshot, 10, 0)
    )
    assert np.all(np.isfinite(new.dist.cats[0]))
