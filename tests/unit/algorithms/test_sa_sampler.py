"""Unit tests for the simulated annealing sampler and its spec."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
import pytest
from optuna import distributions
from optuna.trial import create_trial
from pydantic import ValidationError

from intelloptim.algorithms.backend import SASampler
from intelloptim.algorithms.backend.builtin._common import completed_trials
from intelloptim.algorithms.samplers import resolve_sampler
from intelloptim.core.samplers import SASamplerSpec

optuna.logging.set_verbosity(optuna.logging.WARNING)

_FLOAT = distributions.FloatDistribution(-10.0, 10.0)
_SPACE: Dict[str, distributions.BaseDistribution] = {
    "x": _FLOAT,
    "y": _FLOAT,
}


def _quadratic(params: Dict[str, Any]) -> float:
    return float(params["x"] ** 2 + params["y"] ** 2)


def _run(
    storage: optuna.storages.BaseStorage,
    name: str,
    make_sampler: Callable[[], SASampler],
    n_trials: int,
    space: Optional[Dict[str, distributions.BaseDistribution]] = None,
    objective: Callable[[Dict[str, Any]], float] = _quadratic,
    direction: str = "minimize",
    fresh_each_ask: bool = True,
) -> List[Dict[str, Any]]:
    """Mimic the orchestrator: load the study, ask, tell."""
    space = space or _SPACE
    optuna.create_study(
        study_name=name, storage=storage, direction=direction, load_if_exists=True
    )
    sampler = make_sampler()
    history: List[Dict[str, Any]] = []
    for _ in range(n_trials):
        if fresh_each_ask:
            sampler = make_sampler()
        study = optuna.load_study(study_name=name, storage=storage, sampler=sampler)
        trial = study.ask(fixed_distributions=space)
        history.append(dict(trial.params))
        study.tell(trial, objective(trial.params))
    return history


def _study_with_values(
    values: List[float], direction: str = "minimize"
) -> optuna.study.Study:
    study = optuna.create_study(direction=direction)
    for index, value in enumerate(values):
        study.add_trial(
            create_trial(
                params={"x": float(index), "y": 0.0},
                distributions=_SPACE,
                value=value,
            )
        )
    return study


class TestConstructorValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperature": 0.0},
            {"temperature": -1.0},
            {"temperature": float("inf")},
            {"temperature": float("nan")},
            {"cooldown_factor": 0.0},
            {"cooldown_factor": 1.5},
            {"neighbor_range_factor": 0.0},
            {"neighbor_range_factor": 2.0},
            {"n_startup_trials": -1},
        ],
    )
    def test_rejects_out_of_range(self, kwargs: Dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            SASampler(**kwargs)

    def test_accepts_boundaries(self) -> None:
        SASampler(cooldown_factor=1.0, neighbor_range_factor=1.0, n_startup_trials=0)


class TestStatelessness:
    def test_fresh_sampler_per_ask_matches_long_lived(self) -> None:
        def make() -> SASampler:
            return SASampler(seed=7, n_startup_trials=5)

        fresh = _run(
            optuna.storages.InMemoryStorage(), "s", make, 40, fresh_each_ask=True
        )
        lived = _run(
            optuna.storages.InMemoryStorage(), "s", make, 40, fresh_each_ask=False
        )
        assert fresh == lived

    def test_same_seed_same_history_same_suggestions(self) -> None:
        a = _run(optuna.storages.InMemoryStorage(), "s", lambda: SASampler(seed=1), 25)
        b = _run(optuna.storages.InMemoryStorage(), "s", lambda: SASampler(seed=1), 25)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        a = _run(optuna.storages.InMemoryStorage(), "s", lambda: SASampler(seed=1), 25)
        b = _run(optuna.storages.InMemoryStorage(), "s", lambda: SASampler(seed=2), 25)
        assert a != b

    def test_unseeded_is_deterministic_per_study_name(self) -> None:
        a = _run(optuna.storages.InMemoryStorage(), "alpha", SASampler, 25)
        b = _run(optuna.storages.InMemoryStorage(), "alpha", SASampler, 25)
        c = _run(optuna.storages.InMemoryStorage(), "beta", SASampler, 25)
        assert a == b
        assert a != c

    def test_warmup_trials_are_not_identical(self) -> None:
        history = _run(
            optuna.storages.InMemoryStorage(),
            "s",
            lambda: SASampler(seed=3, n_startup_trials=10),
            10,
            objective=lambda p: 0.0,
        )
        assert len({p["x"] for p in history}) == 10
        assert all(p["x"] != p["y"] for p in history)


class TestAnnealing:
    def test_temperature_decreases_and_respects_floor(self) -> None:
        sampler = SASampler(temperature=10.0, cooldown_factor=0.5)
        assert sampler._temperature_at(0) == 10.0
        assert sampler._temperature_at(2) == 2.5
        assert sampler._temperature_at(10_000) > 0.0

    def test_constant_temperature_with_unit_cooldown(self) -> None:
        sampler = SASampler(temperature=3.0, cooldown_factor=1.0)
        assert sampler._temperature_at(500) == 3.0

    def test_better_or_equal_is_always_accepted(self) -> None:
        accept = SASampler._acceptance_probability
        assert accept(5.0, 4.0, 1.0, maximize=False) == 1.0
        assert accept(5.0, 5.0, 1.0, maximize=False) == 1.0
        assert accept(5.0, 6.0, 1.0, maximize=True) == 1.0

    def test_worse_follows_metropolis_rule(self) -> None:
        accept = SASampler._acceptance_probability
        assert accept(1.0, 3.0, 2.0, maximize=False) == pytest.approx(math.exp(-1.0))
        assert accept(3.0, 1.0, 2.0, maximize=True) == pytest.approx(math.exp(-1.0))

    def test_acceptance_drops_as_temperature_cools(self) -> None:
        accept = SASampler._acceptance_probability
        hot = accept(0.0, 1.0, 10.0, maximize=False)
        cold = accept(0.0, 1.0, 0.1, maximize=False)
        assert cold < hot

    def test_infinite_values(self) -> None:
        accept = SASampler._acceptance_probability
        assert accept(1.0, float("inf"), 1.0, maximize=False) == 0.0
        assert accept(float("inf"), 1.0, 1.0, maximize=False) == 1.0
        assert accept(float("inf"), float("inf"), 1.0, maximize=False) == 1.0

    @pytest.mark.parametrize("direction", ["minimize", "maximize"])
    def test_frozen_chain_follows_best_point(self, direction: str) -> None:
        sampler = SASampler(
            temperature=1e-9, cooldown_factor=1.0, n_startup_trials=2, seed=0
        )
        values = [5.0, 3.0, 4.0, 1.0, 2.0, 9.0]
        study = _study_with_values(values, direction)
        completed = completed_trials(study)
        current, _ = sampler._replay(study, completed, direction == "maximize")
        # After the warm-up, only strictly better trials are accepted at a
        # near-zero temperature, so the chain ends on the overall best.
        assert current.value == (1.0 if direction == "minimize" else 9.0)


class TestWarmup:
    @pytest.mark.parametrize("direction", ["minimize", "maximize"])
    def test_chain_starts_from_best_warmup_trial(self, direction: str) -> None:
        sampler = SASampler(
            temperature=1e-9, cooldown_factor=1.0, n_startup_trials=3, seed=0
        )
        study = _study_with_values([4.0, 1.0, 7.0], direction)
        completed = completed_trials(study)
        current, _ = sampler._replay(study, completed, direction == "maximize")
        assert current.value == (1.0 if direction == "minimize" else 7.0)

    def test_no_relative_proposal_during_warmup(self) -> None:
        sampler = SASampler(n_startup_trials=5, seed=0)
        study = _study_with_values([1.0, 2.0, 3.0])
        study.sampler = sampler
        trial = study.ask()
        assert sampler.sample_relative(study, trial._cached_frozen_trial, _SPACE) == {}


class TestDistributions:
    def test_suggestions_respect_every_distribution(self) -> None:
        space: Dict[str, distributions.BaseDistribution] = {
            "f": distributions.FloatDistribution(-2.0, 2.0),
            "fl": distributions.FloatDistribution(1e-4, 1e2, log=True),
            "fs": distributions.FloatDistribution(0.0, 1.0, step=0.25),
            "i": distributions.IntDistribution(-5, 5),
            "il": distributions.IntDistribution(1, 1000, log=True),
            "is": distributions.IntDistribution(0, 20, step=5),
            "c": distributions.CategoricalDistribution(("a", "b", "c")),
        }

        def objective(p: Dict[str, Any]) -> float:
            return abs(p["f"]) + p["i"] ** 2 / 25.0

        history = _run(
            optuna.storages.InMemoryStorage(),
            "d",
            lambda: SASampler(seed=11, n_startup_trials=5, neighbor_range_factor=0.3),
            80,
            space=space,
            objective=objective,
        )
        for p in history:
            assert -2.0 <= p["f"] <= 2.0
            assert 1e-4 <= p["fl"] <= 1e2
            assert p["fs"] in (0.0, 0.25, 0.5, 0.75, 1.0)
            assert isinstance(p["i"], int) and -5 <= p["i"] <= 5
            assert isinstance(p["il"], int) and 1 <= p["il"] <= 1000
            assert p["is"] in (0, 5, 10, 15, 20)
            assert p["c"] in ("a", "b", "c")
        assert {p["c"] for p in history} == {"a", "b", "c"}

    def test_single_valued_parameter_is_left_to_independent_sampling(self) -> None:
        space: Dict[str, distributions.BaseDistribution] = {
            "x": _FLOAT,
            "y": distributions.FloatDistribution(1.0, 1.0),
        }
        history = _run(
            optuna.storages.InMemoryStorage(),
            "s",
            lambda: SASampler(seed=1, n_startup_trials=3),
            12,
            space=space,
            objective=lambda p: p["x"] ** 2,
        )
        assert all(p["y"] == 1.0 for p in history)

    def test_unsupported_distribution_raises(self) -> None:
        class Unknown(distributions.BaseDistribution):
            def to_external_repr(self, param_value_in_internal_repr: float) -> Any:
                return param_value_in_internal_repr

            def to_internal_repr(self, param_value_in_external_repr: Any) -> float:
                return float(param_value_in_external_repr)

            def single(self) -> bool:
                return False

            def _contains(self, param_value_in_internal_repr: float) -> bool:
                return True

        sampler = SASampler(seed=0)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Unsupported"):
            sampler._neighbor(Unknown(), 0.0, rng)


class TestQuality:
    @pytest.mark.parametrize("direction", ["minimize", "maximize"])
    def test_improves_on_random_search(self, direction: str) -> None:
        sign = 1.0 if direction == "minimize" else -1.0

        def objective(p: Dict[str, Any]) -> float:
            return sign * _quadratic(p)

        def best(history: List[Dict[str, Any]]) -> float:
            return min(_quadratic(p) for p in history)

        sa = _run(
            optuna.storages.InMemoryStorage(),
            "q",
            lambda: SASampler(
                temperature=10.0,
                cooldown_factor=0.95,
                neighbor_range_factor=0.1,
                n_startup_trials=10,
                seed=5,
            ),
            150,
            objective=objective,
            direction=direction,
        )
        assert best(sa) < 1.0


class TestErrors:
    def test_multi_objective_raises(self) -> None:
        study = optuna.create_study(
            directions=["minimize", "minimize"], sampler=SASampler(seed=0)
        )
        with pytest.raises(ValueError, match="single-objective"):
            study.ask(fixed_distributions=_SPACE)

    def test_non_finite_values_do_not_break_the_chain(self) -> None:
        sampler = SASampler(n_startup_trials=2, seed=0)
        study = _study_with_values([1.0, float("inf"), 2.0, float("inf"), 0.5])
        completed = completed_trials(study)
        current, temperature = sampler._replay(study, completed, False)
        assert current.value == 0.5
        assert math.isfinite(temperature) and temperature > 0


class TestSpecAndResolver:
    def test_resolver_forwards_all_fields(self) -> None:
        spec = SASamplerSpec(
            seed=3,
            n_startup_trials=4,
            temperature=2.0,
            cooldown_factor=0.8,
            neighbor_range_factor=0.2,
        )
        sampler = resolve_sampler(spec)
        assert isinstance(sampler, SASampler)
        assert sampler._n_startup_trials == 4
        assert sampler._temperature == 2.0
        assert sampler._cooldown_factor == 0.8
        assert sampler._neighbor_range_factor == 0.2
        assert sampler._seed == 3

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"temperature": 0.0},
            {"temperature": float("inf")},
            {"cooldown_factor": 0.0},
            {"cooldown_factor": 1.1},
            {"neighbor_range_factor": 0.0},
            {"neighbor_range_factor": 1.1},
            {"n_startup_trials": -1},
        ],
    )
    def test_spec_rejects_invalid_values(self, kwargs: Dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            SASamplerSpec(**kwargs)

    def test_spec_is_frozen_and_strict(self) -> None:
        spec = SASamplerSpec()
        with pytest.raises(ValidationError):
            SASamplerSpec(unknown=1)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            spec.seed = 1  # type: ignore[misc]
