"""Unit tests for the OWG sampler and its spec."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

import numpy as np
import optuna
import pytest
from optuna import distributions
from optuna.trial import create_trial
from pydantic import TypeAdapter, ValidationError

from intelloptim.algorithms.backend import OWGSampler as OWG
from intelloptim.algorithms.backend.builtin.owg import (
    _from_unit,
    _radical_inverse,
    _to_unit,
)
from intelloptim.algorithms.samplers import resolve_sampler
from intelloptim.core.samplers import (
    OWGSamplerSpec as Spec,
)
from intelloptim.core.samplers import SamplerKind, SamplerSpec

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("intelloptim").setLevel(logging.ERROR)

_MODEL_KEY = "intopt:owg"
_SPACE: Dict[str, distributions.BaseDistribution] = {
    "x": distributions.FloatDistribution(-10.0, 10.0),
    "y": distributions.FloatDistribution(-10.0, 10.0),
}


def _quadratic(params: Dict[str, Any]) -> float:
    return float((params["x"] - 3.0) ** 2 + (params["y"] + 2.0) ** 2)


def _run(
    storage: optuna.storages.BaseStorage,
    make_sampler: Callable[[], OWG],
    n_trials: int,
    space: Optional[Dict[str, distributions.BaseDistribution]] = None,
    objective: Callable[[Dict[str, Any]], float] = _quadratic,
    direction: str = "minimize",
    rebuild_every: Optional[int] = None,
    name: str = "s",
) -> List[Dict[str, Any]]:
    """Mimic the orchestrator: fresh sampler per ask, rebuild on a schedule."""
    space = space or _SPACE
    optuna.create_study(
        study_name=name, storage=storage, direction=direction, load_if_exists=True
    )
    history: List[Dict[str, Any]] = []
    for index in range(n_trials):
        study = optuna.load_study(
            study_name=name, storage=storage, sampler=make_sampler()
        )
        trial = study.ask(fixed_distributions=space)
        history.append(dict(trial.params))
        study.tell(trial, objective(trial.params))
        if rebuild_every and (index + 1) % rebuild_every == 0:
            plain = optuna.load_study(study_name=name, storage=storage)
            make_sampler().rebuild(plain)
    return history


def _study_with_points(
    points: List[Dict[str, float]],
    values: List[float],
    direction: str = "minimize",
) -> optuna.study.Study:
    study = optuna.create_study(direction=direction)
    for params, value in zip(points, values, strict=True):
        study.add_trial(create_trial(params=params, distributions=_SPACE, value=value))
    return study


class TestConstructorValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rebuild_every": 0},
            {"initial_design": "sobol"},
            {"sharpness": -1.0},
            {"sharpness": float("inf")},
            {"min_std": 0.0},
            {"min_std": 1.5},
            {"exploration_prob": -0.1},
            {"exploration_prob": 1.1},
            {"rebuild_delay_seconds": -1.0},
            {"rebuild_delay_seconds": float("nan")},
        ],
    )
    def test_rejects_out_of_range(self, kwargs: Dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            OWG(**kwargs)

    def test_accepts_boundaries(self) -> None:
        OWG(sharpness=0.0, min_std=1.0, exploration_prob=1.0, rebuild_every=1)


class TestUnitMapping:
    @pytest.mark.parametrize(
        "dist,value",
        [
            (distributions.FloatDistribution(-2.0, 6.0), 1.5),
            (distributions.FloatDistribution(1e-3, 1e3, log=True), 0.5),
            (distributions.IntDistribution(0, 10), 7),
            (distributions.IntDistribution(1, 100, log=True), 12),
            (distributions.IntDistribution(0, 20, step=5), 15),
        ],
    )
    def test_round_trip(self, dist: distributions.BaseDistribution, value: Any) -> None:
        assert _from_unit(dist, _to_unit(dist, value)) == pytest.approx(value)

    def test_extremes_stay_in_bounds(self) -> None:
        dist = distributions.IntDistribution(1, 5)
        assert _from_unit(dist, 0.0) == 1
        assert _from_unit(dist, 1.0) == 5
        assert _from_unit(dist, -3.0) == 1
        assert _from_unit(dist, 3.0) == 5

    def test_integers_are_equally_likely(self) -> None:
        dist = distributions.IntDistribution(0, 3)
        values = [_from_unit(dist, u) for u in (np.arange(4000) + 0.5) / 4000]
        counts = np.bincount(values)
        assert counts.min() == counts.max() == 1000

    def test_stepped_float_snaps_to_grid(self) -> None:
        dist = distributions.FloatDistribution(0.0, 1.0, step=0.25)
        assert _from_unit(dist, 0.62) == 0.5


class TestInitialDesign:
    def _points(
        self,
        design: str,
        n: int,
        seed: Optional[int] = 1,
        every: Optional[int] = None,
        name: str = "s",
    ) -> List[Dict[str, Any]]:
        space = {n_: distributions.FloatDistribution(0.0, 1.0) for n_ in "abc"}
        return _run(
            optuna.storages.InMemoryStorage(),
            lambda: OWG(rebuild_every=every or n, initial_design=design, seed=seed),
            n,
            space=space,
            objective=lambda p: 0.0,
            name=name,
        )

    def test_lhs_has_one_point_per_stratum_in_every_dimension(self) -> None:
        n = 16
        history = self._points("lhs", n)
        for name in "abc":
            strata = sorted(int(p[name] * n) for p in history)
            assert strata == list(range(n))

    def test_lhs_beyond_the_design_is_uniform_random(self) -> None:
        history = self._points("lhs", 30, every=10)
        tail = [p["a"] for p in history[10:]]
        assert all(0.0 <= v <= 1.0 for v in tail)
        assert len(set(tail)) == len(tail)

    def test_halton_matches_scrambled_radical_inverse(self) -> None:
        history = self._points("halton", 12)
        assert len({p["a"] for p in history}) == 12
        for name in "abc":
            assert all(0.0 <= p[name] < 1.0 for p in history)
        # Dimension 0 uses base 2: a shifted van der Corput sequence, so the
        # gaps between consecutive sorted points take at most 3 distinct sizes.
        ordered = np.sort([p["a"] for p in history])
        shifted = {round(float(g), 9) for g in np.diff(ordered)}
        assert len(shifted) <= 3

    def test_halton_dimensions_are_decorrelated(self) -> None:
        history = self._points("halton", 64)
        a = np.array([p["a"] for p in history])
        b = np.array([p["b"] for p in history])
        assert abs(np.corrcoef(a, b)[0, 1]) < 0.3

    @pytest.mark.parametrize("design", ["lhs", "halton"])
    def test_deterministic_for_seed(self, design: str) -> None:
        assert self._points(design, 10, seed=4) == self._points(design, 10, seed=4)
        assert self._points(design, 10, seed=4) != self._points(design, 10, seed=5)

    @pytest.mark.parametrize("design", ["lhs", "halton"])
    def test_unseeded_is_deterministic_per_study_name(self, design: str) -> None:
        a = self._points(design, 10, seed=None, name="alpha")
        b = self._points(design, 10, seed=None, name="alpha")
        c = self._points(design, 10, seed=None, name="beta")
        assert a == b
        assert a != c

    def test_halton_beyond_32_dimensions_falls_back_to_uniform(self) -> None:
        space = {
            f"p{i:02d}": distributions.FloatDistribution(0.0, 1.0) for i in range(34)
        }
        history = _run(
            optuna.storages.InMemoryStorage(),
            lambda: OWG(rebuild_every=5, initial_design="halton", seed=0),
            3,
            space=space,
            objective=lambda p: 0.0,
        )
        assert all(len(p) == 34 for p in history)

    def test_radical_inverse(self) -> None:
        assert [_radical_inverse(i, 2) for i in (1, 2, 3, 4)] == [
            0.5,
            0.25,
            0.75,
            0.125,
        ]
        assert _radical_inverse(5, 3) == pytest.approx(2 / 3 + 1 / 9)


class TestRebuild:
    def test_writes_model_that_a_fresh_sampler_reads(self) -> None:
        storage = optuna.storages.InMemoryStorage()
        _run(storage, lambda: OWG(rebuild_every=10, seed=1), 10, rebuild_every=10)
        study = optuna.load_study(study_name="s", storage=storage)
        model = study.user_attrs[_MODEL_KEY]
        assert model["n_trials"] == 10
        assert model["numeric"] == ["x", "y"]
        space = {"x": _SPACE["x"], "y": _SPACE["y"]}
        assert OWG(seed=1)._load_model(study, space) is not None

    @pytest.mark.parametrize("direction", ["minimize", "maximize"])
    def test_mean_moves_towards_the_optimum(self, direction: str) -> None:
        sign = 1.0 if direction == "minimize" else -1.0
        storage = optuna.storages.InMemoryStorage()
        _run(
            storage,
            lambda: OWG(rebuild_every=20, seed=2, sharpness=5.0),
            20,
            objective=lambda p: sign * _quadratic(p),
            direction=direction,
            rebuild_every=20,
        )
        study = optuna.load_study(study_name="s", storage=storage)
        mean = study.user_attrs[_MODEL_KEY]["mean"]
        optimum = [(3.0 + 10) / 20, (-2.0 + 10) / 20]
        uniform_centre = [0.5, 0.5]
        assert np.linalg.norm(np.subtract(mean, optimum)) < np.linalg.norm(
            np.subtract(uniform_centre, optimum)
        )
        assert np.linalg.norm(np.subtract(mean, optimum)) < 0.12

    def test_zero_sharpness_gives_unweighted_moments(self) -> None:
        points = [{"x": float(i), "y": 0.0} for i in (-8, -2, 1, 4, 9)]
        values = [1.0, 5.0, 2.0, 9.0, 3.0]
        study = _study_with_points(points, values)
        OWG(sharpness=0.0, min_std=0.001).rebuild(study)
        model = study.user_attrs[_MODEL_KEY]
        units = np.array([(p["x"] + 10) / 20 for p in points])
        assert model["mean"][0] == pytest.approx(units.mean())
        assert model["cov"][0][0] == pytest.approx(units.var(ddof=1) + 0.001**2)

    def test_larger_sharpness_shrinks_the_covariance(self) -> None:
        rng = np.random.default_rng(0)
        points = [
            {"x": float(v), "y": float(w)} for v, w in rng.uniform(-9, 9, (40, 2))
        ]
        values = [_quadratic(p) for p in points]

        def variance(beta: float) -> float:
            study = _study_with_points(points, values)
            OWG(sharpness=beta).rebuild(study)
            return float(study.user_attrs[_MODEL_KEY]["cov"][0][0])

        assert variance(0.0) > variance(2.0) > variance(6.0)

    def test_covariance_has_a_floor(self) -> None:
        points = [{"x": 1.0, "y": 1.0}] * 4
        study = _study_with_points(points, [1.0, 2.0, 3.0, 4.0])
        OWG(min_std=0.05).rebuild(study)
        cov = np.array(study.user_attrs[_MODEL_KEY]["cov"])
        assert np.all(np.diag(cov) >= 0.05**2 - 1e-12)
        assert np.all(np.linalg.eigvalsh(cov) > 0)

    def test_categorical_histogram_favours_the_good_category(self) -> None:
        space = {
            "x": _SPACE["x"],
            "c": distributions.CategoricalDistribution(("good", "bad", "ugly")),
        }

        def objective(p: Dict[str, Any]) -> float:
            return abs(p["x"]) + {"good": 0.0, "bad": 5.0, "ugly": 9.0}[p["c"]]

        storage = optuna.storages.InMemoryStorage()
        _run(
            storage,
            lambda: OWG(rebuild_every=30, seed=3, sharpness=4.0),
            30,
            space=space,
            objective=objective,
            rebuild_every=30,
        )
        study = optuna.load_study(study_name="s", storage=storage)
        probs = study.user_attrs[_MODEL_KEY]["categorical"]["c"]
        assert probs[0] > probs[1] > probs[2] > 0.0
        assert sum(probs) == pytest.approx(1.0)

    def test_not_enough_data_is_a_no_op(self) -> None:
        study = optuna.create_study()
        assert OWG().rebuild(study) is None
        study = _study_with_points([{"x": 1.0, "y": 1.0}], [1.0])
        assert OWG().rebuild(study) is None
        assert _MODEL_KEY not in study.user_attrs

    def test_non_finite_values_are_ignored(self) -> None:
        points = [{"x": float(i), "y": 0.0} for i in range(4)]
        study = _study_with_points(points, [1.0, float("inf"), 2.0, float("-inf")])
        summary = OWG().rebuild(study)
        assert summary is not None and summary["n_trials_fit"] == 2

        study = _study_with_points(points[:2], [float("inf"), float("inf")])
        assert OWG().rebuild(study) is None

    def test_previous_model_is_kept_when_there_is_no_data(self) -> None:
        study = optuna.create_study()
        study.set_user_attr(_MODEL_KEY, {"version": 1, "marker": True})
        OWG().rebuild(study)
        assert study.user_attrs[_MODEL_KEY] == {"version": 1, "marker": True}

    def test_summary(self) -> None:
        points = [{"x": float(i), "y": float(-i)} for i in range(6)]
        study = _study_with_points(points, [float(i) for i in range(6)])
        summary = OWG(sharpness=0.0).rebuild(study)
        assert summary is not None
        assert summary["n_trials_fit"] == 6
        assert summary["n_eff"] == pytest.approx(6.0)

    def test_rebuild_delay_sleeps_before_storing(self) -> None:
        points = [{"x": float(i), "y": 0.0} for i in range(3)]
        study = _study_with_points(points, [1.0, 2.0, 3.0])
        stored_during_sleep: List[bool] = []
        with patch("intelloptim.algorithms.backend.builtin.owg.time.sleep") as sleep:
            sleep.side_effect = lambda _s: stored_during_sleep.append(
                _MODEL_KEY in study.user_attrs
            )
            OWG(rebuild_delay_seconds=2.5).rebuild(study)
        sleep.assert_called_once_with(2.5)
        assert stored_during_sleep == [False]
        assert _MODEL_KEY in study.user_attrs

    def test_no_sleep_without_delay(self) -> None:
        points = [{"x": float(i), "y": 0.0} for i in range(3)]
        study = _study_with_points(points, [1.0, 2.0, 3.0])
        with patch("intelloptim.algorithms.backend.builtin.owg.time.sleep") as sleep:
            OWG().rebuild(study)
        sleep.assert_not_called()

    def test_multi_objective_is_rejected(self) -> None:
        study = optuna.create_study(directions=["minimize", "minimize"])
        with pytest.raises(ValueError, match="single-objective"):
            OWG().rebuild(study)


class TestModelSampling:
    def _fitted(
        self, space: Optional[Dict[str, distributions.BaseDistribution]] = None
    ) -> optuna.study.Study:
        storage = optuna.storages.InMemoryStorage()
        _run(
            storage,
            lambda: OWG(rebuild_every=10, seed=1, sharpness=5.0),
            10,
            space=space,
            objective=lambda p: abs(p["x"]),
            rebuild_every=10,
        )
        return optuna.load_study(study_name="s", storage=storage)

    def test_suggestions_come_from_the_model(self) -> None:
        storage = optuna.storages.InMemoryStorage()
        history = _run(
            storage,
            lambda: OWG(rebuild_every=20, seed=1, sharpness=8.0, exploration_prob=0.0),
            60,
            rebuild_every=20,
        )
        before = np.std([p["x"] for p in history[:20]])
        after = np.std([p["x"] for p in history[40:]])
        assert after < before / 2

    def test_all_suggestions_respect_every_distribution(self) -> None:
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
            lambda: OWG(rebuild_every=10, seed=11),
            80,
            space=space,
            objective=objective,
            rebuild_every=10,
        )
        for p in history:
            assert -2.0 <= p["f"] <= 2.0
            assert 1e-4 <= p["fl"] <= 1e2
            assert p["fs"] in (0.0, 0.25, 0.5, 0.75, 1.0)
            assert isinstance(p["i"], int) and -5 <= p["i"] <= 5
            assert isinstance(p["il"], int) and 1 <= p["il"] <= 1000
            assert p["is"] in (0, 5, 10, 15, 20)
            assert p["c"] in ("a", "b", "c")

    def test_exploration_one_ignores_the_model(self) -> None:
        storage = optuna.storages.InMemoryStorage()
        history = _run(
            storage,
            lambda: OWG(rebuild_every=10, seed=1, sharpness=20.0, exploration_prob=1.0),
            50,
            rebuild_every=10,
        )
        late = [p["x"] for p in history[20:]]
        assert np.std(late) > 4.0

    def test_stale_between_rebuilds(self) -> None:
        storage = optuna.storages.InMemoryStorage()
        _run(storage, lambda: OWG(rebuild_every=10, seed=1), 10, rebuild_every=10)
        study = optuna.load_study(study_name="s", storage=storage)
        first = study.user_attrs[_MODEL_KEY]["n_trials"]
        _run(storage, lambda: OWG(rebuild_every=10, seed=1), 7)
        study = optuna.load_study(study_name="s", storage=storage)
        assert study.user_attrs[_MODEL_KEY]["n_trials"] == first == 10
        assert len(study.trials) == 17

    def test_mismatching_search_space_ignores_the_model(self) -> None:
        study = self._fitted()
        sampler = OWG(seed=1)
        other = {"x": _SPACE["x"], "z": _SPACE["y"]}
        assert sampler._load_model(study, other) is None
        narrower = {
            "x": distributions.FloatDistribution(-5.0, 5.0),
            "y": _SPACE["y"],
        }
        assert sampler._load_model(study, narrower) is None
        trial = study.ask()
        assert sampler.sample_relative(study, trial._cached_frozen_trial, other) == {}

    def test_unknown_model_version_is_ignored(self) -> None:
        study = self._fitted()
        model = dict(study.user_attrs[_MODEL_KEY])
        model["version"] = 999
        study.set_user_attr(_MODEL_KEY, model)
        assert OWG(seed=1)._load_model(study, dict(_SPACE)) is None

    def test_truncated_draw_stays_in_unit_cube(self) -> None:
        rng = np.random.default_rng(0)
        mean = np.array([0.98, 0.02])
        cov = np.eye(2) * 0.5
        draws = np.array([OWG._draw_truncated(mean, cov, rng) for _ in range(300)])
        assert draws.min() >= 0.0 and draws.max() <= 1.0

    def test_single_valued_parameter_is_left_to_independent_sampling(self) -> None:
        space: Dict[str, distributions.BaseDistribution] = {
            "x": _SPACE["x"],
            "k": distributions.FloatDistribution(1.0, 1.0),
        }
        history = _run(
            optuna.storages.InMemoryStorage(),
            lambda: OWG(rebuild_every=5, seed=1),
            20,
            space=space,
            objective=lambda p: p["x"] ** 2,
            rebuild_every=5,
        )
        assert all(p["k"] == 1.0 for p in history)

    def test_multi_objective_is_rejected_at_sampling(self) -> None:
        study = optuna.create_study(
            directions=["minimize", "minimize"], sampler=OWG(seed=0)
        )
        with pytest.raises(ValueError, match="single-objective"):
            study.ask(fixed_distributions=_SPACE)

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

        study = optuna.create_study(sampler=OWG(seed=0))
        trial = study.ask()
        with pytest.raises(ValueError, match="Unsupported"):
            OWG(seed=0).sample_independent(
                study, trial._cached_frozen_trial, "u", Unknown()
            )


class TestStatelessness:
    def test_fresh_sampler_per_ask_matches_long_lived(self) -> None:
        def make() -> OWG:
            return OWG(rebuild_every=8, seed=7)

        fresh = _run(optuna.storages.InMemoryStorage(), make, 30, rebuild_every=8)

        storage = optuna.storages.InMemoryStorage()
        optuna.create_study(study_name="s", storage=storage)
        sampler = make()
        lived: List[Dict[str, Any]] = []
        for index in range(30):
            study = optuna.load_study(study_name="s", storage=storage, sampler=sampler)
            trial = study.ask(fixed_distributions=_SPACE)
            lived.append(dict(trial.params))
            study.tell(trial, _quadratic(trial.params))
            if (index + 1) % 8 == 0:
                sampler.rebuild(optuna.load_study(study_name="s", storage=storage))
        assert fresh == lived


class TestEndToEnd:
    def test_beats_random_search_with_rebuilds(self) -> None:
        def best(history: List[Dict[str, Any]]) -> float:
            return min(_quadratic(p) for p in history)

        scores = []
        for seed in range(4):
            history = _run(
                optuna.storages.InMemoryStorage(),
                lambda seed=seed: OWG(rebuild_every=10, seed=seed),
                80,
                rebuild_every=10,
            )
            rng = np.random.default_rng(seed)
            random_best = min(
                _quadratic({"x": x, "y": y}) for x, y in rng.uniform(-10, 10, (80, 2))
            )
            scores.append((best(history), random_best))
        wins = sum(sa < rnd for sa, rnd in scores)
        assert wins >= 3

    def test_never_rebuilt_stays_on_the_initial_design(self) -> None:
        storage = optuna.storages.InMemoryStorage()
        _run(storage, lambda: OWG(rebuild_every=10, seed=1), 25)
        study = optuna.load_study(study_name="s", storage=storage)
        assert _MODEL_KEY not in study.user_attrs


class TestSpecAndResolver:
    def test_defaults_and_kind(self) -> None:
        spec = Spec()
        assert spec.kind == SamplerKind.OWG
        assert spec.rebuild_every == 20
        assert spec.initial_design == "lhs"

    def test_is_remote_and_rebuildable(self) -> None:
        assert Spec.runs_remote is True
        assert Spec.rebuild_policy is not None
        assert Spec.rebuild_policy.every_n_tells is not None

    def test_resolver_forwards_all_fields(self) -> None:
        spec = Spec(
            seed=3,
            rebuild_every=7,
            initial_design="halton",
            sharpness=2.0,
            min_std=0.05,
            exploration_prob=0.2,
            rebuild_delay_seconds=1.5,
        )
        sampler = resolve_sampler(spec)
        assert isinstance(sampler, OWG)
        assert sampler._seed == 3
        assert sampler._rebuild_every == 7
        assert sampler._initial_design == "halton"
        assert sampler._sharpness == 2.0
        assert sampler._min_std == 0.05
        assert sampler._exploration_prob == 0.2
        assert sampler._rebuild_delay_seconds == 1.5

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rebuild_every": 0},
            {"initial_design": "sobol"},
            {"sharpness": -0.1},
            {"sharpness": float("inf")},
            {"min_std": 0.0},
            {"min_std": 1.1},
            {"exploration_prob": 1.1},
            {"rebuild_delay_seconds": 61.0},
            {"rebuild_delay_seconds": -1.0},
        ],
    )
    def test_spec_rejects_invalid_values(self, kwargs: Dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Spec(**kwargs)

    def test_spec_is_frozen_and_strict(self) -> None:
        spec = Spec()
        with pytest.raises(ValidationError):
            Spec(unknown=1)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            spec.seed = 1  # type: ignore[misc]

    def test_round_trips_through_the_sampler_union(self) -> None:
        adapter: TypeAdapter[Any] = TypeAdapter(SamplerSpec)
        spec = Spec(seed=1, rebuild_every=5, initial_design="halton")
        restored = adapter.validate_python(spec.model_dump(mode="json"))
        assert restored == spec
        assert isinstance(restored, Spec)
