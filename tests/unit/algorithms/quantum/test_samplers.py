"""The four quantum samplers on an in-memory Optuna study (numpy executor)."""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any, Callable, Dict, Type

import numpy as np
import optuna
import pytest
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from intelloptim.algorithms.backend.quantum import (
    QuantumGausSampler,
    QuantumMixtureSampler,
    SimQuantumGausSampler,
    SimQuantumMixtureSampler,
)
from intelloptim.algorithms.backend.quantum.commons.base import QuantumPolicySampler
from intelloptim.algorithms.backend.quantum.commons.config import QuantumPolicyConfig
from intelloptim.algorithms.backend.quantum.commons.executor import ALLOW_HARDWARE_ENV
from intelloptim.algorithms.backend.quantum.commons.resources import (
    QuantumBudgetExceededError,
    ResourceLedger,
)
from intelloptim.algorithms.backend.quantum.commons.snapshot import (
    BEHAVIOR_KEY,
    SNAPSHOT_KEY,
    Snapshot,
)
from intelloptim.algorithms.space import store_declared_space
from tests.support.numpy_executor import NumpyExecutor

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("intelloptim").setLevel(logging.ERROR)

FLOATS: Dict[str, BaseDistribution] = {
    f"x{i}": FloatDistribution(-5.0, 5.0) for i in range(4)
}
MIXED: Dict[str, BaseDistribution] = {
    "lr": FloatDistribution(1e-4, 1.0, log=True),
    "x": FloatDistribution(-3.0, 3.0),
    "n": IntDistribution(1, 32, log=True),
    "k": IntDistribution(0, 10, step=2),
    "w": FloatDistribution(0.0, 1.0, step=0.25),
    "act": CategoricalDistribution(["relu", "tanh", "gelu"]),
    "flag": CategoricalDistribution([True, False, None, "x"]),
}


def _exact(rng: np.random.Generator, ledger: ResourceLedger) -> NumpyExecutor:
    return NumpyExecutor(None, None, ledger)


def _sampler(
    cls: Type[QuantumPolicySampler] = SimQuantumGausSampler, **overrides: Any
) -> QuantumPolicySampler:
    algorithm = "gaus" if cls.algorithm == "gaus" else "mixture"
    base: Dict[str, Any] = {"algorithm": algorithm, "seed": 3}
    if algorithm == "gaus":
        base["n_feature_qubits"] = 4
    else:
        base.update(n_mode_qubits=2, n_feature_qubits=3)
    base.update(overrides)
    return cls(QuantumPolicyConfig(**base), _exact, NumpyExecutor)


def _study(
    space: Dict[str, BaseDistribution], sampler: Any = None, **kw: Any
) -> optuna.Study:
    study = optuna.create_study(sampler=sampler, **kw)
    store_declared_space(study, space)
    return study


def _quadratic(params: Dict[str, Any]) -> float:
    return float(sum((v - 1.0) ** 2 for v in params.values()))


def _mixed_objective(p: Dict[str, Any]) -> float:
    return (
        (np.log10(p["lr"]) + 2) ** 2
        + (p["x"] - 1) ** 2
        + (np.log2(p["n"]) - 3) ** 2
        + (p["k"] - 6) ** 2 / 4
        + (p["w"] - 0.5) ** 2
        + {"relu": 1.0, "tanh": 0.0, "gelu": 0.5}[p["act"]]
        + (0.0 if p["flag"] is None else 1.0)
    )


def _run(
    study: optuna.Study,
    make: Callable[[], QuantumPolicySampler],
    space: Dict[str, BaseDistribution],
    n_trials: int,
    objective: Callable[[Dict[str, Any]], float] = _quadratic,
    rebuild_every: int = 10,
) -> None:
    """Ask/tell with a *fresh sampler per trial*, as the server does."""
    for i in range(n_trials):
        study.sampler = make()
        trial = study.ask(fixed_distributions=space)
        study.tell(trial, objective(trial.params))
        if (i + 1) % rebuild_every == 0:
            make().rebuild(study)


def _best(study: optuna.Study) -> float:
    return min(t.value for t in study.trials if t.value is not None)


ALL_SIM = [SimQuantumGausSampler, SimQuantumMixtureSampler]


def test_unit_tests_never_import_qiskit() -> None:
    code = (
        "import sys, intelloptim, intelloptim.algorithms.backend.quantum as q;"
        "from intelloptim.algorithms.backend.quantum.commons import base, learner, policy;"
        "sys.exit(1 if 'qiskit' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


class TestAsk:
    @pytest.mark.parametrize("cls", ALL_SIM)
    def test_values_are_valid_and_typed(self, cls: Type[QuantumPolicySampler]) -> None:
        study = _study(MIXED, _sampler(cls))
        for _ in range(12):
            trial = study.ask(fixed_distributions=MIXED)
            p = trial.params
            assert 1e-4 <= p["lr"] <= 1.0 and -3.0 <= p["x"] <= 3.0
            assert isinstance(p["n"], int) and 1 <= p["n"] <= 32
            assert isinstance(p["k"], int) and p["k"] in (0, 2, 4, 6, 8, 10)
            assert p["w"] in (0.0, 0.25, 0.5, 0.75, 1.0)
            assert p["act"] in ("relu", "tanh", "gelu")
            assert p["flag"] in (True, False, None, "x")
            study.tell(trial, 0.0)

    def test_behavior_metadata_is_recorded(self) -> None:
        study = _study(MIXED, _sampler(SimQuantumMixtureSampler))
        for _ in range(6):
            study.tell(study.ask(fixed_distributions=MIXED), 0.0)
        for frozen in study.trials:
            b = frozen.system_attrs[BEHAVIOR_KEY]
            assert b["version"] == 0 and 0 <= b["mode"] < 4
            assert len(b["act"]) == 5 and len(b["cats"]) == 2
        assert BEHAVIOR_KEY == "intopt:qnt_policy"

    def test_asks_never_build_an_executor(self) -> None:
        calls = []

        def factory(rng: Any, ledger: ResourceLedger) -> NumpyExecutor:
            calls.append(1)
            return NumpyExecutor(None, None, ledger)

        sampler = SimQuantumGausSampler(
            QuantumPolicyConfig(algorithm="gaus", n_feature_qubits=4, seed=1),
            factory,
            NumpyExecutor,
        )
        study = _study(FLOATS, sampler)
        for _ in range(5):
            study.tell(study.ask(fixed_distributions=FLOATS), 1.0)
        assert calls == []

    def test_a_fresh_sampler_per_trial_gives_the_same_history(self) -> None:
        kept = _study(MIXED, _sampler())
        for _ in range(12):
            t = kept.ask(fixed_distributions=MIXED)
            kept.tell(t, _mixed_objective(t.params))
        fresh = _study(MIXED)
        _run(fresh, _sampler, MIXED, 12, _mixed_objective, rebuild_every=10**9)
        assert [t.params for t in kept.trials] == [t.params for t in fresh.trials]

    def test_same_seed_reproduces_and_a_new_seed_differs(self) -> None:
        def params(seed: int) -> Any:
            study = _study(MIXED, _sampler(seed=seed))
            return study.ask(fixed_distributions=MIXED).params

        assert params(1) == params(1) and params(1) != params(2)

    def test_without_a_declared_space_nothing_is_relative(self) -> None:
        study = optuna.create_study(sampler=_sampler())
        assert study.ask().params == {}

    def test_undeclared_parameters_are_drawn_independently_and_reproducibly(
        self,
    ) -> None:
        def draw() -> float:
            trial = _study(FLOATS, _sampler()).ask(fixed_distributions=FLOATS)
            return trial.suggest_float("extra", 0.0, 1.0)

        assert draw() == draw()

    def test_multi_objective_is_rejected(self) -> None:
        study = optuna.create_study(
            directions=["minimize", "minimize"], sampler=_sampler()
        )
        store_declared_space(study, FLOATS)
        with pytest.raises(ValueError, match="single-objective"):
            study.ask(fixed_distributions=FLOATS)

    def test_the_wrong_algorithm_for_the_class_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="runs the 'gaus' algorithm"):
            SimQuantumGausSampler(QuantumPolicyConfig(algorithm="mixture"), _exact)

    def test_a_space_that_needs_too_many_qubits_fails_clearly(self) -> None:
        space = {f"c{i}": CategoricalDistribution(list(range(16))) for i in range(5)}
        study = _study(space, _sampler())
        with pytest.raises(ValueError, match="at most 16 are supported"):
            study.ask(fixed_distributions=space)

    def test_all_categorical_space_works(self) -> None:
        space = {
            "a": CategoricalDistribution(["u", "v"]),
            "b": CategoricalDistribution([1, 2, 3]),
        }
        study = _study(space, None)
        _run(
            study,
            lambda: _sampler(),
            space,
            12,
            lambda p: float(p["b"]),
            rebuild_every=6,
        )
        assert Snapshot.from_json(study.user_attrs[SNAPSHOT_KEY]).version == 2


class TestRebuild:
    def test_publishes_a_new_version_and_resources(self) -> None:
        study = _study(MIXED)
        _run(study, _sampler, MIXED, 10, _mixed_objective)
        stored = Snapshot.from_json(study.user_attrs[SNAPSHOT_KEY])
        assert stored.version == 1 and stored.n_updates == 1
        assert stored.names == sorted(MIXED) and SNAPSHOT_KEY == "intopt:qnt_policy"
        assert stored.resources["circuits"] >= 37 and stored.resources["shots"] == 0

    def test_later_trials_use_the_new_version(self) -> None:
        study = _study(MIXED)
        _run(study, _sampler, MIXED, 14, _mixed_objective)
        versions = [t.system_attrs[BEHAVIOR_KEY]["version"] for t in study.trials]
        assert versions == [0] * 10 + [1] * 4

    def test_small_cohort_is_left_to_grow(self) -> None:
        study = _study(FLOATS)
        _run(study, lambda: _sampler(min_cohort=6), FLOATS, 3, rebuild_every=3)
        assert SNAPSHOT_KEY not in study.user_attrs
        assert _sampler(min_cohort=6).rebuild(study) is None

    def test_trials_of_older_versions_are_not_reused(self) -> None:
        study = _study(FLOATS, _sampler())
        early = [study.ask(fixed_distributions=FLOATS) for _ in range(12)]
        for t in early[:10]:
            study.tell(t, _quadratic(t.params))
        first = _sampler().rebuild(study)
        assert first is not None and first["cohort_size"] == 10
        for t in early[10:]:  # version-0 trials that finish after the update
            study.tell(t, _quadratic(t.params))
        study.sampler = _sampler()
        late = [study.ask(fixed_distributions=FLOATS) for _ in range(6)]
        for t in late:
            study.tell(t, _quadratic(t.params))
        second = _sampler().rebuild(study)
        assert second is not None
        assert second["cohort_size"] == 6 and second["earlier_version_trials"] == 12

    def test_failed_and_non_finite_trials_are_ignored(self) -> None:
        study = _study(FLOATS, _sampler())
        trials = [study.ask(fixed_distributions=FLOATS) for _ in range(8)]
        for t in trials[:5]:
            study.tell(t, _quadratic(t.params))
        study.tell(trials[5], float("inf"))
        study.tell(trials[6], state=optuna.trial.TrialState.FAIL)
        summary = _sampler().rebuild(study)
        assert summary is not None and summary["cohort_size"] == 5

    def test_a_changed_search_space_is_refused(self) -> None:
        study = _study(FLOATS)
        _run(study, _sampler, FLOATS, 10)
        store_declared_space(study, {"x0": FloatDistribution(-5.0, 5.0)})
        with pytest.raises(ValueError, match="different search space"):
            _sampler().rebuild(study)

    def test_budget_is_enforced_before_any_circuit_runs(self) -> None:
        def limited(rng: Any, ledger: ResourceLedger) -> NumpyExecutor:
            return NumpyExecutor(100, rng, ledger)

        config = QuantumPolicyConfig(
            algorithm="gaus",
            n_feature_qubits=4,
            seed=1,
            shots=100,
            max_total_shots=1000,
        )
        study = _study(FLOATS, SimQuantumGausSampler(config, limited, NumpyExecutor))
        for _ in range(10):
            t = study.ask(fixed_distributions=FLOATS)
            study.tell(t, _quadratic(t.params))
        with pytest.raises(QuantumBudgetExceededError):
            SimQuantumGausSampler(config, limited, NumpyExecutor).rebuild(study)
        assert SNAPSHOT_KEY not in study.user_attrs

    def test_finite_shots_still_learn(self) -> None:
        def shots(rng: Any, ledger: ResourceLedger) -> NumpyExecutor:
            return NumpyExecutor(4000, rng, ledger)

        config = QuantumPolicyConfig(
            algorithm="gaus", n_feature_qubits=4, seed=2, shots=4000
        )
        study = _study(FLOATS)
        _run(
            study,
            lambda: SimQuantumGausSampler(config, shots, NumpyExecutor),
            FLOATS,
            30,
        )
        stored = Snapshot.from_json(study.user_attrs[SNAPSHOT_KEY])
        assert stored.resources["shots"] > 0 and stored.version == 3

    def test_maximize_flips_the_reward(self) -> None:
        study = _study(FLOATS, direction="maximize")
        _run(study, _sampler, FLOATS, 60, lambda p: -_quadratic(p))
        assert max(t.value for t in study.trials) > -3.0


class TestHardwareGuard:
    @pytest.mark.parametrize("cls", [QuantumGausSampler, QuantumMixtureSampler])
    def test_a_hardware_sampler_refuses_to_rebuild_without_the_flag(
        self, cls: Type[QuantumPolicySampler], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ALLOW_HARDWARE_ENV, raising=False)
        built = []

        def factory(rng: Any, ledger: ResourceLedger) -> NumpyExecutor:
            built.append(1)
            return NumpyExecutor(None, None, ledger)

        algorithm = "gaus" if cls is QuantumGausSampler else "mixture"
        extra = (
            {"n_feature_qubits": 4}
            if algorithm == "gaus"
            else {"n_mode_qubits": 2, "n_feature_qubits": 3}
        )
        sampler = cls(
            QuantumPolicyConfig(algorithm=algorithm, seed=1, **extra),
            factory,
            NumpyExecutor,
        )
        study = _study(FLOATS, sampler)
        for _ in range(6):
            t = study.ask(fixed_distributions=FLOATS)
            study.tell(t, _quadratic(t.params))
        with pytest.raises(PermissionError, match=ALLOW_HARDWARE_ENV):
            sampler.rebuild(study)
        monkeypatch.setenv(ALLOW_HARDWARE_ENV, "0")
        with pytest.raises(PermissionError):
            sampler.rebuild(study)
        assert built == []  # the provider was never contacted
        monkeypatch.setenv(ALLOW_HARDWARE_ENV, "1")
        assert sampler.rebuild(study) is not None and built == [1]

    def test_simulated_samplers_need_no_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ALLOW_HARDWARE_ENV, raising=False)
        study = _study(FLOATS)
        _run(study, _sampler, FLOATS, 10)
        assert SNAPSHOT_KEY in study.user_attrs


class TestLearning:
    @pytest.mark.parametrize("encoding", ["register", "head"])
    @pytest.mark.parametrize("cls", ALL_SIM)
    def test_beats_random_search_on_a_mixed_space(
        self, cls: Type[QuantumPolicySampler], encoding: str
    ) -> None:
        quantum, random = [], []
        for seed in range(3):
            study = _study(MIXED)
            _run(
                study,
                lambda s=seed: _sampler(cls, seed=s, categorical_encoding=encoding),  # type: ignore[misc]
                MIXED,
                70,
                _mixed_objective,
            )
            quantum.append(_best(study))
            rand = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
            for _ in range(70):
                t = rand.ask(fixed_distributions=MIXED)
                rand.tell(t, _mixed_objective(t.params))
            random.append(_best(rand))
        assert np.mean(quantum) < np.mean(random)

    def test_categorical_probability_moves_towards_the_good_choice(self) -> None:
        space = {
            "c": CategoricalDistribution(["bad", "good", "bad2"]),
            "x": FloatDistribution(0, 1),
        }
        study = _study(space)
        _run(
            study,
            lambda: _sampler(
                categorical_encoding="head", lr_phi=0.1, lr_theta=0.1, kl_stop=0.2
            ),
            space,
            80,
            lambda p: 0.0 if p["c"] == "good" else 1.0,
            rebuild_every=10,
        )
        stored = Snapshot.from_json(study.user_attrs[SNAPSHOT_KEY])
        assert stored.dist.cats[0][0, 1] > 1 / 3 + 0.1

    def test_mixture_uses_several_modes_in_the_first_cohort(self) -> None:
        study = _study(FLOATS, _sampler(SimQuantumMixtureSampler))
        for _ in range(30):
            study.tell(study.ask(fixed_distributions=FLOATS), 0.0)
        modes = {t.system_attrs[BEHAVIOR_KEY]["mode"] for t in study.trials}
        assert len(modes) >= 3

    def test_updates_keep_probabilities_normalised(self) -> None:
        study = _study(MIXED)
        _run(
            study,
            lambda: _sampler(SimQuantumMixtureSampler),
            MIXED,
            40,
            _mixed_objective,
        )
        stored = Snapshot.from_json(study.user_attrs[SNAPSHOT_KEY])
        assert stored.dist.weights.sum() == pytest.approx(1.0)
        assert np.all(stored.dist.sigmas > 0)
        for probs in stored.dist.cats:
            assert probs.sum(axis=1) == pytest.approx(np.ones(4))
