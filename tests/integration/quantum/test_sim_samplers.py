"""The simulated samplers end to end with the real Qiskit simulator."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, Type

import numpy as np
import optuna
import pytest
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

pytest.importorskip("qiskit")
pytestmark = pytest.mark.qiskit

from intelloptim.algorithms import resolve_sampler  # noqa: E402
from intelloptim.algorithms.backend.quantum import (  # noqa: E402
    QuantumGausSampler,
    SimQuantumGausSampler,
    SimQuantumMixtureSampler,
)
from intelloptim.algorithms.backend.quantum.commons.base import (
    QuantumPolicySampler,
)  # noqa: E402
from intelloptim.algorithms.backend.quantum.commons.executor import (  # noqa: E402
    ALLOW_HARDWARE_ENV,
    register_executor,
)
from intelloptim.algorithms.backend.quantum.commons.snapshot import (  # noqa: E402
    BEHAVIOR_KEY,
    SNAPSHOT_KEY,
    Snapshot,
)
from intelloptim.algorithms.backend.quantum.qiskit.simulator import (  # noqa: E402
    StatevectorExecutor,
)
from intelloptim.algorithms.space import store_declared_space  # noqa: E402
from intelloptim.core.samplers import (  # noqa: E402
    QuantumGausSamplerSpec,
    SimQuantumGausSamplerSpec,
    SimQuantumMixtureSamplerSpec,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("intelloptim").setLevel(logging.ERROR)

MIXED: Dict[str, BaseDistribution] = {
    "lr": FloatDistribution(1e-4, 1.0, log=True),
    "x": FloatDistribution(-3.0, 3.0),
    "n": IntDistribution(1, 32, log=True),
    "k": IntDistribution(0, 10, step=2),
    "act": CategoricalDistribution(["relu", "tanh", "gelu"]),
    "flag": CategoricalDistribution([True, False, None]),
}


def _objective(p: Dict[str, Any]) -> float:
    return float(
        (np.log10(p["lr"]) + 2) ** 2
        + (p["x"] - 1) ** 2
        + (np.log2(p["n"]) - 3) ** 2
        + (p["k"] - 6) ** 2 / 4
        + {"relu": 1.0, "tanh": 0.0, "gelu": 0.5}[p["act"]]
        + (0.0 if p["flag"] is None else 1.0)
    )


def _drive(storage: Any, name: str, spec: Any, n: int, every: int = 10) -> optuna.Study:
    """Ask/tell like the server: a fresh study handle and sampler per ask."""
    study = optuna.create_study(study_name=name, storage=storage, load_if_exists=True)
    store_declared_space(study, MIXED)
    for i in range(n):
        study = optuna.load_study(
            study_name=name, storage=storage, sampler=resolve_sampler(spec)
        )
        trial = study.ask(fixed_distributions=MIXED)
        study.tell(trial, _objective(trial.params))
        if (i + 1) % every == 0:
            resolve_sampler(spec).rebuild(
                optuna.load_study(study_name=name, storage=storage)
            )
    return optuna.load_study(study_name=name, storage=storage)


@pytest.mark.parametrize(
    "spec_cls, sampler_cls",
    [
        (SimQuantumGausSamplerSpec, SimQuantumGausSampler),
        (SimQuantumMixtureSamplerSpec, SimQuantumMixtureSampler),
    ],
)
@pytest.mark.parametrize("encoding", ["register", "head"])
def test_the_resolver_builds_the_named_sampler(
    spec_cls: Any, sampler_cls: Type[QuantumPolicySampler], encoding: str
) -> None:
    assert type(resolve_sampler(spec_cls(categorical_encoding=encoding))) is sampler_cls


@pytest.mark.parametrize("encoding", ["register", "head"])
@pytest.mark.parametrize(
    "spec_cls", [SimQuantumGausSamplerSpec, SimQuantumMixtureSamplerSpec]
)
def test_full_loop_on_sqlite_storage_with_a_fresh_sampler_per_ask(
    spec_cls: Any, encoding: str
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        storage = "sqlite:///" + os.path.join(tmp, "o.db")
        spec = spec_cls(
            seed=5, rebuild_every=5, min_cohort=2, categorical_encoding=encoding
        )
        study = _drive(storage, "mixed", spec, 12, every=5)
        versions = [t.system_attrs[BEHAVIOR_KEY]["version"] for t in study.trials]
        assert versions == [0] * 5 + [1] * 5 + [2] * 2
        stored = Snapshot.from_json(study.user_attrs[SNAPSHOT_KEY])
        assert stored.version == 2 and stored.names == sorted(MIXED)
        assert all(isinstance(t.params["n"], int) for t in study.trials)


@pytest.mark.parametrize("encoding", ["register", "head"])
def test_mixed_space_study_beats_random_at_equal_budget(encoding: str) -> None:
    quantum, random = [], []
    for seed in range(3):
        spec = SimQuantumMixtureSamplerSpec(seed=seed, categorical_encoding=encoding)
        study = optuna.create_study()
        store_declared_space(study, MIXED)
        for i in range(60):
            study.sampler = resolve_sampler(spec)
            trial = study.ask(fixed_distributions=MIXED)
            study.tell(trial, _objective(trial.params))
            if (i + 1) % 10 == 0:
                resolve_sampler(spec).rebuild(study)
        quantum.append(min(t.value for t in study.trials))
        rand = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
        for _ in range(60):
            trial = rand.ask(fixed_distributions=MIXED)
            rand.tell(trial, _objective(trial.params))
        random.append(min(t.value for t in rand.trials))
    assert np.mean(quantum) < np.mean(random)


class TestPluginProviderOnTheHardwarePath:
    """A plugin provider that reuses the simulator, behind the hardware guard."""

    SPEC: Dict[str, Any] = {
        "provider": {"provider": "pytest_dummy", "options": {"note": "simulator"}},
        "shots": 500,
        "max_total_shots": 10**8,
        "seed": 4,
        "rebuild_every": 6,
        "min_cohort": 2,
        "ppo_epochs": 1,
    }

    @pytest.fixture(autouse=True)
    def _provider(self) -> None:
        self.created: list = []

        def factory(shots: int, ledger: Any, **options: Any) -> StatevectorExecutor:
            self.created.append(options)
            return StatevectorExecutor(shots, np.random.default_rng(0), ledger)

        register_executor("pytest_dummy", factory)

    def test_runs_when_the_environment_allows_hardware(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ALLOW_HARDWARE_ENV, "1")
        spec = QuantumGausSamplerSpec(**self.SPEC)
        assert type(resolve_sampler(spec)) is QuantumGausSampler
        study = optuna.create_study()
        store_declared_space(study, MIXED)
        for _ in range(6):
            study.sampler = resolve_sampler(spec)
            trial = study.ask(fixed_distributions=MIXED)
            study.tell(trial, _objective(trial.params))
        summary = resolve_sampler(spec).rebuild(study)
        assert summary is not None and summary["shots_total"] > 0
        assert self.created == [{"note": "simulator"}]

    def test_is_refused_without_the_flag_and_never_contacts_the_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ALLOW_HARDWARE_ENV, raising=False)
        spec = QuantumGausSamplerSpec(**self.SPEC)
        study = optuna.create_study()
        store_declared_space(study, MIXED)
        for _ in range(6):
            study.sampler = resolve_sampler(spec)
            trial = study.ask(fixed_distributions=MIXED)
            study.tell(trial, _objective(trial.params))
        with pytest.raises(PermissionError):
            resolve_sampler(spec).rebuild(study)
        assert self.created == []
