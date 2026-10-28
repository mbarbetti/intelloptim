"""Optuna sampler shared by the quantum policy samplers.

:class:`QuantumPolicySampler` is the engine behind the four public samplers (see
:mod:`~intelloptim.algorithms.backend.quantum.gaus` and
:mod:`~intelloptim.algorithms.backend.quantum.mixture`). The policy is a mixture
of bounded, mixed-variable distributions whose parameters come from a
parameterised quantum circuit, trained with PPO on the objective values of
completed trials (design notes: ``.claude/notes/quantum-samplers.md``).

The sampler keeps no state between calls: the orchestrator builds a fresh one for
every ask. The current policy is a versioned snapshot in the study's user
attributes, written only by :meth:`QuantumPolicySampler.rebuild` (the optimizer
worker's rebuild step, once every ``rebuild_every`` completed trials). Asks only
read that snapshot, draw a candidate from the distribution cached in it, and
record on the trial which version and component produced it. No circuit runs
during an ask once a snapshot exists.
"""

from __future__ import annotations

import os
import zlib
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from optuna import distributions
from optuna.samplers import BaseSampler, RandomSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.trial import FrozenTrial

from intelloptim._internal.logging import get_logger
from intelloptim.algorithms.backend.builtin._common import (
    completed_trials,
    is_maximize,
    make_rng,
    seed_key,
)
from intelloptim.algorithms.backend.quantum.commons.config import QuantumPolicyConfig
from intelloptim.algorithms.backend.quantum.commons.distributions import sample_joint
from intelloptim.algorithms.backend.quantum.commons.executor import (
    ALLOW_HARDWARE_ENV,
    QuantumExecutor,
    executor_for,
)
from intelloptim.algorithms.backend.quantum.commons.learner import (
    Cohort,
    initial_snapshot,
    update,
)
from intelloptim.algorithms.backend.quantum.commons.policy import QuantumPolicy
from intelloptim.algorithms.backend.quantum.commons.resources import ResourceLedger
from intelloptim.algorithms.backend.quantum.commons.snapshot import (
    BEHAVIOR_KEY,
    SNAPSHOT_KEY,
    Snapshot,
    shape_digest,
)
from intelloptim.algorithms.backend.quantum.commons.space import SpaceLayout
from intelloptim.algorithms.space import declared_space

_log = get_logger(__name__)

_TAG_INIT = 11
_TAG_SAMPLE = 12
_TAG_EXECUTOR = 13

ExecutorFactory = Callable[[np.random.Generator, ResourceLedger], QuantumExecutor]
"""Builds the executor an update trains on, from an rng and the ledger to update."""

_PRIOR_CACHE: Dict[Tuple[str, int, str], Snapshot] = {}
_PRIOR_CACHE_SIZE = 32


def _exact_simulator() -> QuantumExecutor:
    from intelloptim.algorithms.backend.quantum.qiskit.simulator import (
        StatevectorExecutor,
    )

    return StatevectorExecutor()


def simulated_factory(spec: Any) -> ExecutorFactory:
    """Return the executor factory of a simulated sampler spec (Qiskit statevector)."""

    def make(rng: np.random.Generator, ledger: ResourceLedger) -> QuantumExecutor:
        from intelloptim.algorithms.backend.quantum.qiskit.simulator import (
            StatevectorExecutor,
        )

        return StatevectorExecutor(spec.shots, rng if spec.shots else None, ledger)

    return make


def hardware_factory(spec: Any) -> ExecutorFactory:
    """Return the executor factory of a hardware sampler spec (provider registry)."""
    provider = spec.provider
    options = (
        dict(provider.options)
        if hasattr(provider, "options")
        else {"backend_name": provider.backend_name}
    )

    def make(rng: np.random.Generator, ledger: ResourceLedger) -> QuantumExecutor:
        return executor_for(
            provider.provider, shots=spec.shots, ledger=ledger, **options
        )

    return make


class QuantumPolicySampler(BaseSampler):
    """Quantum policy sampler for single-objective studies.

    Float, integer and categorical parameters are supported (see
    :mod:`~intelloptim.algorithms.backend.quantum.commons.space`).

    Args:
        config (QuantumPolicyConfig): Hyperparameters and the choice of policy.
        executor_factory (ExecutorFactory): Builds the executor a rebuild trains
            on. Asks never call it.
        prior_factory (Optional[Callable[[], QuantumExecutor]]): Builds the exact
            simulator that defines version 0, the broad initial policy, so that
            hardware is never used for it. Defaults to the Qiskit statevector.

    Raises:
        ValueError: If *config* names a different algorithm than the class runs.
    """

    algorithm: ClassVar[str] = ""
    hardware: ClassVar[bool] = False

    def __init__(
        self,
        config: QuantumPolicyConfig,
        executor_factory: ExecutorFactory,
        prior_factory: Optional[Callable[[], QuantumExecutor]] = None,
    ) -> None:
        if self.algorithm and config.algorithm != self.algorithm:
            raise ValueError(
                f"{type(self).__name__} runs the {self.algorithm!r} algorithm, "
                f"not {config.algorithm!r}"
            )
        self._config = config
        self._executor_factory = executor_factory
        self._prior_factory = prior_factory or _exact_simulator

    @classmethod
    def from_spec(cls, spec: Any) -> "QuantumPolicySampler":
        """Build the sampler that a sampler spec describes."""
        factory = hardware_factory(spec) if cls.hardware else simulated_factory(spec)
        return cls(QuantumPolicyConfig.from_spec(spec), factory)

    # ------------------------------------------------------------------
    # search space
    # ------------------------------------------------------------------

    def _layout(self, study: Study) -> SpaceLayout:
        space = declared_space(study)
        if space is None:
            space = dict(IntersectionSearchSpace().calculate(study))
        return SpaceLayout.from_space(space, self._config.categorical_encoding)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, distributions.BaseDistribution]:
        """Return the declared search space; empty if there is none yet."""
        is_maximize(study, type(self).__name__)
        space = declared_space(study)
        return dict(space) if space else {}

    # ------------------------------------------------------------------
    # snapshot access
    # ------------------------------------------------------------------

    def _digest(self, layout: SpaceLayout) -> str:
        return shape_digest(self._config.model_shape(), layout.digest_payload())

    def _load(self, study: Study, layout: SpaceLayout) -> Snapshot:
        """Return the stored snapshot, or the deterministic version ``0``."""
        digest = self._digest(layout)
        stored = study.user_attrs.get(SNAPSHOT_KEY)
        if stored is not None:
            snapshot = Snapshot.from_json(stored)
            if snapshot.digest != digest:
                raise ValueError(
                    "the stored quantum policy was built for a different search "
                    "space or sampler layout; start a new study"
                )
            return snapshot
        key = (digest, seed_key(self._config.seed, study), type(self).__name__)
        cached = _PRIOR_CACHE.get(key)
        if cached is None:
            policy = QuantumPolicy(self._config, layout)
            rng = make_rng(self._config.seed, study, 0, _TAG_INIT)
            cached = initial_snapshot(self._config, policy, self._prior_factory(), rng)
            if len(_PRIOR_CACHE) >= _PRIOR_CACHE_SIZE:
                _PRIOR_CACHE.pop(next(iter(_PRIOR_CACHE)))
            _PRIOR_CACHE[key] = cached
        return cached

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, distributions.BaseDistribution],
    ) -> Dict[str, Any]:
        """Draw all parameters of the trial jointly from the current policy.

        Args:
            study (Study): The study being sampled.
            trial (FrozenTrial): The trial being created.
            search_space (Dict[str, BaseDistribution]): The declared space.

        Returns:
            Dict[str, Any]: One value per parameter: floats, integers, or the
            original choice objects. The version and component of the draw are
            recorded in the trial's system attributes.
        """
        is_maximize(study, type(self).__name__)
        if not search_space:
            return {}
        layout = self._layout(study)
        snapshot = self._load(study, layout)
        rng = make_rng(self._config.seed, study, trial.number, _TAG_SAMPLE)
        k, actions, cats = sample_joint(snapshot.dist, layout, rng)
        # Optuna offers no public way to write a trial attribute from a sampler.
        study._storage.set_trial_system_attr(
            trial._trial_id,
            BEHAVIOR_KEY,
            {
                "version": snapshot.version,
                "mode": int(k),
                "act": actions.tolist(),
                "cats": cats.tolist(),
            },
        )
        return layout.params(actions, cats)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:
        """Draw uniformly a parameter the policy does not cover."""
        seed = zlib.crc32(f"{self._config.seed}:{trial.number}:{param_name}".encode())
        return RandomSampler(seed=seed).sample_independent(
            study, trial, param_name, param_distribution
        )

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------

    def rebuild(self, study: Study) -> Optional[Dict[str, Any]]:
        """Update the policy from the completed trials of the current version.

        Called by the optimizer worker. Only trials drawn from the current
        snapshot version are used; trials drawn from older versions (already used
        by an earlier update, or completed too late for it) are ignored. Nothing
        happens if fewer than ``min_cohort`` such trials have a usable value.

        Args:
            study (Study): The study to learn from and to write the snapshot to.

        Returns:
            Optional[Dict[str, Any]]: A summary of the update, or ``None`` if it
            was skipped.

        Raises:
            PermissionError: For a hardware sampler, if the worker environment
                does not allow hardware runs.
            ValueError: If the study is multi-objective or its search space is
                not supported.
            QuantumBudgetExceededError: If the update does not fit in the shot
                budget; the stored policy is left unchanged.
        """
        if self.hardware and os.environ.get(ALLOW_HARDWARE_ENV) != "1":
            raise PermissionError(
                f"running on quantum hardware is disabled; set "
                f"{ALLOW_HARDWARE_ENV}=1 in the worker environment to allow it"
            )
        maximize = is_maximize(study, type(self).__name__)
        layout = self._layout(study)
        policy = QuantumPolicy(self._config, layout)
        snapshot = self._load(study, layout)

        ks: List[int] = []
        actions: List[List[float]] = []
        cats: List[List[int]] = []
        rewards: List[float] = []
        earlier = 0
        for trial in completed_trials(study):
            behavior = trial.system_attrs.get(BEHAVIOR_KEY)
            if behavior is None or trial.value is None or not np.isfinite(trial.value):
                continue
            if behavior["version"] != snapshot.version:
                earlier += 1
                continue
            ks.append(int(behavior["mode"]))
            actions.append(list(behavior["act"]))
            cats.append(list(behavior["cats"]))
            rewards.append(float(trial.value) if maximize else -float(trial.value))

        if len(ks) < self._config.min_cohort:
            _log.info(
                "quantum policy not updated: cohort too small",
                extra={"usable": len(ks), "needed": self._config.min_cohort},
            )
            return None

        ledger = ResourceLedger(
            circuits=snapshot.resources["circuits"],
            shots=snapshot.resources["shots"],
            max_shots=self._config.max_total_shots,
        )
        rng = make_rng(self._config.seed, study, snapshot.version, _TAG_EXECUTOR)
        executor = self._executor_factory(rng, ledger)
        new, summary = update(
            self._config,
            policy,
            executor,
            snapshot,
            Cohort(
                np.asarray(ks, dtype=int),
                np.asarray(actions, dtype=float).reshape(len(ks), len(layout.numeric)),
                np.asarray(cats, dtype=int).reshape(len(ks), len(layout.categorical)),
                np.asarray(rewards),
            ),
        )
        study.set_user_attr(SNAPSHOT_KEY, new.to_json())
        summary["earlier_version_trials"] = earlier
        return summary


__all__ = [
    "ExecutorFactory",
    "QuantumPolicySampler",
    "hardware_factory",
    "simulated_factory",
]
