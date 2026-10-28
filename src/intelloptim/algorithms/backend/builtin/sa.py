"""Simulated annealing (SA) sampler.

:class:`SASampler` is configured by
:class:`~intelloptim.core.samplers.builtin.SASamplerSpec`. It keeps no state
between calls: on every ask it replays the Metropolis chain over the study's
completed trials.
"""

from __future__ import annotations

import math
import zlib
from typing import Any, Dict, List, Optional, Tuple

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

_log = get_logger(__name__)

_MIN_TEMPERATURE = 1e-12

_TAG_ACCEPT = 1
_TAG_PROPOSE = 2
_TAG_INDEPENDENT = 3


class SASampler(BaseSampler):
    """Simulated annealing (SA) sampler for single-objective studies.

    The sampler walks a chain of *accepted* points. Each new trial is a random
    neighbour of the current accepted point. Once a trial completes, it
    becomes the current point if it is at least as good as the previous one,
    or otherwise with probability ``exp(-delta / T)``, where ``delta`` is how
    much worse it is and ``T`` is a temperature that is multiplied by
    ``cooldown_factor`` after every trial. Early on, worse points are accepted
    often (exploration); as ``T`` decreases, the chain becomes greedy
    (exploitation).

    The sampler keeps no state between calls. On every ask it replays the
    chain over the study's completed trials, in trial-number order, so a
    freshly created instance suggests exactly what a long-lived one would. With
    parallel workers, this means the chain is defined by the completed trials,
    not by the order in which asks were served.

    The neighbourhood depends on the parameter type. Floats and integers are
    drawn uniformly within ``neighbor_range_factor`` of the search range around
    the current value (in log-space for log-scaled parameters, snapped to the
    grid for stepped ones). Categorical parameters switch to a different
    category with probability ``neighbor_range_factor`` and otherwise keep the
    current one.

    This implementation is freely inspired by the `Optuna simulated annealing
    sampler example
    <https://github.com/optuna/optuna-examples/blob/main/samplers/simulated_annealing_sampler.py>`_.

    Args:
        temperature (float): Initial temperature. It is compared with
            differences in objective value, so it should be on the same scale
            as the objective. Must be positive and finite. Defaults to
            ``100.0``.
        cooldown_factor (float): Factor applied to the temperature after every
            completed trial, in ``(0, 1]``. ``1.0`` keeps the temperature
            constant. Defaults to ``0.9``.
        neighbor_range_factor (float): Size of the neighbourhood as a fraction
            of the search range (or the probability of changing a categorical
            value), in ``(0, 1]``. Defaults to ``0.1``.
        n_startup_trials (int): Number of initial trials sampled uniformly at
            random before annealing starts. The chain starts from the best of
            them. Values below ``1`` are treated as ``1``, since the chain
            needs at least one completed trial. Defaults to ``10``.
        seed (Optional[int]): Random seed for reproducibility. ``None`` seeds
            from the study name, so suggestions are still deterministic for a
            given study and its history. Defaults to ``None``.

    Raises:
        ValueError: If an argument is outside its valid range. At sampling
            time, if the study has more than one objective or a parameter
            has an unsupported distribution.
    """

    def __init__(
        self,
        temperature: float = 100.0,
        cooldown_factor: float = 0.9,
        neighbor_range_factor: float = 0.1,
        n_startup_trials: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        if not (math.isfinite(temperature) and temperature > 0):
            raise ValueError(f"temperature must be positive and finite: {temperature}")
        if not 0 < cooldown_factor <= 1:
            raise ValueError(f"cooldown_factor must be in (0, 1]: {cooldown_factor}")
        if not 0 < neighbor_range_factor <= 1:
            raise ValueError(
                f"neighbor_range_factor must be in (0, 1]: {neighbor_range_factor}"
            )
        if n_startup_trials < 0:
            raise ValueError(f"n_startup_trials must be >= 0: {n_startup_trials}")
        self._temperature = temperature
        self._cooldown_factor = cooldown_factor
        self._neighbor_range_factor = neighbor_range_factor
        self._n_startup_trials = max(n_startup_trials, 1)
        self._seed = seed

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, distributions.BaseDistribution]:
        """Return the parameters shared by all completed trials.

        Parameters with a single possible value are left out; they are handled
        by :meth:`sample_independent`.

        Args:
            study (Study): The study being sampled.
            trial (FrozenTrial): The trial being created.

        Returns:
            Dict[str, BaseDistribution]: Search space for :meth:`sample_relative`.
        """
        is_maximize(study, "SASampler")
        space = IntersectionSearchSpace().calculate(study)
        return {name: dist for name, dist in space.items() if not dist.single()}

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, distributions.BaseDistribution],
    ) -> Dict[str, Any]:
        """Propose a neighbour of the current accepted point.

        Args:
            study (Study): The study being sampled.
            trial (FrozenTrial): The trial being created.
            search_space (Dict[str, BaseDistribution]): Parameters to sample, as
                returned by :meth:`infer_relative_search_space`.

        Returns:
            Dict[str, Any]: Suggested values. Empty during the random warm-up,
            in which case every parameter goes through
            :meth:`sample_independent`.
        """
        maximize = is_maximize(study, "SASampler")
        completed = completed_trials(study)
        if not search_space or len(completed) < self._n_startup_trials:
            return {}

        current, temperature = self._replay(study, completed, maximize)
        rng = make_rng(self._seed, study, trial.number, _TAG_PROPOSE)
        params = {
            name: self._neighbor(dist, current.params[name], rng)
            for name, dist in search_space.items()
        }
        _log.debug(
            "sa proposal",
            extra={
                "trial": trial.number,
                "current_trial": current.number,
                "temperature": temperature,
            },
        )
        return params

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:
        """Sample uniformly at random, reproducibly for this trial and parameter.

        Used during the warm-up and for parameters outside the relative search
        space. A short-lived random sampler is seeded from the trial number and
        parameter name, so a fresh instance per ask does not repeat its draws.

        Args:
            study (Study): The study being sampled.
            trial (FrozenTrial): The trial being created.
            param_name (str): Name of the parameter to sample.
            param_distribution (BaseDistribution): Distribution of the parameter.

        Returns:
            Any: The sampled value.
        """
        entropy = np.random.SeedSequence(
            [
                seed_key(self._seed, study),
                trial.number,
                _TAG_INDEPENDENT,
                zlib.crc32(param_name.encode("utf-8")),
            ]
        )
        sampler = RandomSampler(seed=int(entropy.generate_state(1)[0]))
        return sampler.sample_independent(study, trial, param_name, param_distribution)

    def _temperature_at(self, step: int) -> float:
        return max(
            self._temperature * self._cooldown_factor**step,
            _MIN_TEMPERATURE,
        )

    def _replay(
        self, study: Study, completed: List[FrozenTrial], maximize: bool
    ) -> Tuple[FrozenTrial, float]:
        """Rebuild the chain from history.

        Returns:
            Tuple[FrozenTrial, float]: The current accepted trial and the
            temperature to use for the next proposal.
        """
        warmup = completed[: self._n_startup_trials]
        pick = max if maximize else min
        current = pick(warmup, key=self._value)
        for step, candidate in enumerate(completed[self._n_startup_trials :]):
            probability = self._acceptance_probability(
                self._value(current),
                self._value(candidate),
                self._temperature_at(step),
                maximize,
            )
            draw = make_rng(self._seed, study, candidate.number, _TAG_ACCEPT).uniform()
            if draw < probability:
                current = candidate
        next_step = len(completed) - self._n_startup_trials
        return current, self._temperature_at(next_step)

    @staticmethod
    def _value(trial: FrozenTrial) -> float:
        value = trial.value
        if value is None:
            raise ValueError(f"completed trial {trial.number} has no value")
        return value

    @staticmethod
    def _acceptance_probability(
        current: float, candidate: float, temperature: float, maximize: bool
    ) -> float:
        """Metropolis acceptance probability of *candidate* over *current*."""
        if candidate == current:
            return 1.0
        worsening = current - candidate if maximize else candidate - current
        if worsening <= 0:
            return 1.0
        if math.isinf(worsening):
            return 0.0
        return math.exp(-worsening / temperature)

    def _neighbor(
        self,
        dist: distributions.BaseDistribution,
        current: Any,
        rng: np.random.Generator,
    ) -> Any:
        if isinstance(dist, distributions.CategoricalDistribution):
            return self._neighbor_categorical(dist, current, rng)
        if isinstance(dist, distributions.FloatDistribution):
            return self._neighbor_numeric(
                dist.low, dist.high, dist.log, dist.step, current, rng
            )
        if isinstance(dist, distributions.IntDistribution):
            value = self._neighbor_numeric(
                dist.low, dist.high, dist.log, dist.step, current, rng
            )
            return int(round(value))
        raise ValueError(f"Unsupported distribution: {dist!r}")

    def _neighbor_numeric(
        self,
        low: float,
        high: float,
        log: bool,
        step: Optional[float],
        current: float,
        rng: np.random.Generator,
    ) -> float:
        lo, hi, centre = (
            (math.log(low), math.log(high), math.log(current))
            if log
            else (low, high, current)
        )
        width = (hi - lo) * self._neighbor_range_factor
        value = float(rng.uniform(max(centre - width, lo), min(centre + width, hi)))
        if log:
            value = math.exp(value)
        if step is not None:
            value = low + round((value - low) / step) * step
        return min(max(value, low), high)

    def _neighbor_categorical(
        self,
        dist: distributions.CategoricalDistribution,
        current: Any,
        rng: np.random.Generator,
    ) -> Any:
        choices = dist.choices
        if len(choices) < 2 or rng.uniform() >= self._neighbor_range_factor:
            return current
        index = int(rng.integers(len(choices) - 1))
        if index >= choices.index(current):
            index += 1
        return choices[index]


__all__ = ["SASampler"]
