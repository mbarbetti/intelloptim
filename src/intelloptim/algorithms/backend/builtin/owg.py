"""Objective-weighted Gaussian (OWG) sampler.

:class:`OWGSampler` is configured by
:class:`~intelloptim.core.samplers.builtin.OWGSamplerSpec`. Its fit is an
explicit, periodic ``rebuild`` step whose result is persisted in the study's
user attributes, which makes it a vehicle for exercising the rebuild plane.
"""

from __future__ import annotations

import math
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from optuna import distributions
from optuna.samplers import BaseSampler
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

_TAG_DESIGN = 4
_TAG_PERMUTATION = 5
_TAG_MODEL = 6

_MODEL_KEY = "intopt:owg"
_MODEL_VERSION = 1
_CATEGORICAL_SMOOTHING = 0.05
_MAX_TRUNCATION_TRIES = 50
_HALTON_PRIMES = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
)  # fmt: skip


def _numeric_bounds(dist: distributions.BaseDistribution) -> Tuple[float, float]:
    """Return the interval a numeric distribution is mapped from, in sampling space.

    Integers are widened by half a step so that the end values are as likely as
    the inner ones once rounded. Log-scaled distributions work in log-space.
    """
    if isinstance(dist, distributions.IntDistribution):
        half = 0.5 * dist.step
        low, high = dist.low - half, dist.high + half
    elif isinstance(dist, distributions.FloatDistribution):
        low, high = dist.low, dist.high
    else:
        raise ValueError(f"Unsupported distribution: {dist!r}")
    return (math.log(low), math.log(high)) if dist.log else (low, high)


def _to_unit(dist: distributions.BaseDistribution, value: float) -> float:
    """Map a numeric parameter value to ``[0, 1]``."""
    low, high = _numeric_bounds(dist)
    if high == low:
        return 0.5
    position = math.log(value) if getattr(dist, "log", False) else float(value)
    return min(max((position - low) / (high - low), 0.0), 1.0)


def _from_unit(dist: distributions.BaseDistribution, unit: float) -> Any:
    """Map ``[0, 1]`` back to a valid value of a numeric distribution."""
    low, high = _numeric_bounds(dist)
    position = low + min(max(unit, 0.0), 1.0) * (high - low)
    value = math.exp(position) if getattr(dist, "log", False) else position
    if isinstance(dist, distributions.IntDistribution):
        snapped = dist.low + round((value - dist.low) / dist.step) * dist.step
        return int(min(max(snapped, dist.low), dist.high))
    assert isinstance(dist, distributions.FloatDistribution)
    if dist.step is not None:
        value = dist.low + round((value - dist.low) / dist.step) * dist.step
    return float(min(max(value, dist.low), dist.high))


def _radical_inverse(index: int, base: int) -> float:
    """Van der Corput radical inverse of *index* in *base*."""
    fraction, result = 1.0, 0.0
    while index > 0:
        fraction /= base
        result += fraction * (index % base)
        index //= base
    return result


class OWGSampler(BaseSampler):
    """Objective-weighted Gaussian (OWG) sampler.

    Fits a Gaussian to past trials, weighting each by its objective value.

    The sampler works in two phases. Until a model has been fitted, it follows
    a space-filling initial design: a Latin hypercube of ``rebuild_every``
    points (one point per stratum in every dimension) or a scrambled Halton
    sequence, followed by uniform random draws. Each :meth:`rebuild` then
    summarises the completed trials as a histogram in which every trial counts
    with a weight that grows with its objective value, and stores the moments
    of that histogram as the sampling distribution: a multivariate Gaussian for
    numeric parameters and a weighted category histogram for categorical ones.
    New trials are drawn from the stored model until the next rebuild, so the
    model is deliberately stale in between.

    The weight of a trial is ``exp(-sharpness * z)`` for minimisation (``+`` for
    maximisation), where ``z`` is the objective standardised over the fitted
    trials. ``sharpness = 0`` weighs all trials equally, which reduces the
    sampler to fitting the whole history; larger values concentrate the model
    on the best trials. The Gaussian is defined on the unit cube (in log-space
    for log-scaled parameters) and truncated to it, its standard deviation never
    falls below ``min_std``, and with probability ``exploration_prob`` a trial
    ignores the model and is drawn uniformly.

    The sampler keeps no state between calls. The model lives in the study's
    user attributes under ``"intopt:owg"``, so a
    rebuild performed in one process (for example an optimizer worker) is seen
    by every sampler instance created afterwards. All random draws depend only
    on the seed and the trial number, so suggestions are reproducible.

    Args:
        rebuild_every (int): Number of completed trials between rebuilds, as
            scheduled by the orchestrator. It also sets the size of the Latin
            hypercube. Must be positive. Defaults to ``20``.
        initial_design (str): Design used before the first fit, ``"lhs"`` or
            ``"halton"``. Halton supports up to 32 parameters; beyond that the
            extra parameters are drawn uniformly. Defaults to ``"lhs"``.
        sharpness (float): How strongly the objective favours good trials in
            the weights. Must be non-negative and finite. Defaults to ``3.0``.
        min_std (float): Lower bound on the standard deviation of every
            numeric parameter, as a fraction of its range, in ``(0, 1]``.
            Defaults to ``0.01``.
        exploration_prob (float): Probability that a trial is drawn uniformly
            instead of from the model, in ``[0, 1]``. Defaults to ``0.05``.
        rebuild_delay_seconds (float): Time :meth:`rebuild` sleeps before
            storing the model, to mimic an expensive refit. Defaults to
            ``0.0``.
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
        rebuild_every: int = 20,
        initial_design: str = "lhs",
        sharpness: float = 3.0,
        min_std: float = 0.01,
        exploration_prob: float = 0.05,
        rebuild_delay_seconds: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if rebuild_every < 1:
            raise ValueError(f"rebuild_every must be >= 1: {rebuild_every}")
        if initial_design not in ("lhs", "halton"):
            raise ValueError(
                f"initial_design must be 'lhs' or 'halton': {initial_design!r}"
            )
        if not (math.isfinite(sharpness) and sharpness >= 0):
            raise ValueError(f"sharpness must be non-negative and finite: {sharpness}")
        if not 0 < min_std <= 1:
            raise ValueError(f"min_std must be in (0, 1]: {min_std}")
        if not 0 <= exploration_prob <= 1:
            raise ValueError(f"exploration_prob must be in [0, 1]: {exploration_prob}")
        if not (math.isfinite(rebuild_delay_seconds) and rebuild_delay_seconds >= 0):
            raise ValueError(
                "rebuild_delay_seconds must be non-negative and finite: "
                f"{rebuild_delay_seconds}"
            )
        self._rebuild_every = rebuild_every
        self._initial_design = initial_design
        self._sharpness = sharpness
        self._min_std = min_std
        self._exploration_prob = exploration_prob
        self._rebuild_delay_seconds = rebuild_delay_seconds
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
        is_maximize(study, "OWGSampler")
        space = IntersectionSearchSpace().calculate(study)
        return {name: dist for name, dist in space.items() if not dist.single()}

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, distributions.BaseDistribution],
    ) -> Dict[str, Any]:
        """Draw the parameters from the stored model.

        Args:
            study (Study): The study being sampled.
            trial (FrozenTrial): The trial being created.
            search_space (Dict[str, BaseDistribution]): Parameters to sample, as
                returned by :meth:`infer_relative_search_space`.

        Returns:
            Dict[str, Any]: Suggested values. Empty when no model has been
            fitted yet, or when the stored model does not match the search
            space; every parameter then goes through :meth:`sample_independent`.
        """
        is_maximize(study, "OWGSampler")
        model = self._load_model(study, search_space)
        if model is None:
            return {}

        rng = make_rng(self._seed, study, trial.number, _TAG_MODEL)
        explore = bool(rng.uniform() < self._exploration_prob)
        params: Dict[str, Any] = {}

        numeric: List[str] = model["numeric"]
        if numeric:
            if explore:
                units = rng.uniform(size=len(numeric))
            else:
                units = self._draw_truncated(
                    np.asarray(model["mean"]), np.asarray(model["cov"]), rng
                )
            for name, unit in zip(numeric, units, strict=True):
                params[name] = _from_unit(search_space[name], float(unit))

        for name, probabilities in model["categorical"].items():
            choices = search_space[name].choices  # type: ignore[attr-defined]
            if explore:
                probabilities = [1.0 / len(choices)] * len(choices)
            params[name] = choices[int(rng.choice(len(choices), p=probabilities))]

        _log.debug(
            "owg proposal",
            extra={
                "trial": trial.number,
                "n_trials_fit": model["n_trials"],
                "explore": explore,
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
        """Sample from the initial design, reproducibly for this trial.

        Used before the first fit and for parameters outside the stored model.
        Beyond the Latin hypercube, or beyond 32 Halton dimensions, the draw is
        uniform. The Halton dimension of a parameter is the number of
        parameters already suggested in the trial, so it follows the order in
        which the study declares them.

        Args:
            study (Study): The study being sampled.
            trial (FrozenTrial): The trial being created.
            param_name (str): Name of the parameter to sample.
            param_distribution (BaseDistribution): Distribution of the parameter.

        Returns:
            Any: The sampled value.
        """
        is_maximize(study, "OWGSampler")
        unit = self._initial_unit(study, trial, param_name)
        dist = param_distribution
        if isinstance(dist, distributions.CategoricalDistribution):
            count = len(dist.choices)
            return dist.choices[min(int(unit * count), count - 1)]
        return _from_unit(dist, unit)

    def rebuild(self, study: Study) -> Optional[Dict[str, Any]]:
        """Fit the model on the completed trials and store it in the study.

        Intended to be called by the optimizer worker whenever the rebuild
        policy fires. Without at least two completed trials with a finite
        objective, nothing is stored and any previous model is kept.

        Args:
            study (Study): The study to fit; it receives the model as a user
                attribute.

        Returns:
            Optional[Dict[str, Any]]: Summary of the fit (``n_trials_fit`` and
            the effective sample size ``n_eff``), or ``None`` if nothing was
            fitted.

        Raises:
            ValueError: If the study has more than one objective or a
                parameter has an unsupported distribution.
        """
        maximize = is_maximize(study, "OWGSampler")
        space = IntersectionSearchSpace().calculate(study)
        space = {name: dist for name, dist in space.items() if not dist.single()}
        trials = [
            t
            for t in completed_trials(study)
            if t.value is not None and math.isfinite(t.value)
        ]
        if not space or len(trials) < 2:
            _log.info(
                "owg: not enough data to fit",
                extra={"n_trials": len(trials), "n_params": len(space)},
            )
            return None

        weights = self._weights(np.array([t.value for t in trials]), maximize)
        model = self._fit(space, trials, weights)
        if self._rebuild_delay_seconds > 0:
            time.sleep(self._rebuild_delay_seconds)
        study.set_user_attr(_MODEL_KEY, model)
        n_eff = float(1.0 / np.sum(weights**2))
        _log.info(
            "owg rebuilt",
            extra={"n_trials_fit": len(trials), "n_eff": n_eff},
        )
        return {"n_trials_fit": len(trials), "n_eff": n_eff}

    def _weights(self, values: np.ndarray, maximize: bool) -> np.ndarray:
        """Return normalised trial weights that favour good objective values."""
        spread = float(np.std(values))
        scores = (values - np.mean(values)) / spread if spread > 1e-12 else 0 * values
        logits = (self._sharpness if maximize else -self._sharpness) * scores
        weights = np.exp(logits - np.max(logits))
        return np.asarray(weights / np.sum(weights))

    def _fit(
        self,
        space: Dict[str, distributions.BaseDistribution],
        trials: List[FrozenTrial],
        weights: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute the weighted moments of the trials, as a JSON-able model."""
        numeric = sorted(
            n
            for n, d in space.items()
            if not isinstance(d, distributions.CategoricalDistribution)
        )
        categorical = sorted(set(space) - set(numeric))
        model: Dict[str, Any] = {
            "version": _MODEL_VERSION,
            "n_trials": len(trials),
            "distributions": {
                n: distributions.distribution_to_json(d) for n, d in space.items()
            },
            "numeric": numeric,
            "mean": [],
            "cov": [],
            "categorical": {},
        }
        if numeric:
            points = np.array(
                [[_to_unit(space[n], t.params[n]) for n in numeric] for t in trials]
            )
            mean = weights @ points
            centred = points - mean
            reliability = max(1.0 - float(np.sum(weights**2)), 1e-12)
            cov = (centred * weights[:, None]).T @ centred / reliability
            cov = cov + np.eye(len(numeric)) * self._min_std**2
            model["mean"] = mean.tolist()
            model["cov"] = cov.tolist()
        for name in categorical:
            dist = space[name]
            assert isinstance(dist, distributions.CategoricalDistribution)
            histogram = np.zeros(len(dist.choices))
            for trial, weight in zip(trials, weights, strict=True):
                histogram[dist.choices.index(trial.params[name])] += weight
            histogram = histogram + _CATEGORICAL_SMOOTHING
            model["categorical"][name] = (histogram / np.sum(histogram)).tolist()
        return model

    def _load_model(
        self,
        study: Study,
        search_space: Dict[str, distributions.BaseDistribution],
    ) -> Optional[Dict[str, Any]]:
        """Return the stored model if it describes exactly *search_space*."""
        stored = study.user_attrs.get(_MODEL_KEY)
        if not search_space or not isinstance(stored, dict):
            return None
        if stored.get("version") != _MODEL_VERSION:
            return None
        known: Dict[str, str] = stored["distributions"]
        if set(known) != set(search_space):
            return None
        if any(
            distributions.distribution_to_json(dist) != known[name]
            for name, dist in search_space.items()
        ):
            return None
        return stored

    @staticmethod
    def _draw_truncated(
        mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw from the Gaussian truncated to the unit cube (rejection, then clip)."""
        sample = rng.multivariate_normal(mean, cov)
        for _ in range(_MAX_TRUNCATION_TRIES - 1):
            if np.all((sample >= 0.0) & (sample <= 1.0)):
                return np.asarray(sample)
            sample = rng.multivariate_normal(mean, cov)
        return np.asarray(np.clip(sample, 0.0, 1.0))

    def _initial_unit(self, study: Study, trial: FrozenTrial, name: str) -> float:
        """Return the initial-design coordinate in ``[0, 1)`` for one parameter."""
        key = seed_key(self._seed, study)
        name_hash = zlib.crc32(name.encode("utf-8"))
        uniform = float(
            make_rng(self._seed, study, trial.number, _TAG_DESIGN, name_hash).uniform()
        )
        if trial.number >= self._rebuild_every and self._initial_design == "lhs":
            return uniform
        if self._initial_design == "lhs":
            order = np.random.default_rng(
                np.random.SeedSequence([key, _TAG_PERMUTATION, name_hash])
            ).permutation(self._rebuild_every)
            return float((order[trial.number] + uniform) / self._rebuild_every)
        dimension = len(trial.params)
        if dimension >= len(_HALTON_PRIMES):
            return uniform
        shift = float(
            np.random.default_rng(
                np.random.SeedSequence([key, _TAG_PERMUTATION, dimension])
            ).uniform()
        )
        point = _radical_inverse(trial.number + 1, _HALTON_PRIMES[dimension])
        return (point + shift) % 1.0


__all__ = ["OWGSampler"]
