"""Sampler specs for Optuna's built-in strategies.

Each class in this module is a plain configuration object — it captures the
parameters that control one sampling strategy, nothing more. Constructing a
spec is cheap and has no side effects, and no Optuna import happens here;
validation happens at construction time so you catch misconfigured samplers
before a study is ever submitted. The actual ``optuna.samplers.*`` instance is
created server-side by :func:`~intelloptim.algorithms.samplers.resolve_sampler`
at study-resolution time.

Five strategies are available:

- :class:`RandomSamplerSpec` — draws each parameter independently and
  uniformly; a zero-configuration baseline that needs no warm-up and never
  exploits prior results. Use it to sanity-check the search space or when
  you have too few trials for a surrogate model to pay off.
- :class:`TPESamplerSpec` — Tree-structured Parzen Estimator; models the
  densities of good and poor trials and samples where their ratio is highest.
  A solid default for most single-objective studies.
- :class:`GPSamplerSpec` — Gaussian-process Bayesian optimisation; fits a
  surrogate to past trials and maximises an acquisition function. Best for
  low-dimensional, expensive-to-evaluate objectives where the refit cost is
  acceptable.
- :class:`NSGAIISamplerSpec` — NSGA-II genetic algorithm; evolves a population
  using non-dominated sorting. Use it when the study has two or more
  objectives.
- :class:`QMCSamplerSpec` — Quasi-Monte Carlo low-discrepancy sequences; fills
  the space more uniformly than random draws. Use ``qmc_type="sobol"`` in low
  dimensions or ``"halton"`` in higher ones, for space-filling initial
  exploration.

All five share the same base: they inherit from :class:`BaseSamplerSpec`,
which makes them immutable and strict about unknown fields.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from intelloptim.core.samplers.base import BaseSamplerSpec, SamplerKind


class RandomSamplerSpec(BaseSamplerSpec):
    """Pure random sampler — draws each parameter independently and uniformly.

    A zero-configuration baseline that requires no warm-up trials and never
    exploits prior results. Useful for sanity-checking the search space or
    when the number of trials is too small to benefit from surrogate models.

    See `Optuna RandomSampler
    <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html>`_
    for full details.

    Args:
        seed (Optional[int]): Random seed for reproducibility. ``None`` lets
            Optuna pick a seed automatically.
    """

    kind: Literal[SamplerKind.RANDOM] = SamplerKind.RANDOM

    seed: Optional[int] = None


class TPESamplerSpec(BaseSamplerSpec):
    """Tree-structured Parzen Estimator (TPE) sampler.

    Builds two density estimates over past trial values — one over the best
    results, one over the rest — and suggests the next point by maximising
    their ratio. The recommended default for most single-objective studies:
    it balances exploitation of promising regions with exploration, and adds
    no surrogate-refit overhead.

    See `Optuna TPESampler
    <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html>`_
    for full details.

    Args:
        seed (Optional[int]): Random seed for reproducibility. ``None`` lets
            Optuna pick a seed automatically.
        n_startup_trials (int): Number of initial trials sampled at random
            before TPE activates. Defaults to ``10``.
        n_ei_candidates (int): Candidate samples drawn from the "good"
            distribution per ask call. Higher values improve
            expected-improvement estimates at the cost of more computation.
            Defaults to ``24``.
        multivariate (bool): When ``True``, model joint parameter
            dependencies instead of treating each parameter independently.
            Recommended when parameters interact. Defaults to ``True``.
        group (bool): When ``True`` and ``multivariate=True``, decompose
            the search space into independent subspaces and sample each
            jointly. Requires ``multivariate=True``. Defaults to ``False``.
    """

    kind: Literal[SamplerKind.TPE] = SamplerKind.TPE

    seed: Optional[int] = None
    n_startup_trials: int = Field(default=10, ge=0)
    n_ei_candidates: int = Field(default=24, gt=0)
    multivariate: bool = True
    group: bool = False

    @model_validator(mode="after")
    def _check_group(self) -> "TPESamplerSpec":
        if self.group and not self.multivariate:
            raise ValueError("TPESamplerSpec: group=True requires multivariate=True")
        return self


class GPSamplerSpec(BaseSamplerSpec):
    """Gaussian-process (GP) sampler.

    Fits a GP surrogate to observed trials and maximises an acquisition
    function to suggest the next point. Best suited for low-dimensional
    search spaces (roughly ≤ 20 parameters) where each evaluation is
    expensive and the GP refit cost is acceptable.

    See `Optuna GPSampler
    <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html>`_
    for full details.

    Args:
        seed (Optional[int]): Random seed for reproducibility. ``None`` lets
            Optuna pick a seed automatically.
        n_startup_trials (int): Number of initial random trials before the
            GP model is first fitted. Defaults to ``10``.
        deterministic_objective (bool): Set to ``True`` when the objective
            contains no stochastic noise (e.g. a deterministic simulation).
            Fixes the GP noise variance near zero, sharpening predictions
            but preventing the model from absorbing random variation.
            Defaults to ``False``.
    """

    kind: Literal[SamplerKind.GP] = SamplerKind.GP

    seed: Optional[int] = None
    n_startup_trials: int = Field(default=10, ge=0)
    deterministic_objective: bool = False


class NSGAIISamplerSpec(BaseSamplerSpec):
    """Multi-objective sampler using the NSGA-II genetic algorithm.

    Maintains a population of candidate solutions and evolves it using
    non-dominated sorting and crowding-distance selection. Use this sampler
    when the study has two or more objectives (pass a list to the
    ``direction`` argument of ``create_study``).

    See `Optuna NSGAIISampler
    <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html>`_
    for full details.

    Args:
        population_size (int): Number of individuals maintained across
            generations. Larger populations explore more broadly but require
            more trials before converging. Defaults to ``50``.
        mutation_prob (Optional[float]): Per-parameter mutation probability.
            ``None`` uses Optuna's default of ``1.0 / n_params``. Defaults
            to ``None``.
        crossover_prob (float): Probability that two parents exchange
            parameters when producing an offspring. Defaults to ``0.9``.
        swapping_prob (float): Per-parameter probability of swapping values
            between parents during a crossover. Defaults to ``0.5``.
        seed (Optional[int]): Random seed for reproducibility. ``None`` lets
            Optuna pick a seed automatically.
    """

    kind: Literal[SamplerKind.NSGA2] = SamplerKind.NSGA2

    population_size: int = Field(default=50, ge=2)
    mutation_prob: Optional[float] = Field(default=None, ge=0, le=1)
    crossover_prob: float = Field(default=0.9, ge=0, le=1)
    swapping_prob: float = Field(default=0.5, ge=0, le=1)
    seed: Optional[int] = None


class QMCSamplerSpec(BaseSamplerSpec):
    """Quasi-Monte Carlo (QMC) sampler using low-discrepancy sequences.

    Unlike pseudo-random sampling, QMC sequences fill the parameter space
    more uniformly, typically yielding better coverage for the same number
    of trials. A good choice for space-filling initial exploration or when
    reproducibility of the sampling pattern matters.

    See `Optuna QMCSampler
    <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.QMCSampler.html>`_
    for full details.

    Args:
        qmc_type (Literal["sobol", "halton"]): QMC sequence family.
            ``"sobol"`` has better uniformity in low dimensions; ``"halton"``
            scales more gracefully to higher dimensions. Defaults to
            ``"sobol"``.
        scramble (bool): Apply random scrambling to break the deterministic
            structure of the sequence, producing different (but still
            low-discrepancy) samples across runs. Defaults to ``False``.
        seed (Optional[int]): Seed used when ``scramble=True``; ignored
            otherwise. ``None`` picks a seed at random. Defaults to ``None``.
    """

    kind: Literal[SamplerKind.QMC] = SamplerKind.QMC

    qmc_type: Literal["sobol", "halton"] = "sobol"
    scramble: bool = False
    seed: Optional[int] = None


__all__ = [
    "RandomSamplerSpec",
    "TPESamplerSpec",
    "GPSamplerSpec",
    "NSGAIISamplerSpec",
    "QMCSamplerSpec",
]
