"""Sampler specs for intelloptim's own (non-Optuna) strategies.

Each class in this module is a plain configuration object — it captures the
parameters that control one sampling strategy, nothing more. Constructing a
spec is cheap and has no side effects, and no sampling logic lives here;
validation happens at construction time so you catch misconfigured samplers
before a study is ever submitted. The runtime implementation lives in
:mod:`intelloptim.algorithms.backend` and is wrapped into an Optuna sampler by
:func:`~intelloptim.algorithms.samplers.resolve_sampler` at study-resolution
time.

Two strategies are available:

- :class:`SASamplerSpec` — simulated annealing; each trial perturbs the last
  accepted point within a neighbourhood sized as a fraction of the search
  range, and accepts worse points with a probability that shrinks as the
  temperature cools. Use it for cheap objectives where a local, stochastic
  search is preferable to fitting a surrogate. The runtime currently supports
  only continuous parameters without a step or log scale.
- :class:`OWGSamplerSpec` — a space-filling initial
  design (Latin hypercube or Halton) followed by sampling from a Gaussian whose
  moments are fitted with the objective values as weights. The fit is a
  periodic, remote *rebuild* step, so use it to exercise the rebuild policy and
  the optimizer worker plane, or for cheap objectives where concentrating
  samples around the best region so far is enough.

Both inherit from :class:`BaseSamplerSpec`, which makes them immutable and
strict about unknown fields. :class:`SASamplerSpec` runs inline in the
orchestrator (``runs_remote=False``); the Gaussian spec extends
:class:`BaseRebuildSamplerSpec`, so it runs on an optimizer worker and is
rebuilt according to a :class:`RebuildPolicy`.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from intelloptim.core.samplers.base import (
    BaseRebuildSamplerSpec,
    BaseSamplerSpec,
    SamplerKind,
)


class SASamplerSpec(BaseSamplerSpec):
    """Simulated annealing sampler for single-objective studies.

    Each trial is a random neighbour of the current accepted point. A completed
    trial becomes the new current point if it is at least as good, or otherwise
    with a probability that shrinks as the temperature cools, so the search
    moves from exploration to exploitation. The sampler keeps no state between
    asks: it replays the chain from the study's completed trials, which makes
    its suggestions reproducible.

    Floats and integers are perturbed within a fraction of their range (in
    log-space for log-scaled parameters, on the grid for stepped ones);
    categorical parameters switch category with a fixed probability.

    Args:
        seed (Optional[int]): Random seed for reproducibility. ``None`` seeds
            from the study name, so suggestions are still deterministic for a
            given study and its history.
        n_startup_trials (int): Number of initial trials sampled uniformly at
            random before annealing starts; the chain starts from the best of
            them. Values below ``1`` behave as ``1``. Defaults to ``10``.
        temperature (float): Initial temperature, on the same scale as the
            objective values. Must be positive. Defaults to ``100.0``.
        cooldown_factor (float): Factor applied to the temperature after every
            completed trial, in ``(0, 1]``. Defaults to ``0.9``.
        neighbor_range_factor (float): Neighbourhood size as a fraction of the
            search range, or the probability of changing a categorical value,
            in ``(0, 1]``. Defaults to ``0.1``.
    """

    kind: Literal[SamplerKind.SA] = SamplerKind.SA

    seed: Optional[int] = None
    n_startup_trials: int = Field(default=10, ge=0)
    temperature: float = Field(default=100.0, gt=0, allow_inf_nan=False)
    cooldown_factor: float = Field(default=0.9, gt=0, le=1)
    neighbor_range_factor: float = Field(default=0.1, gt=0, le=1)


class OWGSamplerSpec(BaseRebuildSamplerSpec):
    """Objective-weighted Gaussian (OWG) sampler.

    Samples from a Gaussian fitted with the objective values as weights.

    Until the first rebuild, trials follow a space-filling initial design.
    Every rebuild then weighs the completed trials by their objective value
    (better trials count more) and stores the moments of the resulting
    histogram as the sampling distribution: a multivariate Gaussian for numeric
    parameters and a weighted histogram for categorical ones. New trials are
    drawn from the last stored model, which therefore lags behind the study
    between rebuilds. The sampler keeps no state between asks.

    The weight of a trial is ``exp(-sharpness * z)`` for minimisation (``+``
    for maximisation), with ``z`` the objective standardised over the fitted
    trials.

    Args:
        seed (Optional[int]): Random seed for reproducibility. ``None`` seeds
            from the study name, so suggestions are still deterministic for a
            given study and its history.
        rebuild_every (int): Number of completed trials between rebuilds. It
            overrides the default rebuild policy trigger and sets the size of
            the Latin hypercube. Defaults to ``20``.
        initial_design (Literal["lhs", "halton"]): Design used before the first
            fit: a Latin hypercube or a scrambled Halton sequence (up to 32
            parameters). Defaults to ``"lhs"``.
        sharpness (float): How strongly the objective favours good trials in
            the weights. ``0`` weighs all trials equally; larger values
            concentrate the model on the best trials. Defaults to ``3.0``.
        min_std (float): Lower bound on the standard deviation of every numeric
            parameter, as a fraction of its range, in ``(0, 1]``. Defaults to
            ``0.01``.
        exploration_prob (float): Probability that a trial is drawn uniformly
            instead of from the model, in ``[0, 1]``. Defaults to ``0.05``.
        rebuild_delay_seconds (float): Time a rebuild sleeps before storing the
            model, up to 60 seconds, to mimic an expensive refit and keep the
            study in the rebuilding state long enough to observe it. Defaults
            to ``0.0``.
    """

    kind: Literal[SamplerKind.OWG] = SamplerKind.OWG

    seed: Optional[int] = None
    rebuild_every: int = Field(default=20, gt=0)
    initial_design: Literal["lhs", "halton"] = "lhs"
    sharpness: float = Field(default=3.0, ge=0, allow_inf_nan=False)
    min_std: float = Field(default=0.01, gt=0, le=1)
    exploration_prob: float = Field(default=0.05, ge=0, le=1)
    rebuild_delay_seconds: float = Field(default=0.0, ge=0, le=60)


__all__ = ["SASamplerSpec", "OWGSamplerSpec"]
