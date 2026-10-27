"""Pruner specs for Optuna's built-in strategies.

Each class in this module is a plain configuration object — it captures the
parameters that control one pruning strategy, nothing more. Constructing a
spec is cheap and has no side effects; validation happens at construction time
so you catch misconfigured pruners before a study is ever submitted.

Four strategies are available:

- :class:`NopPrunerSpec` — disables pruning entirely; every trial runs to
  completion. Use when your objective has no intermediate checkpoints or when
  you want a baseline run without early stopping.
- :class:`MedianPrunerSpec` — stops a trial when its best intermediate value is
  worse than the median of already-completed trials at the same step. A solid
  default for most iterative training workloads.
- :class:`PercentilePrunerSpec` — like the median rule but generalised: stops a
  trial when it falls below a configurable percentile of the reference
  population. Use ``percentile=25`` for aggressive pruning (bottom quartile
  cut) or ``percentile=75`` for lenient pruning.
- :class:`ThresholdPrunerSpec` — stops a trial the moment any intermediate value
  crosses a hard lower or upper bound. Best when you know an absolute
  performance requirement (e.g. validation loss must stay below 2.0).

All four share the same base: they inherit from :class:`BasePrunerSpec`, which
makes them immutable and strict about unknown fields.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from intelloptim.core.pruners.base import BasePrunerSpec, PrunerKind


class NopPrunerSpec(BasePrunerSpec):
    """Pruner that never stops a trial early.

    Every trial runs to completion regardless of its intermediate values.
    A sensible starting point when your objective has no natural intermediate
    checkpoints, or when you want a reference run without early stopping.

    See `Optuna NopPruner
    <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.NopPruner.html>`_
    for full details.
    """

    kind: Literal[PrunerKind.NOP] = PrunerKind.NOP


class MedianPrunerSpec(BasePrunerSpec):
    """Pruner using the median stopping rule.

    After each reported checkpoint (*step*), the trial's best value so far
    is compared against completed trials that reported at the same step. If
    it is worse than the median of that group, the trial is stopped.

    A step is any integer label you pass to ``trial.report(value, step=...)``;
    epoch numbers and iteration counts are the most common choices.

    See `Optuna MedianPruner
    <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html>`_
    for full details.

    Args:
        n_startup_trials (int): Minimum number of completed trials before
            pruning begins. During this warm-up phase all trials run to
            completion.
        n_warmup_steps (int): Number of steps at the start of each trial
            where pruning is suppressed, regardless of intermediate values.
            Useful when early readings are noisy or not yet representative.
        interval_steps (int): Steps between consecutive pruning checks,
            counted from the end of the warm-up phase. A check missed because
            no value was reported at that step fires at the next reported step.
        n_min_trials (int): Minimum number of completed-trial values required
            at the current step to make a pruning decision. If fewer are
            available, the trial continues.
    """

    kind: Literal[PrunerKind.MEDIAN] = PrunerKind.MEDIAN

    n_startup_trials: int = Field(default=5, ge=0)
    n_warmup_steps: int = Field(default=0, ge=0)
    interval_steps: int = Field(default=1, ge=1)
    n_min_trials: int = Field(default=1, ge=1)


class PercentilePrunerSpec(BasePrunerSpec):
    """Pruner using a configurable percentile threshold.

    Works like :class:`MedianPrunerSpec` but generalises the cut-off: a trial is
    stopped when its best value so far falls in the bottom ``percentile`` of
    completed trials that reported at the same step. Setting
    ``percentile=50`` reproduces the median rule.

    See `Optuna PercentilePruner
    <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PercentilePruner.html>`_
    for full details.

    Args:
        percentile (float): Fraction of completed trials to prune, expressed
            as a value in ``[0, 100]``. For example, ``25.0`` prunes the
            bottom quartile.
        n_startup_trials (int): Minimum number of completed trials before
            pruning begins.
        n_warmup_steps (int): Number of steps at the start of each trial
            where pruning is suppressed.
        interval_steps (int): Steps between consecutive pruning checks,
            counted from the end of the warm-up phase.
        n_min_trials (int): Minimum number of reference values required at
            the current step to make a pruning decision.
    """

    kind: Literal[PrunerKind.PERCENTILE] = PrunerKind.PERCENTILE

    percentile: float = Field(default=25.0, ge=0, le=100)
    n_startup_trials: int = Field(default=5, ge=0)
    n_warmup_steps: int = Field(default=0, ge=0)
    interval_steps: int = Field(default=1, ge=1)
    n_min_trials: int = Field(default=1, ge=1)


class ThresholdPrunerSpec(BasePrunerSpec):
    """Pruner that enforces hard value bounds.

    A trial is stopped as soon as any intermediate value falls outside the
    interval ``[lower, upper]`` or becomes ``nan``. Unlike the median and
    percentile pruners, the decision is absolute — it does not depend on
    how other trials are performing.

    See `Optuna ThresholdPruner
    <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.ThresholdPruner.html>`_
    for full details.

    Args:
        lower (Optional[float]): Minimum allowed intermediate value. ``None``
            means no lower bound is applied.
        upper (Optional[float]): Maximum allowed intermediate value. ``None``
            means no upper bound is applied. At least one of ``lower`` and
            ``upper`` is required, and ``lower`` must be below ``upper``.
        n_warmup_steps (int): Number of steps at the start of each trial
            where threshold checks are skipped.
        interval_steps (int): Steps between consecutive threshold checks,
            counted from the end of the warm-up phase.
    """

    kind: Literal[PrunerKind.THRESHOLD] = PrunerKind.THRESHOLD

    lower: Optional[float] = Field(default=None, allow_inf_nan=False)
    upper: Optional[float] = Field(default=None, allow_inf_nan=False)
    n_warmup_steps: int = Field(default=0, ge=0)
    interval_steps: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_bounds(self) -> "ThresholdPrunerSpec":
        if self.lower is None and self.upper is None:
            raise ValueError("ThresholdPrunerSpec: set at least one of lower or upper")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            raise ValueError(
                f"ThresholdPrunerSpec: lower ({self.lower}) must be < "
                f"upper ({self.upper})"
            )
        return self


__all__ = [
    "NopPrunerSpec",
    "MedianPrunerSpec",
    "PercentilePrunerSpec",
    "ThresholdPrunerSpec",
]
