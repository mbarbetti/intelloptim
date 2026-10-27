"""Spec-to-Optuna resolution for pruners.

One public name is exported:

- :func:`resolve_pruner` — instantiate an Optuna pruner from a
  :data:`~intelloptim.core.pruners.PrunerSpec`.

Each :class:`~intelloptim.core.pruners.PrunerKind` maps to a private resolver
that forwards the spec's validated fields to the matching strategy;
``_RESOLVERS`` and the ``_resolve_*`` functions are implementation details.
The server calls :func:`resolve_pruner` when it builds a study.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import optuna

from intelloptim.core.pruners import (
    MedianPrunerSpec,
    NopPrunerSpec,
    PercentilePrunerSpec,
    PrunerKind,
    PrunerSpec,
    ThresholdPrunerSpec,
)


def _resolve_nop(spec: NopPrunerSpec) -> optuna.pruners.BasePruner:
    return optuna.pruners.NopPruner()


def _resolve_median(spec: MedianPrunerSpec) -> optuna.pruners.BasePruner:
    return optuna.pruners.MedianPruner(
        n_startup_trials=spec.n_startup_trials,
        n_warmup_steps=spec.n_warmup_steps,
        interval_steps=spec.interval_steps,
        n_min_trials=spec.n_min_trials,
    )


def _resolve_percentile(spec: PercentilePrunerSpec) -> optuna.pruners.BasePruner:
    return optuna.pruners.PercentilePruner(
        percentile=spec.percentile,
        n_startup_trials=spec.n_startup_trials,
        n_warmup_steps=spec.n_warmup_steps,
        interval_steps=spec.interval_steps,
        n_min_trials=spec.n_min_trials,
    )


def _resolve_threshold(spec: ThresholdPrunerSpec) -> optuna.pruners.BasePruner:
    return optuna.pruners.ThresholdPruner(
        lower=spec.lower,
        upper=spec.upper,
        n_warmup_steps=spec.n_warmup_steps,
        interval_steps=spec.interval_steps,
    )


_RESOLVERS: Dict[PrunerKind, Callable[[Any], optuna.pruners.BasePruner]] = {
    PrunerKind.NOP: _resolve_nop,
    PrunerKind.MEDIAN: _resolve_median,
    PrunerKind.PERCENTILE: _resolve_percentile,
    PrunerKind.THRESHOLD: _resolve_threshold,
}


def resolve_pruner(spec: PrunerSpec) -> optuna.pruners.BasePruner:
    """Instantiate the pruner described by *spec*.

    Args:
        spec (PrunerSpec): A validated pruner spec of any supported kind.

    Returns:
        optuna.pruners.BasePruner: A fresh pruner instance, ready to plug into
        ``optuna.create_study(...)``.

    Raises:
        KeyError: If ``spec.kind`` has no registered resolver, meaning a
            :class:`~intelloptim.core.pruners.PrunerKind` value was added
            without a matching entry in ``_RESOLVERS``.
    """
    return _RESOLVERS[spec.kind](spec)


__all__ = ["resolve_pruner"]
