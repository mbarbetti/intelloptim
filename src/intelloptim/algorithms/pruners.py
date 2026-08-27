"""Resolve pruner specs to Optuna pruner instances.

Called by the optimizer worker when it initialises an :class:`optuna.Study`.
Each :class:`~intelloptim.core.pruners.PrunerKind` maps to a private resolver
that constructs the matching :class:`optuna.pruners.BasePruner` from the
spec's validated fields.

The only public entry point is :func:`resolve_pruner`; ``_RESOLVERS`` and the
private ``_resolve_*`` functions are implementation details.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import optuna

from intelloptim.core.pruners import (
    PrunerKind,
    NopPrunerSpec, 
    MedianPrunerSpec,
    PercentilePrunerSpec,
    ThresholdPrunerSpec,
    PrunerSpec, 
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
    """Instantiate the Optuna pruner described by *spec*.

    Args:
        spec (PrunerSpec): A validated pruner spec — one of
            :class:`~intelloptim.core.pruners.NopPrunerSpec`,
            :class:`~intelloptim.core.pruners.MedianPrunerSpec`,
            :class:`~intelloptim.core.pruners.PercentilePrunerSpec`, or
            :class:`~intelloptim.core.pruners.ThresholdPrunerSpec`.

    Returns:
        optuna.pruners.BasePruner: A configured pruner ready to be passed to
        :class:`optuna.Study`.

    Raises:
        KeyError: If ``spec.kind`` has no registered resolver. This indicates a
            :class:`~intelloptim.core.pruners.PrunerKind` value was added
            without updating ``_RESOLVERS``.
    """
    return _RESOLVERS[spec.kind](spec)


__all__ = ["resolve_pruner"]
