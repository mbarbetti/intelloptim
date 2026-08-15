"""Resolve a :class:`~intelloptim.core.pruners.Pruner` spec to a
concrete :class:`optuna.pruners.BasePruner` instance.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import optuna

from intelloptim.core.pruners import MedianPruner, NopPruner, Pruner, PrunerKind


def _resolve_nop(_spec: NopPruner) -> optuna.pruners.BasePruner:
    return optuna.pruners.NopPruner()


def _resolve_median(spec: MedianPruner) -> optuna.pruners.BasePruner:
    return optuna.pruners.MedianPruner(
        n_startup_trials=spec.n_startup_trials,
        n_warmup_steps=spec.n_warmup_steps,
        interval_steps=spec.interval_steps,
    )


_RESOLVERS: Dict[PrunerKind, Callable[[Any], optuna.pruners.BasePruner]] = {
    PrunerKind.NOP: _resolve_nop,
    PrunerKind.MEDIAN: _resolve_median,
}


def resolve_pruner(spec: Pruner) -> optuna.pruners.BasePruner:
    """Instantiate the Optuna pruner corresponding to *spec*."""
    return _RESOLVERS[spec.kind](spec)


__all__ = ["resolve_pruner"]
