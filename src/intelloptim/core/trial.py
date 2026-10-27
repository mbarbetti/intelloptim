"""Trial lifecycle states.

A *trial* is one evaluation of the objective: the sampler proposes parameter
values, an evaluator computes the objective value, and the result is reported
back to the study. :class:`TrialState` names the states a trial moves through,
from waiting to evaluated, pruned or failed. It is shared by the client, the
server and the wire schemas.

The client-side wrapper that exposes ``trial.suggest_float(...)`` and friends
to objective code lives in :mod:`intelloptim.client.trial`.
"""

from __future__ import annotations

from enum import Enum


class TrialState(str, Enum):
    """Lifecycle states a trial can occupy."""

    WAITING = "waiting"
    """Created (e.g. via enqueue) but not yet handed to an evaluator."""

    RUNNING = "running"
    """Handed to an evaluator; awaiting the matching ``tell``."""

    COMPLETE = "complete"
    """Evaluated successfully; ``value`` is set."""

    PRUNED = "pruned"
    """Stopped early by the pruner; ``value`` may be ``None``."""

    FAILED = "failed"
    """The evaluation raised; ``value`` is ``None``."""


__all__ = ["TrialState"]
