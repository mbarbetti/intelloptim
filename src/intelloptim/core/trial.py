"""Trial domain types.

A :class:`Trial` is a passive, server-authoritative record of one
optimization step: its identifier, its position in the study, the
parameter values the sampler suggested, and (once evaluated) its value
and state.

The client-side wrapper that exposes ``trial.suggest_float(...)`` etc. is
a thin layer on top and lives in ``intelloptim.client.trial`` (Phase 2).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TrialState(str, Enum):
    """Lifecycle states a trial can occupy."""

    WAITING = "waiting"
    """Created (e.g. via enqueue) but not yet handed to an evaluator."""

    RUNNING = "running"
    """Handed to an evaluator; awaiting the matching ``tell``."""

    COMPLETE = "complete"
    """Evaluated successfully; ``value`` is set."""

    PRUNED = "pruned"
    """Evaluator stopped the trial early; ``value`` may be ``None``."""

    FAILED = "failed"
    """Evaluator raised; ``value`` is ``None``."""


class Trial(BaseModel):
    """A single optimization step.

    Args:
        id: Server-assigned, opaque trial identifier.
        number: Zero-based ordinal within the study.
        params: Mapping of parameter name to suggested value. Only the
            sampler may write here; the client treats it as read-only.
        state: Current lifecycle state.
        value: Objective value once the trial completes; ``None``
            otherwise.
        intermediate_values: Optional per-step intermediate values used
            by pruners.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    id: str
    number: int
    params: Dict[str, Any]
    state: TrialState = TrialState.RUNNING
    value: Optional[float] = None
    intermediate_values: Dict[int, float] = Field(default_factory=dict)


__all__ = ["Trial", "TrialState"]
