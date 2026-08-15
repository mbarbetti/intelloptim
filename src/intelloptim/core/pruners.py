"""Pruner declarations.

Pruners decide whether an in-progress trial should be stopped early on
the basis of its intermediate values. The same Optuna pruners are wrapped
here as declarative specs so they participate in the canonical study hash.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PrunerKind(str, Enum):
    NOP = "nop"
    MEDIAN = "median"


class _PrunerBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NopPruner(_PrunerBase):
    """The no-op pruner — never prunes any trial."""

    kind: Literal[PrunerKind.NOP] = PrunerKind.NOP


class MedianPruner(_PrunerBase):
    """Prune trials whose best intermediate value is worse than the median.

    Args:
        n_startup_trials: Number of trials to complete before pruning is
            considered. Pruning before the surrogate has data is wasteful.
        n_warmup_steps: Number of steps within a single trial before its
            intermediate values are used for pruning decisions.
        interval_steps: Frequency (in steps) at which to evaluate the
            pruning decision.
    """

    kind: Literal[PrunerKind.MEDIAN] = PrunerKind.MEDIAN
    n_startup_trials: int = 5
    n_warmup_steps: int = 0
    interval_steps: int = 1


Pruner = Annotated[Union[NopPruner, MedianPruner], Field(discriminator="kind")]
"""Discriminated-union alias for the supported pruner kinds (tagged by ``kind``)."""

__all__ = ["MedianPruner", "NopPruner", "Pruner", "PrunerKind"]
