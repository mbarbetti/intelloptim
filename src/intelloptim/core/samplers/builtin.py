"""Synthetic remote-dispatched sampler used to exercise the optimizer
worker plane without quantum dependencies.

:class:`RemoteRandom` is a first-class sampler spec ``runs_remote=True``
that the orchestrator routes through Celery. Its runtime implementation
on the worker side is a plain :class:`optuna.samplers.RandomSampler`,
so it's algorithmically uninteresting — its value is that it lets us
test the Celery dispatch path, the REBUILDING state machine, and the
``intelloptim-worker --role optimizer`` CLI end-to-end without standing
up a quantum backend.

The ``rebuild_every`` field overrides the class default
:class:`RebuildPolicy` so individual tests can pick a short trigger
threshold (e.g. ``rebuild_every=2``) without monkeypatching the
orchestrator.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Optional

from intelloptim.core.samplers.base import (
    SamplerKind,
    RebuildPolicy,
    BaseSamplerSpec,
    BaseRebuildSamplerSpec,
)


class SASamplerSpec(BaseSamplerSpec):
    kind: Literal[SamplerKind.SA] = SamplerKind.SA
    
    seed: Optional[int] = None
    n_startup_trials: int = 10
    temperature: float = 100.0
    cooldown_factor: float = 0.9
    neighbor_range_factor: float = 0.1
    
    runs_remote: ClassVar[bool] = False


__all__ = ["SASamplerSpec"]
