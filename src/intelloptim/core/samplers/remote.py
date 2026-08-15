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

from pydantic import Field

from intelloptim.core.samplers.base import RebuildPolicy, Sampler, SamplerKind


class RemoteRandom(Sampler):
    """Random sampler dispatched via the optimizer-worker plane.

    Args:
        seed: Random seed for reproducibility. ``None`` lets Optuna's
            ``RandomSampler`` pick.
        rebuild_every: Override the class-level rebuild trigger. When
            set, the orchestrator schedules a rebuild after every N
            completed tells; ``None`` keeps the default policy.
    """

    kind: Literal[SamplerKind.REMOTE_RANDOM] = SamplerKind.REMOTE_RANDOM

    seed: Optional[int] = None
    rebuild_every: Optional[int] = Field(default=None, gt=0)

    runs_remote: ClassVar[bool] = True

    rebuild_policy: ClassVar[RebuildPolicy] = RebuildPolicy(
        every_n_tells=10, every_seconds=None, on_demand=True
    )


__all__ = ["RemoteRandom"]
