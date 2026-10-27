"""Sampler base types.

A *sampler* decides which point in the search space each trial should
evaluate next, balancing exploration of untried regions against
exploitation of promising ones. This module provides the building blocks
shared by every sampler:

- :class:`SamplerKind` — an enum that uniquely identifies each sampling
  strategy. Every concrete sampler carries one of these values as its
  ``kind`` field, which lets the framework dispatch to the right logic
  without inspecting the full object.

- :class:`RebuildPolicy` — an immutable model describing when a stateful
  sampler should refit its internal state from accumulated trials rather
  than after every single trial.

- :class:`BaseSamplerSpec` — the immutable base model every sampler spec
  inherits from. Inheriting from it enforces that specs are frozen (no
  in-place mutation after construction) and reject unknown fields, so a
  study definition always means exactly what it says.

- :class:`BaseRebuildSamplerSpec` — a :class:`BaseSamplerSpec` variant for
  heavier, stateful samplers: it defaults to remote execution and carries a
  sensible :class:`RebuildPolicy`, so subclasses need no per-class wiring.

Sampler specs are pure configuration objects that travel over the wire as
part of a study spec and participate in the canonical study hash.

Concrete sampler specs live in sibling modules and subclass
:class:`BaseSamplerSpec` (or :class:`BaseRebuildSamplerSpec`) to add their
own configuration fields.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field


class SamplerKind(str, Enum):
    """Supported sampling strategy identifiers."""

    RANDOM = "random"
    TPE = "tpe"
    GP = "gp"
    # CMAES = "cmaes"  # not yet supported
    NSGA2 = "nsga2"
    # NSGA3 = "nsga3"  # not yet supported
    QMC = "qmc"
    SA = "sa"
    OWG = "owg"
    REMOTE_RANDOM = "remote_random"
    QUANTUM_QAOA = "quantum_qaoa"
    QUANTUM_VQE = "quantum_vqe"


class RebuildPolicy(BaseModel):
    """Define when the optimizer worker should rebuild a sampler's state.

    Heavy samplers, such as Bayesian optimization surrogates and hybrid
    quantum-classical loops, typically refit from accumulated trials rather
    than after each trial. During a rebuild, the study is in the ``REBUILDING``
    state, and ``/ask`` returns ``503 Retry-After``.

    Args:
        every_n_tells (Optional[int]): Rebuild after every N completed tells.
            Set to ``None`` to disable this trigger. Defaults to ``None``.
        every_seconds (Optional[float]): Rebuild after this many seconds have
            elapsed since the previous rebuild. Set to ``None`` to disable
            this trigger. Defaults to ``None``.
        on_demand (bool): Whether to allow rebuilds requested through
            ``POST /studies/{id}/rebuild``. Defaults to ``True``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    every_n_tells: Optional[int] = Field(default=None, gt=0)
    every_seconds: Optional[float] = Field(default=None, gt=0)
    on_demand: bool = True


class BaseSamplerSpec(BaseModel):
    """Shared base class for all sampler spec models.

    Concrete subclasses add their own configuration fields and set the
    :attr:`kind` discriminator. Two guarantees apply to every spec built on
    this base: fields cannot be changed after construction (frozen), and
    unknown fields raise a validation error (strict). Together these mean a
    sampler spec always means exactly what it says — no silent defaults, no
    accidental mutation.

    This model describes configuration only; runtime sampling behavior is
    provided separately by a
    :class:`~intelloptim.algorithms.protocols.SamplerPlugin` implementation.
    The :attr:`runs_remote` and :attr:`rebuild_policy` class variables declare
    the sampler's execution mode; being class-level rather than fields, they
    never enter the study hash.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    runs_remote: ClassVar[bool] = False
    """Whether the sampler should be dispatched to an optimizer worker
    (Celery) or run inline in the orchestrator process."""

    rebuild_policy: ClassVar[Optional[RebuildPolicy]] = None
    """When set, the orchestrator schedules sampler rebuilds per the
    policy. Lightweight inline samplers leave this as ``None``."""


class BaseRebuildSamplerSpec(BaseSamplerSpec):
    """Shared base class for stateful, rebuildable sampler specs.

    Extends :class:`BaseSamplerSpec` for heavier strategies — Bayesian
    surrogates, hybrid quantum-classical loops — that refit periodically
    rather than after every trial. Subclasses inherit remote execution
    (:attr:`runs_remote` = ``True``) and a default :class:`RebuildPolicy`, so
    declaring one requires only a :attr:`kind` and any strategy-specific
    fields; the orchestrator routes them through the Celery optimizer plane
    without per-subclass configuration.
    """

    runs_remote: ClassVar[bool] = True

    rebuild_policy: ClassVar[Optional[RebuildPolicy]] = RebuildPolicy(
        every_n_tells=10,
        every_seconds=None,
        on_demand=True,
    )


__all__ = [
    "SamplerKind",
    "RebuildPolicy",
    "BaseSamplerSpec",
    "BaseRebuildSamplerSpec",
]
