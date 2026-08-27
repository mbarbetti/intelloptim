"""Sampler base spec.

Sampler specs are **declarative pydantic models** the user constructs
(``intopt.samplers.TPE(seed=42)``) that travel over the wire as part of a
study spec and participate in the canonical study hash.

The runtime plugin contract (:class:`~intelloptim.algorithms.protocols.SamplerPlugin`)
lives in :mod:`intelloptim.algorithms.protocols`, keeping heavy dependencies
(``optuna``, ``qiskit``) out of this pure core layer.
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
    REMOTE_RANDOM = "remote_random"
    QUANTUM_QAOA = "quantum_qaoa"
    QUANTUM_VQE = "quantum_vqe"


class RebuildPolicy(BaseModel):
    """Define when the optimizer worker should rebuild a sampler's state.

    Heavy samplers, such as Bayesian optimization surrogates and hybrid
    quantum-classical loops, typically refit from accumulated trials rather
    than after each trial. During a rebuild, the study enters
    :attr:`StudyState.REBUILDING`, and ``/ask`` returns ``503 Retry-After``.

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
    """Base model for a sampler's declarative, serializable configuration.

    Concrete subclasses define their sampler identity and execution mode with
    the :attr:`kind` and :attr:`runs_remote` class variables. Sampler-specific
    options are declared as pydantic model fields and therefore participate in
    validation, serialization, and the canonical study hash.

    This model describes configuration only; runtime sampling behavior is
    provided separately by a :class:`~intelloptim.algorithms.protocols.SamplerPlugin` implementation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    runs_remote: ClassVar[bool] = False
    """Whether the sampler should be dispatched to an optimizer worker
    (Celery) or run inline in the orchestrator process."""

    rebuild_policy: ClassVar[Optional[RebuildPolicy]] = None
    """When set, the orchestrator schedules sampler rebuilds per the
    policy. Lightweight inline samplers leave this as ``None``."""
    
    
class BaseRebuildSamplerSpec(BaseSamplerSpec):
    """Marker base class for all rebuildable sampler specs.

    Subclasses must declare a ``kind`` (string discriminator) and any
    backend-specific kwargs as pydantic fields. They inherit the default
    :attr:`runs_remote` = ``True`` and a sensible :attr:`rebuild_policy`,
    so the orchestrator routes them through the Celery optimizer plane
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
