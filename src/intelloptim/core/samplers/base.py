"""Sampler base spec and plugin protocol.

Two concerns share this file:

* :class:`Sampler` and its subclasses are **declarative specs** —
  pydantic models the user constructs (``intopt.samplers.TPE(seed=42)``)
  and that travel over the wire as part of a study spec. They participate
  in the canonical study hash.
* :class:`SamplerPlugin` is the **runtime contract** an implementation
  must satisfy to be discoverable by the orchestrator. Implementations
  are registered via the ``intelloptim.samplers`` entry-point group; the
  server resolves a :class:`Sampler` spec to a :class:`SamplerPlugin`
  instance at study-resolution time.

This split is deliberate: the spec is portable and pure, the plugin may
import heavy dependencies (``optuna``, ``qiskit``) and is resolved only
where they are available.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from intelloptim.core.parameters import Parameter


class SamplerKind(str, Enum):
    """Discriminator value for the sampler-spec union."""

    TPE = "tpe"
    REMOTE_RANDOM = "remote_random"
    QUANTUM_QAOA = "quantum_qaoa"
    QUANTUM_VQE = "quantum_vqe"


class RebuildPolicy(BaseModel):
    """When the orchestrator should ask the optimizer worker to recompute
    the sampler's internal state.

    A heavy sampler (BO surrogate, hybrid quantum-classical loop) typically
    refits on accumulated trials rather than per-trial. While the rebuild
    is in flight, the study is in :attr:`StudyState.REBUILDING` and the
    server returns ``503 Retry-After`` from ``/ask``.

    Args:
        every_n_tells: Trigger after every N completed tells. ``None`` to
            disable this trigger.
        every_seconds: Trigger after T seconds since the last rebuild.
            ``None`` to disable.
        on_demand: Whether ``POST /studies/{id}/rebuild`` is honoured.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    every_n_tells: Optional[int] = Field(default=None, gt=0)
    every_seconds: Optional[float] = Field(default=None, gt=0)
    on_demand: bool = True


class Sampler(BaseModel):
    """Base class for declarative sampler specs.

    Subclasses set :attr:`kind` and :attr:`runs_remote` as class
    constants and add their own pydantic fields for kwargs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    runs_remote: ClassVar[bool] = False
    """Whether the sampler should be dispatched to an optimizer worker
    (Celery) or run inline in the orchestrator process."""

    rebuild_policy: ClassVar[Optional[RebuildPolicy]] = None
    """When set, the orchestrator schedules sampler rebuilds per the
    policy. Lightweight inline samplers leave this as ``None``."""


class SamplerPlugin(Protocol):
    """Runtime contract every sampler implementation must satisfy.

    Implementations live outside ``core/`` (in ``server/`` for inline
    classical samplers, or in external packages for heavy / quantum ones)
    and are discovered via the ``intelloptim.samplers`` entry-point group.

    Methods are intentionally narrow: an :func:`ask` returns the next
    parameter dict; a :func:`tell` reports the resulting value back so
    the sampler can update; an optional :func:`rebuild` recomputes the
    sampler's internal state on the accumulated history.
    """

    kind: SamplerKind
    runs_remote: bool

    def ask(self, parameters: "List[Parameter]") -> Dict[str, Any]:
        """Return the next set of parameter values to evaluate."""

    def tell(self, params: Dict[str, Any], value: float) -> None:
        """Report the objective value for *params* to update internal state."""

    def rebuild(
        self,
        history: "List[tuple[Dict[str, Any], float]]",
    ) -> None:
        """Recompute internal state from scratch using *history*.

        Called by the orchestrator after a :class:`RebuildPolicy` trigger
        fires. May be a no-op for samplers that maintain no fittable state.
        """


__all__ = ["RebuildPolicy", "Sampler", "SamplerKind", "SamplerPlugin"]
