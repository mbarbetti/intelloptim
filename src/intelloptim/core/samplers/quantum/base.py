"""Vendor-neutral quantum sampler base.

External quantum implementations (Qiskit-based, PennyLane-based, D-Wave-
based, ...) subclass :class:`QuantumSampler` to declare a spec, and
register a runtime :class:`~intelloptim.core.samplers.base.SamplerPlugin`
implementation via the ``intelloptim.samplers`` entry-point group.

This file contains *no* vendor imports. ``intelloptim`` ships the slot
wired through to the optimizer-worker plane (``runs_remote = True``); the
implementation lives outside this repository.
"""

from __future__ import annotations

from typing import ClassVar

from intelloptim.core.samplers.base import RebuildPolicy, Sampler


class QuantumSampler(Sampler):
    """Marker base class for all quantum sampler specs.

    Subclasses must declare a ``kind`` (string discriminator) and any
    backend-specific kwargs as pydantic fields. They inherit the default
    :attr:`runs_remote` = ``True`` and a sensible :attr:`rebuild_policy`,
    so the orchestrator routes them through the Celery optimizer plane
    without per-subclass configuration.
    """

    runs_remote: ClassVar[bool] = True

    rebuild_policy: ClassVar[RebuildPolicy] = RebuildPolicy(
        every_n_tells=10,
        every_seconds=None,
        on_demand=True,
    )


__all__ = ["QuantumSampler"]
