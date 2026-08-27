"""Runtime errors raised by the algorithm layer.

These errors originate server-side during spec resolution or plugin dispatch.
Plugin authors and orchestrator consumers should import from here rather than
from the specific module where the error is raised.
"""

from __future__ import annotations


class SamplerNotImplementedError(NotImplementedError):
    """Raised when no implementation is registered for a sampler kind.

    Thrown by :func:`~intelloptim.algorithms.samplers.resolve_sampler` when
    the spec's ``kind`` has no entry in ``_RESOLVERS`` — either because the
    kind is a plugin stub (e.g. quantum samplers) and no plugin has been
    registered, or because an unrecognised kind was passed.

    Plugin authors should catch this error to detect missing registrations.
    See ``docs/samplers/quantum_plugin.md`` for the plugin contract.
    """


__all__ = ["SamplerNotImplementedError"]
