"""Runtime errors raised by the algorithm layer.

These errors originate server-side while a spec is being resolved to a runtime
sampler or pruner. Import them from here rather than from the module that
raises them.
"""

from __future__ import annotations


class SamplerNotImplementedError(NotImplementedError):
    """Raised when no runtime implementation exists for a sampler kind.

    Raised by :func:`~intelloptim.algorithms.samplers.resolve_sampler` when the
    spec's ``kind`` has no registered resolver. This happens when the kind is
    a plugin stub (e.g. a quantum sampler) and no plugin provides it, or when
    the kind is not recognised at all.

    Catch this error to detect a missing sampler registration.
    """


__all__ = ["SamplerNotImplementedError"]
