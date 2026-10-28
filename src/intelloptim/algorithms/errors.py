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


class SamplerDependencyError(ImportError):
    """Raised when a sampler needs a package that is not installed.

    Some Optuna-backed samplers rely on optional packages (``GP`` needs
    ``torch`` and ``scipy``, ``QMC`` needs ``scipy``). Without this check they
    resolve fine and fail on the first ask with a bare ``ModuleNotFoundError``.
    The message names the missing packages and the extra that installs them,
    for example ``pip install "intelloptim[gp]"``.
    """


__all__ = ["SamplerDependencyError", "SamplerNotImplementedError"]
