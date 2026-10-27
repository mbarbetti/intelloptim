"""Non-Optuna sampler backends.

Implementations here are for samplers whose spec lives in
:mod:`intelloptim.core.samplers.builtin` or
:mod:`intelloptim.core.samplers.quantum` and whose runtime logic does not
simply delegate to an Optuna sampler class.

Each backend module exposes a concrete sampler class that is wired into
:data:`~intelloptim.algorithms.samplers._RESOLVERS`, either directly or through
:func:`~intelloptim.algorithms.samplers.load_sampler_plugins`.
"""

from intelloptim.algorithms.backend.builtin import (
    OWGSampler,
    SASampler,
)

__all__ = ["OWGSampler", "SASampler"]
