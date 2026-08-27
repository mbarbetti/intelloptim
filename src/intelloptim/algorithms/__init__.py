"""Algorithm layer: resolve declarative sampler/pruner specs to concrete
Optuna objects, and discover third-party sampler plugins.

This package is the open-source home of intelloptim's optimization
algorithms. The private ``intelloptim-server`` imports
:func:`resolve_sampler`, :func:`resolve_pruner`, and
:func:`load_sampler_plugins` from here rather than re-implementing them, so
the algorithm layer and the sampler-plugin contract stay public.
"""

from intelloptim.algorithms import backend, errors, protocols
from intelloptim.algorithms.pruners import resolve_pruner
from intelloptim.algorithms.samplers import load_sampler_plugins, resolve_sampler

__all__ = [
    "backend",
    "errors",
    "protocols",
    "load_sampler_plugins",
    "resolve_pruner",
    "resolve_sampler",
]
