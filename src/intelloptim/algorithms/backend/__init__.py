"""Non-Optuna sampler backends.

Implementations here are for samplers whose spec lives in
:mod:`intelloptim.core.samplers.builtin` or
:mod:`intelloptim.core.samplers.quantum` and whose runtime logic does not
simply delegate to an Optuna sampler class.

Each backend module exposes a concrete class that satisfies the
:class:`~intelloptim.algorithms.protocols.SamplerPlugin` protocol and is
wired into :data:`~intelloptim.algorithms.samplers._RESOLVERS` by
:func:`~intelloptim.algorithms.samplers.load_sampler_plugins` or directly.
"""

from intelloptim.algorithms.backend.builtin import SASampler
