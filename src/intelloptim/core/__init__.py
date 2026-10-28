"""Pure domain types for intelloptim.

This package is the framework-agnostic kernel that every other component
(client, server, worker, algorithm layer) builds on. It describes *what* a study
is made of, and leaves *how* it is run to those components: the specs here are
plain, validated configuration objects that travel over the wire unchanged.

To keep it usable everywhere, the package is kept pure: it requires only
minimal dependencies and performs no I/O or networking, so it can be imported
by any component without pulling in extra requirements. Turning a spec into a
working sampler or pruner is one of the jobs of :mod:`intelloptim.algorithms`.

Sub-modules:

- :mod:`intelloptim.core.parameters` — specs for the parameters of the search
  space (:class:`~intelloptim.core.parameters.FloatSpec`,
  :class:`~intelloptim.core.parameters.IntSpec`,
  :class:`~intelloptim.core.parameters.CategoricalSpec`).
- :mod:`intelloptim.core.samplers` — specs for the sampling strategies and the
  union that accepts any of them.
- :mod:`intelloptim.core.pruners` — specs for the early-stopping strategies and
  the union that accepts any of them.
- :mod:`intelloptim.core.objective` — the registry that attaches a study
  specification to a named objective function.

Nothing is re-exported here: import from the sub-module that defines the name.
"""
