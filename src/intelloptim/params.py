"""Public re-exports for parameter declarations.

The classes in :mod:`intelloptim.core.parameters` are named ``FloatSpec``,
``IntSpec`` and ``CategoricalSpec`` to make clear that they are declarative
specs. Here they are exposed under shorter names for building a study::

    import intelloptim as intopt

    [
        intopt.params.Float("lr", 1e-5, 1e-1, log=True),
        intopt.params.Int("n_layers", 1, 10),
        intopt.params.Categorical("activation", ["relu", "tanh"]),
    ]
"""

from intelloptim.core.parameters import CategoricalSpec as Categorical
from intelloptim.core.parameters import FloatSpec as Float
from intelloptim.core.parameters import IntSpec as Int

__all__ = [
    "Float",
    "Int",
    "Categorical",
]
