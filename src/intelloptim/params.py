"""Public re-exports for parameter declarations.

Usage::

    import intelloptim as intopt

    [
        intopt.params.Float("lr", 1e-5, 1e-1, log=True),
        intopt.params.Int("n_layers", 1, 10),
        intopt.params.Categorical("activation", ["relu", "tanh"]),
    ]
"""

from intelloptim.core.parameters import (
    Categorical,
    Float,
    Int,
    Parameter,
    ParameterKind,
)

__all__ = ["Categorical", "Float", "Int", "Parameter", "ParameterKind"]
