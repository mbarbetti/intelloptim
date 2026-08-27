"""intelloptim — distributed black-box optimization service.

The recommended import alias is::

    import intelloptim as intopt

Public surface:

- ``intopt.Client`` — REST/WS client (lazy import; requires the ``[client]``
  extra at install time, but already shipped in this build).
- ``intopt.Trial`` — client-side trial wrapper with Optuna-shaped
  ``suggest_*`` methods, used inside the user's objective.
- ``intopt.TrialState`` — lifecycle states for trials.
- ``intopt.objective`` — decorator that registers a named callable for the
  optional :command:`intelloptim-worker` evaluator runner.
- ``intopt.params`` / ``intopt.pruners`` / ``intopt.samplers`` — declarative
  specs used to build a study.
"""

from typing import TYPE_CHECKING, Any

from . import params, algorithms
from intelloptim.core import pruners, samplers
from intelloptim.core.objective import objective
from intelloptim.core.trial import TrialState
from intelloptim.version import __version__

if TYPE_CHECKING:
    from intelloptim.client.api import Client
    from intelloptim.client.trial import Trial


def __getattr__(name: str) -> Any:
    if name == "Client":
        from intelloptim.client.api import Client as _Client

        return _Client
    if name == "Trial":
        from intelloptim.client.trial import Trial as _Trial

        return _Trial
    raise AttributeError(f"module 'intelloptim' has no attribute {name!r}")


__all__ = [
    "__version__",
    "Client",
    "Trial",
    "TrialState",
    "objective",
    "params",
    "pruners",
    "samplers",
    "algorithms",
]
