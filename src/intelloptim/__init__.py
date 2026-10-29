"""intelloptim — distributed black-box optimization service.

The recommended import alias is::

    import intelloptim as intopt

Public surface:

- ``intopt.Client`` — REST/WS client (imported on first use).
- ``intopt.Trial`` — client-side trial wrapper whose ``sampled_*`` accessors
  read the values the server sampled, used inside the user's objective.
- ``intopt.TrialPruned`` — raised by an objective to end a pruned trial.
- ``intopt.TrialState`` — lifecycle states for trials.
- ``intopt.objective`` — decorator that registers a named callable for the
  optional :command:`intelloptim-worker` evaluator runner.
- ``intopt.params`` / ``intopt.pruners`` / ``intopt.samplers`` — declarative
  specs used to build a study.
"""

from typing import TYPE_CHECKING, Any

from intelloptim._internal.schemas import TrialState
from intelloptim.core import pruners, samplers
from intelloptim.core.objective import objective
from intelloptim.version import __version__

from . import algorithms, params

if TYPE_CHECKING:
    from intelloptim.client.api import Client
    from intelloptim.client.errors import TrialPruned
    from intelloptim.client.trial import Trial


def __getattr__(name: str) -> Any:
    if name == "Client":
        from intelloptim.client.api import Client as _Client

        return _Client
    if name == "Trial":
        from intelloptim.client.trial import Trial as _Trial

        return _Trial
    if name == "TrialPruned":
        from intelloptim.client.errors import TrialPruned as _TrialPruned

        return _TrialPruned
    raise AttributeError(f"module 'intelloptim' has no attribute {name!r}")


__all__ = [
    "__version__",
    "Client",
    "Trial",
    "TrialPruned",
    "TrialState",
    "objective",
    "params",
    "pruners",
    "samplers",
    "algorithms",
]
