"""intelloptim — black-box optimization client, algorithms and wire protocol.

The recommended import alias is::

    import intelloptim as intopt

Ready to use after the import:

- ``intopt.params``: the parameters that define a search space.
- ``intopt.samplers`` and ``intopt.pruners``: the specs that configure how a
  study samples and prunes.
- ``intopt.objective``: decorator that registers an objective, optionally with
  its study spec, so that ``intelloptim-worker`` can run it.

Loaded on first access:

- ``intopt.client``: the SDK that talks to a server (``Client``, ``Trial``,
  ``TrialState``, the ``errors``).
- ``intopt.algorithms``: resolves specs to Optuna samplers and pruners. It is
  meant for servers and sampler plugins, and running a study does not need it.
"""

import importlib
from typing import TYPE_CHECKING, Any

from intelloptim.core import pruners, samplers
from intelloptim.core.objective import objective
from intelloptim.version import __version__

from . import params

if TYPE_CHECKING:
    from intelloptim import client

_LAZY_SUBMODULES = frozenset({"algorithms", "client"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "client",
    "objective",
    "params",
    "pruners",
    "samplers",
]
