"""Pruner configuration package.

Pruners control *early stopping*: at each intermediate checkpoint of a trial,
the active pruner decides whether that trial is still worth finishing. Pruning
can substantially reduce wall-clock time for a study because poorly-performing
trials are abandoned rather than run to completion.

This package exports all pruner spec classes and the :data:`PrunerSpec` type
alias. Use :data:`PrunerSpec` whenever you need to accept any pruner in a type
annotation or a pydantic model field — it covers all supported strategies and
validates the ``kind`` field automatically.

Typical usage::

    import intelloptim as intopt

    pruner_spec = intopt.pruners.MedianPrunerSpec(
        n_startup_trials=5,
        n_warmup_steps=10,
    )
    study = client.create_study(..., pruner_spec=pruner_spec)

Sub-modules:

- :mod:`intelloptim.core.pruners.base` — the shared enum and base spec class
  every concrete pruner spec inherits from.
- :mod:`intelloptim.core.pruners.optuna` — spec classes backed by Optuna's
  built-in pruning strategies.
"""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from intelloptim.core.pruners.base import PrunerKind
from intelloptim.core.pruners.optuna import (
    MedianPrunerSpec,
    NopPrunerSpec,
    PercentilePrunerSpec,
    ThresholdPrunerSpec,
)

PrunerSpec = Annotated[
    Union[
        NopPrunerSpec,
        MedianPrunerSpec,
        PercentilePrunerSpec,
        ThresholdPrunerSpec,
    ],
    Field(discriminator="kind"),
]
"""Discriminated-union alias for the supported pruner kinds (tagged by ``kind``)."""

__all__ = [
    "PrunerKind",
    "NopPrunerSpec",
    "MedianPrunerSpec",
    "PercentilePrunerSpec",
    "ThresholdPrunerSpec",
    "PrunerSpec",
]
