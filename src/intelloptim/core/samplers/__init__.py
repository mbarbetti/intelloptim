"""Sampler configuration package.

Samplers determine the strategy used to suggest the next set of parameter
values for each trial. The choice of sampler governs how the search space is
explored and how prior results are exploited.

This package exports all sampler spec classes and the :data:`SamplerSpec` type
alias. Use :data:`SamplerSpec` whenever you need to accept any sampler in a type
annotation or a pydantic model field — it covers all supported strategies and
validates the ``kind`` field automatically.

Typical usage::

    import intelloptim as intopt

    sampler = intopt.samplers.TPESamplerSpec(seed=42, multivariate=True)
    study = client.create_study(..., sampler=sampler)

Sub-modules:

- :mod:`intelloptim.core.samplers.base` — the shared enums, rebuild policy,
  and base spec classes every concrete sampler spec inherits from.
- :mod:`intelloptim.core.samplers.optuna` — spec classes backed by Optuna's
  built-in sampling strategies.
- :mod:`intelloptim.core.samplers.builtin` — specs for intelloptim's own
  (non-Optuna) samplers whose runtime logic lives in
  :mod:`intelloptim.algorithms.backend`.
- :mod:`intelloptim.core.samplers.quantum` — quantum sampler stubs;
  constructible but require an external plugin to execute.
"""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from intelloptim.core.samplers.base import SamplerKind
from intelloptim.core.samplers.builtin import (
    OWGSamplerSpec,
    SASamplerSpec,
)
from intelloptim.core.samplers.optuna import (
    GPSamplerSpec,
    NSGAIISamplerSpec,
    QMCSamplerSpec,
    RandomSamplerSpec,
    TPESamplerSpec,
)
from intelloptim.core.samplers.quantum import QuantumQAOA, QuantumVQE

SamplerSpec = Annotated[
    Union[
        RandomSamplerSpec,
        TPESamplerSpec,
        GPSamplerSpec,
        NSGAIISamplerSpec,
        QMCSamplerSpec,
        SASamplerSpec,
        OWGSamplerSpec,
        QuantumQAOA,
        QuantumVQE,
    ],
    Field(discriminator="kind"),
]
"""Discriminated-union alias for the supported sampler kinds (tagged by ``kind``)."""

__all__ = [
    "SamplerKind",
    "RandomSamplerSpec",
    "TPESamplerSpec",
    "GPSamplerSpec",
    "NSGAIISamplerSpec",
    "QMCSamplerSpec",
    "SASamplerSpec",
    "OWGSamplerSpec",
    "QuantumQAOA",
    "QuantumVQE",
    "SamplerSpec",
]
