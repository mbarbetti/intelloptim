"""Study spec and configuration.

A :class:`StudySpec` is the user-supplied declaration of an optimization
study: name, direction, parameter list, sampler, and pruner. It is the
input to :func:`intelloptim._internal.hashing.study_id`, which derives
the content-addressed study identifier.

The same spec, once accepted by the server, is also returned to clients
that join an existing study so they can verify they are talking about the
same configuration.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, List, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intelloptim.core.parameters import Parameter
from intelloptim.core.pruners import NopPruner, Pruner
from intelloptim.core.samplers.classical import TPE
from intelloptim.core.samplers.quantum._stubs import QuantumQAOA, QuantumVQE


class Direction(str, Enum):
    """Optimization direction for single-objective studies."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


# Sampler union here mirrors :data:`intelloptim._internal.schemas.SamplerSpec`;
# kept narrow on purpose so adding a sampler is a deliberate two-line change
# (here and in `_internal.schemas`).
StudySampler = Annotated[
    Union[TPE, QuantumQAOA, QuantumVQE], Field(discriminator="kind")
]


class StudySpec(BaseModel):
    """Declarative study specification.

    Args:
        name: Human-readable study label. Two specs with the same name
            but different parameters/sampler/pruner/direction are
            different studies (different ``study_id``).
        direction: ``"minimize"`` or ``"maximize"``.
        parameters: Ordered list of parameter declarations. Order is
            preserved and participates in the hash.
        sampler: Sampler declaration (TPE in v1; quantum stubs are
            constructible but raise on actual use).
        pruner: Pruner declaration. Defaults to :class:`NopPruner`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    direction: Direction
    parameters: List[Parameter]
    sampler: StudySampler
    pruner: Pruner = NopPruner()

    @model_validator(mode="after")
    def _check_unique_param_names(self) -> "StudySpec":
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"StudySpec({self.name!r}): duplicate parameter names: {duplicates}"
            )
        return self


__all__ = ["Direction", "StudySampler", "StudySpec"]
