"""Parameter declarations.

A study spec is a list of :class:`Parameter` instances, each one a
discriminated-union member tagged by its ``kind``. The same instances are
constructed by the user::

    [
        intopt.params.Float("lr", 1e-5, 1e-1, log=True),
        intopt.params.Int("n_layers", 1, 10),
        intopt.params.Categorical("activation", ["relu", "tanh"]),
    ]

serialized over the wire, and consumed by both the orchestrator (to
configure the sampler) and the client-side ``Trial.suggest_*`` methods (to
validate values returned by the server).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ParameterKind(str, Enum):
    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"


class _ParameterBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)


class Float(_ParameterBase):
    """A continuous parameter on a closed interval ``[low, high]``.

    Args:
        name: Parameter name, used as the dict key in trial params.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        log: When ``True``, the sampler explores the parameter on a log
            scale. Requires ``low > 0``.
        step: Optional discretization step (suggested values are snapped
            to ``low + k * step``). Mutually exclusive with ``log=True``.
    """

    kind: Literal[ParameterKind.FLOAT] = ParameterKind.FLOAT
    low: float
    high: float
    log: bool = False
    step: Optional[float] = None

    def __init__(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: Optional[float] = None,
        kind: ParameterKind = ParameterKind.FLOAT,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            name=name, low=low, high=high, log=log, step=step, kind=kind
        )

    @model_validator(mode="after")
    def _check_bounds(self) -> "Float":
        if not self.high > self.low:
            raise ValueError(
                f"Float({self.name!r}): high ({self.high}) must be > low ({self.low})"
            )
        if self.log and self.low <= 0:
            raise ValueError(
                f"Float({self.name!r}): log=True requires low > 0, got {self.low}"
            )
        if self.log and self.step is not None:
            raise ValueError(
                f"Float({self.name!r}): step and log=True are mutually exclusive"
            )
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Float({self.name!r}): step must be > 0, got {self.step}")
        return self


class Int(_ParameterBase):
    """An integer parameter on a closed interval ``[low, high]``.

    Args:
        name: Parameter name.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        log: When ``True``, the sampler explores the parameter on a log
            scale. Requires ``low >= 1``.
        step: Stride between candidate values (default 1).
    """

    kind: Literal[ParameterKind.INT] = ParameterKind.INT
    low: int
    high: int
    log: bool = False
    step: int = 1

    def __init__(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
        step: int = 1,
        kind: ParameterKind = ParameterKind.INT,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            name=name, low=low, high=high, log=log, step=step, kind=kind
        )

    @model_validator(mode="after")
    def _check_bounds(self) -> "Int":
        if self.high < self.low:
            raise ValueError(
                f"Int({self.name!r}): high ({self.high}) must be >= low ({self.low})"
            )
        if self.log and self.low < 1:
            raise ValueError(
                f"Int({self.name!r}): log=True requires low >= 1, got {self.low}"
            )
        if self.step < 1:
            raise ValueError(f"Int({self.name!r}): step must be >= 1, got {self.step}")
        return self


class Categorical(_ParameterBase):
    """A discrete unordered choice over a fixed set of values.

    Args:
        name: Parameter name.
        choices: Non-empty list of allowed values. Order is preserved and
            participates in the canonical hash, so reordering changes the
            study identity.
    """

    kind: Literal[ParameterKind.CATEGORICAL] = ParameterKind.CATEGORICAL
    choices: List[Any]

    def __init__(
        self,
        name: str,
        choices: List[Any],
        *,
        kind: ParameterKind = ParameterKind.CATEGORICAL,
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            name=name, choices=list(choices), kind=kind
        )

    @model_validator(mode="after")
    def _check_choices(self) -> "Categorical":
        if not self.choices:
            raise ValueError(f"Categorical({self.name!r}): choices must be non-empty")
        return self


Parameter = Annotated[Union[Float, Int, Categorical], Field(discriminator="kind")]
"""Discriminated-union alias for the three parameter kinds (tagged by ``kind``)."""

__all__ = ["Categorical", "Float", "Int", "Parameter", "ParameterKind"]
