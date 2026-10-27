"""Parameter specs for the search space.

A *parameter spec* declares one parameter of the search space: its name, its
type and its allowed range or values. A study takes a list of them, which is
sent to the server as-is and used there to define the search space. Specs are
plain configuration objects: creating one is cheap and has no side effects, and
it is validated on the spot, so a declaration the server could not turn into a
search space is rejected before the study is submitted.

Three kinds of parameter are available:

- :class:`FloatSpec` — a continuous value on a closed interval, optionally on
  a log scale or snapped to a grid. Use it for rates, scales and ratios.
- :class:`IntSpec` — an integer on a closed interval, optionally on a log scale
  or with a stride. Use it for counts such as layers, units or batch sizes.
- :class:`CategoricalSpec` — one value out of a fixed, unordered set. Use it
  for choices with no natural order, such as an activation function.

All three inherit from :class:`BaseParameterSpec`, which makes them immutable
and strict about unknown fields. Each carries a ``kind`` tag, which lets
:data:`ParameterSpec` tell them apart when a study is deserialized.

.. note::

    The ``Spec`` suffix is deliberate: it marks these classes as declarations,
    not as working objects. Users rarely need to write it, because
    :mod:`intelloptim.params` re-exports the three classes under shorter
    names, and a study is normally declared with those::

        intopt.params.Float("lr", 1e-5, 1e-1, log=True)         # FloatSpec
        intopt.params.Int("n_layers", 1, 10)                    # IntSpec
        intopt.params.Categorical("act", ["relu", "tanh"])      # CategoricalSpec

    Both names refer to the same class, so ``isinstance`` checks and error
    messages work with either. Inside the package, always use the ``Spec``
    names.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ParameterKind(str, Enum):
    """Supported parameter kinds; the value is the ``kind`` tag on the wire."""

    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"


class BaseParameterSpec(BaseModel):
    """Shared base class for all parameter spec models.

    Concrete subclasses add their own configuration fields and set the
    ``kind`` discriminator. Two guarantees apply to every spec built on this
    base: fields cannot be changed after construction (frozen), and unknown
    fields raise a validation error (strict). Together these mean a parameter
    spec always means exactly what it says — no silent defaults, no accidental
    mutation.

    Args:
        name (str): Parameter name, used as the dict key in trial params. Must
            not be empty.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)


class FloatSpec(BaseParameterSpec):
    """A continuous parameter on a closed interval ``[low, high]``.

    Both bounds must be finite and ``high`` must be strictly greater than
    ``low``. With ``log=True`` the sampler explores the interval on a log
    scale, which suits quantities that span several orders of magnitude. With
    ``step`` it suggests values on the grid ``low + k * step``; when
    ``high - low`` is not a multiple of ``step``, the effective upper bound is
    the largest grid value not exceeding ``high``.

    Args:
        name (str): Parameter name, used as the dict key in trial params.
        low (float): Lower bound (inclusive). Must be finite.
        high (float): Upper bound (inclusive). Must be finite and greater than
            ``low``.
        log (bool): When ``True``, the sampler explores the parameter on a log
            scale. Requires ``low > 0``. Defaults to ``False``.
        step (Optional[float]): Discretization step; suggested values are
            snapped to ``low + k * step``. Must be positive and finite, and is
            mutually exclusive with ``log=True``. Defaults to ``None``.
        kind (ParameterKind): Wire tag. It is set by the class and only
            accepted so that serialized parameters can be rebuilt; leave it at
            its default.

    Raises:
        ValueError: If a bound is not finite, ``high`` is not greater than
            ``low``, ``log`` is combined with a non-positive ``low`` or with
            ``step``, or ``step`` is not positive and finite.
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
    def _check_bounds(self) -> "FloatSpec":
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError(
                f"FloatSpec({self.name!r}): bounds must be finite, "
                f"got low={self.low}, high={self.high}"
            )
        if not self.high > self.low:
            raise ValueError(
                f"FloatSpec({self.name!r}): high ({self.high}) must be > "
                f"low ({self.low})"
            )
        if self.log and self.low <= 0:
            raise ValueError(
                f"FloatSpec({self.name!r}): log=True requires low > 0, "
                f"got {self.low}"
            )
        if self.log and self.step is not None:
            raise ValueError(
                f"FloatSpec({self.name!r}): step and log=True are mutually exclusive"
            )
        if self.step is not None and not (math.isfinite(self.step) and self.step > 0):
            raise ValueError(
                f"FloatSpec({self.name!r}): step must be > 0 and finite, "
                f"got {self.step}"
            )
        return self


class IntSpec(BaseParameterSpec):
    """An integer parameter on a closed interval ``[low, high]``.

    Unlike :class:`FloatSpec`, a constant parameter (``low == high``) is
    allowed. With ``log=True`` the sampler explores the interval on a log
    scale. With ``step`` it suggests values on the grid ``low + k * step``;
    when ``high - low`` is not a multiple of ``step``, the effective upper
    bound is the largest grid value not exceeding ``high``.

    Args:
        name (str): Parameter name, used as the dict key in trial params.
        low (int): Lower bound (inclusive).
        high (int): Upper bound (inclusive). Must not be below ``low``.
        log (bool): When ``True``, the sampler explores the parameter on a log
            scale. Requires ``low >= 1`` and ``step == 1``. Defaults to
            ``False``.
        step (int): Stride between candidate values, at least ``1``. Defaults
            to ``1``.
        kind (ParameterKind): Wire tag. It is set by the class and only
            accepted so that serialized parameters can be rebuilt; leave it at
            its default.

    Raises:
        ValueError: If ``high`` is below ``low``, ``step`` is below ``1``, or
            ``log`` is combined with ``low < 1`` or with ``step != 1``.
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
    def _check_bounds(self) -> "IntSpec":
        if self.high < self.low:
            raise ValueError(
                f"IntSpec({self.name!r}): high ({self.high}) must be >= "
                f"low ({self.low})"
            )
        if self.log and self.low < 1:
            raise ValueError(
                f"IntSpec({self.name!r}): log=True requires low >= 1, got {self.low}"
            )
        if self.step < 1:
            raise ValueError(
                f"IntSpec({self.name!r}): step must be >= 1, got {self.step}"
            )
        if self.log and self.step != 1:
            raise ValueError(
                f"IntSpec({self.name!r}): log=True requires step == 1, "
                f"got {self.step}"
            )
        return self


class CategoricalSpec(BaseParameterSpec):
    """A discrete, unordered choice over a fixed set of values.

    Choices must be scalars (``None``, ``bool``, ``int``, finite ``float`` or
    ``str``) and pairwise distinct. Values that compare equal, such as ``1``
    and ``True``, count as the same choice.

    Args:
        name (str): Parameter name, used as the dict key in trial params.
        choices (List[Any]): Non-empty list of allowed values. Order is
            preserved and participates in the canonical hash, so reordering
            changes the study identity.
        kind (ParameterKind): Wire tag. It is set by the class and only
            accepted so that serialized parameters can be rebuilt; leave it at
            its default.

    Raises:
        ValueError: If ``choices`` is empty, contains a non-scalar or a
            non-finite float, or contains two choices that compare equal.
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
    def _check_choices(self) -> "CategoricalSpec":
        if not self.choices:
            raise ValueError(
                f"CategoricalSpec({self.name!r}): choices must be non-empty"
            )
        for choice in self.choices:
            if choice is not None and not isinstance(choice, (bool, int, float, str)):
                raise ValueError(
                    f"CategoricalSpec({self.name!r}): choices must be None, bool, "
                    f"int, float or str, got {choice!r} ({type(choice).__name__})"
                )
            if isinstance(choice, float) and not math.isfinite(choice):
                raise ValueError(
                    f"CategoricalSpec({self.name!r}): float choices must be "
                    f"finite, got {choice}"
                )
        for index, choice in enumerate(self.choices):
            if any(choice == other for other in self.choices[index + 1 :]):
                raise ValueError(
                    f"CategoricalSpec({self.name!r}): duplicate choices: {choice!r}"
                )
        return self


ParameterSpec = Annotated[
    Union[FloatSpec, IntSpec, CategoricalSpec],
    Field(discriminator="kind"),
]
"""Discriminated-union alias for the supported parameter kinds (tagged by ``kind``)."""

__all__ = [
    "ParameterKind",
    "BaseParameterSpec",
    "FloatSpec",
    "IntSpec",
    "CategoricalSpec",
    "ParameterSpec",
]
