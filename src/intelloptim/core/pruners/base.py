"""Pruner base types.

A *pruner* decides whether a running trial should be stopped early because
its intermediate results look unpromising compared to the rest of the study.
This module provides the two building blocks shared by every pruner:

- :class:`PrunerKind` - an enum that uniquely identifies each pruning
  strategy. Every concrete pruner carries one of these values as its ``kind``
  field, which lets the framework dispatch to the right logic without
  inspecting the full object.

- :class:`BasePrunerSpec` - the immutable base model every pruner spec inherits
  from. Inheriting from it enforces that specs are frozen (no in-place
  mutation after construction) and reject unknown fields, so a study
  definition always means exactly what it says.

Concrete pruner specs live in sibling modules and subclass :class:`BasePrunerSpec`
to add their own configuration fields.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class PrunerKind(str, Enum):
    """Supported pruning strategy identifiers."""

    NOP = "nop"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    THRESHOLD = "threshold"


class BasePrunerSpec(BaseModel):
    """Shared base class for all pruner spec models.

    Concrete subclasses add their own configuration fields. Two guarantees
    apply to every spec built on this base: fields cannot be changed after
    construction (frozen), and unknown fields raise a validation error
    (strict). Together these mean a pruner spec always means exactly what
    it says — no silent defaults, no accidental mutation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


__all__ = [
    "PrunerKind",
    "BasePrunerSpec",
]
