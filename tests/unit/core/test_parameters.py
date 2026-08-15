"""Parameter declarations — validation and serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intelloptim.core.parameters import (
    Categorical,
    Float,
    Int,
    ParameterKind,
)


class TestFloat:
    def test_basic(self) -> None:
        p = Float("lr", 1e-5, 1.0)
        assert p.name == "lr"
        assert p.low == 1e-5
        assert p.high == 1.0
        assert p.log is False
        assert p.step is None
        assert p.kind is ParameterKind.FLOAT

    def test_log_scale(self) -> None:
        p = Float("lr", 1e-5, 1.0, log=True)
        assert p.log is True

    def test_high_must_exceed_low(self) -> None:
        with pytest.raises(ValidationError, match="high"):
            Float("x", 1.0, 0.0)
        with pytest.raises(ValidationError, match="high"):
            Float("x", 1.0, 1.0)

    def test_log_requires_positive_low(self) -> None:
        with pytest.raises(ValidationError, match="log=True requires low > 0"):
            Float("x", 0.0, 1.0, log=True)
        with pytest.raises(ValidationError, match="log=True requires low > 0"):
            Float("x", -1.0, 1.0, log=True)

    def test_step_and_log_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError, match="step and log=True"):
            Float("x", 1e-5, 1.0, log=True, step=0.01)

    def test_step_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="step must be > 0"):
            Float("x", 0.0, 1.0, step=0.0)
        with pytest.raises(ValidationError, match="step must be > 0"):
            Float("x", 0.0, 1.0, step=-0.1)

    def test_serialises_round_trip(self) -> None:
        p = Float("lr", 1e-5, 1.0, log=True)
        dumped = p.model_dump(mode="json")
        revived = Float(**{k: v for k, v in dumped.items() if k != "kind"})
        assert revived == p


class TestInt:
    def test_basic(self) -> None:
        p = Int("n", 1, 10)
        assert (p.name, p.low, p.high, p.step) == ("n", 1, 10, 1)
        assert p.kind is ParameterKind.INT

    def test_step(self) -> None:
        p = Int("n", 1, 10, step=2)
        assert p.step == 2

    def test_high_must_be_at_least_low(self) -> None:
        with pytest.raises(ValidationError, match="high"):
            Int("n", 5, 4)

    def test_equal_bounds_allowed_for_int(self) -> None:
        # A constant integer is a degenerate but valid spec.
        p = Int("n", 7, 7)
        assert p.low == p.high == 7

    def test_log_requires_low_at_least_one(self) -> None:
        with pytest.raises(ValidationError, match="log=True requires low >= 1"):
            Int("n", 0, 10, log=True)


class TestCategorical:
    def test_basic(self) -> None:
        p = Categorical("activation", ["relu", "tanh", "gelu"])
        assert p.choices == ["relu", "tanh", "gelu"]
        assert p.kind is ParameterKind.CATEGORICAL

    def test_choices_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            Categorical("x", [])

    def test_mixed_types_allowed(self) -> None:
        # The categorical encoding doesn't constrain element types; the
        # canonical hash uses the values as-is.
        p = Categorical("x", [1, "two", 3.0, True])
        assert p.choices == [1, "two", 3.0, True]
