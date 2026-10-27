"""Parameter declarations — validation and serialization."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from intelloptim.core.parameters import (
    CategoricalSpec,
    FloatSpec,
    IntSpec,
    ParameterKind,
)


class TestFloat:
    def test_basic(self) -> None:
        p = FloatSpec("lr", 1e-5, 1.0)
        assert p.name == "lr"
        assert p.low == 1e-5
        assert p.high == 1.0
        assert p.log is False
        assert p.step is None
        assert p.kind is ParameterKind.FLOAT

    def test_log_scale(self) -> None:
        p = FloatSpec("lr", 1e-5, 1.0, log=True)
        assert p.log is True

    def test_high_must_exceed_low(self) -> None:
        with pytest.raises(ValidationError, match="high"):
            FloatSpec("x", 1.0, 0.0)
        with pytest.raises(ValidationError, match="high"):
            FloatSpec("x", 1.0, 1.0)

    def test_log_requires_positive_low(self) -> None:
        with pytest.raises(ValidationError, match="log=True requires low > 0"):
            FloatSpec("x", 0.0, 1.0, log=True)
        with pytest.raises(ValidationError, match="log=True requires low > 0"):
            FloatSpec("x", -1.0, 1.0, log=True)

    def test_step_and_log_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError, match="step and log=True"):
            FloatSpec("x", 1e-5, 1.0, log=True, step=0.01)

    def test_step_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="step must be > 0"):
            FloatSpec("x", 0.0, 1.0, step=0.0)
        with pytest.raises(ValidationError, match="step must be > 0"):
            FloatSpec("x", 0.0, 1.0, step=-0.1)

    @pytest.mark.parametrize(
        "low,high",
        [
            (-math.inf, 0.0),
            (0.0, math.inf),
            (math.nan, 1.0),
            (0.0, math.nan),
        ],
    )
    def test_bounds_must_be_finite(self, low: float, high: float) -> None:
        with pytest.raises(ValidationError):
            FloatSpec("x", low, high)

    def test_log_with_infinite_high_rejected(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            FloatSpec("x", 1.0, math.inf, log=True)

    def test_step_must_be_finite(self) -> None:
        with pytest.raises(ValidationError, match="step must be > 0 and finite"):
            FloatSpec("x", 0.0, 1.0, step=math.inf)

    def test_step_not_dividing_the_range_is_allowed(self) -> None:
        # The effective upper bound is the largest low + k * step <= high.
        p = FloatSpec("x", 0.0, 1.0, step=0.3)
        assert (p.low, p.high, p.step) == (0.0, 1.0, 0.3)

    def test_serialises_round_trip(self) -> None:
        p = FloatSpec("lr", 1e-5, 1.0, log=True)
        dumped = p.model_dump(mode="json")
        revived = FloatSpec(**{k: v for k, v in dumped.items() if k != "kind"})
        assert revived == p


class TestInt:
    def test_basic(self) -> None:
        p = IntSpec("n", 1, 10)
        assert (p.name, p.low, p.high, p.step) == ("n", 1, 10, 1)
        assert p.kind is ParameterKind.INT

    def test_step(self) -> None:
        p = IntSpec("n", 1, 10, step=2)
        assert p.step == 2

    def test_high_must_be_at_least_low(self) -> None:
        with pytest.raises(ValidationError, match="high"):
            IntSpec("n", 5, 4)

    def test_equal_bounds_allowed_for_int(self) -> None:
        # A constant integer is a degenerate but valid spec.
        p = IntSpec("n", 7, 7)
        assert p.low == p.high == 7

    def test_log_requires_unit_step(self) -> None:
        with pytest.raises(ValidationError, match="log=True requires step == 1"):
            IntSpec("n", 1, 10, log=True, step=2)
        assert IntSpec("n", 1, 10, log=True).step == 1

    def test_step_not_dividing_the_range_is_allowed(self) -> None:
        # The effective upper bound is the largest low + k * step <= high.
        p = IntSpec("n", 1, 10, step=4)
        assert (p.low, p.high, p.step) == (1, 10, 4)

    def test_step_must_be_at_least_one(self) -> None:
        with pytest.raises(ValidationError, match="step must be >= 1"):
            IntSpec("n", 1, 10, step=0)

    def test_log_requires_low_at_least_one(self) -> None:
        with pytest.raises(ValidationError, match="log=True requires low >= 1"):
            IntSpec("n", 0, 10, log=True)


class TestCategorical:
    def test_basic(self) -> None:
        p = CategoricalSpec("activation", ["relu", "tanh", "gelu"])
        assert p.choices == ["relu", "tanh", "gelu"]
        assert p.kind is ParameterKind.CATEGORICAL

    def test_choices_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            CategoricalSpec("x", [])

    def test_mixed_scalar_types_allowed(self) -> None:
        p = CategoricalSpec("x", [1, "two", 3.5, None, False])
        assert p.choices == [1, "two", 3.5, None, False]

    @pytest.mark.parametrize("bad", [[1, 2], (1, 2), {"a": 1}, object()])
    def test_non_scalar_choices_rejected(self, bad: object) -> None:
        with pytest.raises(ValidationError, match="must be None, bool, int"):
            CategoricalSpec("x", ["ok", bad])  # type: ignore[list-item]

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_float_choices_rejected(self, bad: float) -> None:
        with pytest.raises(ValidationError, match="finite"):
            CategoricalSpec("x", [1.0, bad])

    @pytest.mark.parametrize(
        "choices",
        [["a", "b", "a"], [1, 1], [1, True], [0, False], [1, 1.0], [None, None]],
    )
    def test_duplicate_or_equal_choices_rejected(self, choices: list) -> None:  # type: ignore[type-arg]
        with pytest.raises(ValidationError, match="duplicate choices"):
            CategoricalSpec("x", choices)

    def test_choices_are_copied(self) -> None:
        source = ["a", "b"]
        p = CategoricalSpec("x", source)
        source.append("c")
        assert p.choices == ["a", "b"]
