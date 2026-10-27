"""Pruner declarations — defaults and field validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intelloptim.core.pruners import (
    MedianPrunerSpec,
    NopPrunerSpec,
    PercentilePrunerSpec,
    PrunerKind,
    ThresholdPrunerSpec,
)


class TestNopPruner:
    def test_kind(self) -> None:
        assert NopPrunerSpec().kind is PrunerKind.NOP

    def test_serialises(self) -> None:
        dumped = NopPrunerSpec().model_dump(mode="json")
        assert dumped == {"kind": "nop"}


class TestMedianPruner:
    def test_defaults(self) -> None:
        p = MedianPrunerSpec()
        assert p.kind is PrunerKind.MEDIAN
        assert p.n_startup_trials == 5
        assert p.n_warmup_steps == 0
        assert p.interval_steps == 1

    def test_custom(self) -> None:
        p = MedianPrunerSpec(
            n_startup_trials=20,
            n_warmup_steps=3,
            interval_steps=2,
        )
        assert (p.n_startup_trials, p.n_warmup_steps, p.interval_steps) == (
            20,
            3,
            2,
        )


class TestFieldValidation:
    @pytest.mark.parametrize("spec", [MedianPrunerSpec, PercentilePrunerSpec])
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_startup_trials": -1},
            {"n_warmup_steps": -1},
            {"interval_steps": 0},
            {"n_min_trials": 0},
        ],
    )
    def test_median_and_percentile_reject_invalid(
        self, spec: type, kwargs: dict  # type: ignore[type-arg]
    ) -> None:
        with pytest.raises(ValidationError):
            spec(**kwargs)

    @pytest.mark.parametrize("percentile", [-0.1, 100.1, 150.0])
    def test_percentile_must_be_in_range(self, percentile: float) -> None:
        with pytest.raises(ValidationError):
            PercentilePrunerSpec(percentile=percentile)

    @pytest.mark.parametrize("percentile", [0.0, 50.0, 100.0])
    def test_percentile_boundaries(self, percentile: float) -> None:
        assert PercentilePrunerSpec(percentile=percentile).percentile == percentile

    def test_threshold_needs_a_bound(self) -> None:
        with pytest.raises(ValidationError, match="at least one of lower or upper"):
            ThresholdPrunerSpec()

    def test_threshold_lower_must_be_below_upper(self) -> None:
        with pytest.raises(ValidationError, match="must be <"):
            ThresholdPrunerSpec(lower=5.0, upper=1.0)
        with pytest.raises(ValidationError, match="must be <"):
            ThresholdPrunerSpec(lower=1.0, upper=1.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_threshold_bounds_must_be_finite(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            ThresholdPrunerSpec(lower=bad)

    def test_threshold_one_sided_is_valid(self) -> None:
        assert ThresholdPrunerSpec(upper=2.0).lower is None
        assert ThresholdPrunerSpec(lower=0.0).upper is None

    def test_threshold_step_fields(self) -> None:
        with pytest.raises(ValidationError):
            ThresholdPrunerSpec(upper=1.0, interval_steps=0)
        with pytest.raises(ValidationError):
            ThresholdPrunerSpec(upper=1.0, n_warmup_steps=-1)

    def test_resolvers_accept_every_valid_pruner(self) -> None:
        from intelloptim.algorithms.pruners import resolve_pruner

        for spec in (
            NopPrunerSpec(),
            MedianPrunerSpec(),
            PercentilePrunerSpec(percentile=0.0),
            ThresholdPrunerSpec(lower=0.0),
        ):
            resolve_pruner(spec)
