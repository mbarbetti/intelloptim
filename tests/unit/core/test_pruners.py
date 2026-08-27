"""Pruner declarations — defaults and field validation."""

from __future__ import annotations

from intelloptim.core.pruners import (
    PrunerKind,
    NopPrunerSpec, 
    MedianPrunerSpec, 
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
        assert (
            p.n_startup_trials, 
            p.n_warmup_steps, 
            p.interval_steps
        ) == (
            20,
            3,
            2,
        )
