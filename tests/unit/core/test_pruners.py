"""Pruner declarations — defaults and field validation."""

from __future__ import annotations

from intelloptim.core.pruners import MedianPruner, NopPruner, PrunerKind


class TestNopPruner:
    def test_kind(self) -> None:
        assert NopPruner().kind is PrunerKind.NOP

    def test_serialises(self) -> None:
        dumped = NopPruner().model_dump(mode="json")
        assert dumped == {"kind": "nop"}


class TestMedianPruner:
    def test_defaults(self) -> None:
        p = MedianPruner()
        assert p.kind is PrunerKind.MEDIAN
        assert p.n_startup_trials == 5
        assert p.n_warmup_steps == 0
        assert p.interval_steps == 1

    def test_custom(self) -> None:
        p = MedianPruner(n_startup_trials=20, n_warmup_steps=3, interval_steps=2)
        assert (p.n_startup_trials, p.n_warmup_steps, p.interval_steps) == (
            20,
            3,
            2,
        )
