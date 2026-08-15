"""StudySpec validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intelloptim.core.parameters import Categorical, Float, Int
from intelloptim.core.pruners import MedianPruner, NopPruner
from intelloptim.core.samplers.classical import TPE
from intelloptim.core.samplers.quantum import QuantumQAOA
from intelloptim.core.study import Direction, StudySpec


class TestStudySpec:
    def test_minimal_valid(self) -> None:
        spec = StudySpec(
            name="demo",
            direction=Direction.MINIMIZE,
            parameters=[Float("x", -1.0, 1.0)],
            sampler=TPE(),
        )
        assert spec.name == "demo"
        assert spec.direction is Direction.MINIMIZE
        assert isinstance(spec.pruner, NopPruner)

    def test_full(self) -> None:
        spec = StudySpec(
            name="demo",
            direction=Direction.MAXIMIZE,
            parameters=[
                Float("lr", 1e-5, 1.0, log=True),
                Int("n_layers", 1, 10),
                Categorical("activation", ["relu", "tanh"]),
            ],
            sampler=TPE(seed=42),
            pruner=MedianPruner(n_warmup_steps=5),
        )
        assert len(spec.parameters) == 3
        assert isinstance(spec.pruner, MedianPruner)

    def test_quantum_sampler_accepted(self) -> None:
        spec = StudySpec(
            name="quantum-demo",
            direction=Direction.MINIMIZE,
            parameters=[Categorical("c", [0, 1])],
            sampler=QuantumQAOA(reps=2),
        )
        assert isinstance(spec.sampler, QuantumQAOA)
        assert spec.sampler.runs_remote is True

    def test_duplicate_param_names_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate parameter names"):
            StudySpec(
                name="bad",
                direction=Direction.MINIMIZE,
                parameters=[Float("x", 0.0, 1.0), Float("x", 1.0, 2.0)],
                sampler=TPE(),
            )
