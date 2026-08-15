"""Trial domain types — defaults and serialization."""

from __future__ import annotations

from intelloptim.core.trial import Trial, TrialState


def test_default_state_is_running() -> None:
    t = Trial(id="trial-1", number=0, params={"x": 0.5})
    assert t.state is TrialState.RUNNING
    assert t.value is None
    assert t.intermediate_values == {}


def test_complete_with_value() -> None:
    t = Trial(
        id="trial-2",
        number=1,
        params={"x": 0.3},
        state=TrialState.COMPLETE,
        value=0.42,
    )
    assert t.value == 0.42


def test_round_trip() -> None:
    t = Trial(
        id="trial-3",
        number=2,
        params={"x": 1.0, "k": "a"},
        state=TrialState.PRUNED,
        intermediate_values={1: 0.5, 2: 0.3},
    )
    revived = Trial.model_validate(t.model_dump(mode="json"))
    assert revived == t
