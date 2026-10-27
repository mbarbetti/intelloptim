"""Trial lifecycle states — wire values and serialization."""

from __future__ import annotations

import pytest

from intelloptim.core.trial import TrialState


def test_wire_values() -> None:
    assert {s.name: s.value for s in TrialState} == {
        "WAITING": "waiting",
        "RUNNING": "running",
        "COMPLETE": "complete",
        "PRUNED": "pruned",
        "FAILED": "failed",
    }


@pytest.mark.parametrize("state", list(TrialState))
def test_round_trip_from_wire_value(state: TrialState) -> None:
    assert TrialState(state.value) is state


def test_is_a_string() -> None:
    assert TrialState.COMPLETE == "complete"
