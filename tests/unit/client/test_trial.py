"""Client-side ``Trial`` — ``suggest_*`` and dict-like access."""

from __future__ import annotations

import pytest

from intelloptim.client.trial import Trial


def test_suggest_float_returns_server_value() -> None:
    trial = Trial(trial_id="t-1", number=0, params={"x": 0.42})
    assert trial.suggest_float("x", -1.0, 1.0) == 0.42


def test_suggest_int_casts_to_int() -> None:
    trial = Trial(trial_id="t-2", number=1, params={"n": 7})
    assert trial.suggest_int("n", 0, 10) == 7
    assert isinstance(trial.suggest_int("n", 0, 10), int)


def test_suggest_categorical_returns_value() -> None:
    trial = Trial(trial_id="t-3", number=2, params={"act": "relu"})
    assert trial.suggest_categorical("act", ["relu", "tanh"]) == "relu"


def test_unknown_param_raises_keyerror() -> None:
    trial = Trial(trial_id="t-4", number=3, params={"x": 0.0})
    with pytest.raises(KeyError, match="not declared in this study's spec"):
        trial.suggest_float("y", 0.0, 1.0)


def test_dict_like_access() -> None:
    trial = Trial(trial_id="t-5", number=4, params={"x": 1.5, "k": "a"})
    assert trial["x"] == 1.5
    assert "x" in trial
    assert "z" not in trial
    assert sorted(trial.keys()) == ["k", "x"]


def test_value_attribute_defaults_to_none() -> None:
    trial = Trial(trial_id="t-6", number=5, params={"x": 0.0})
    assert trial.value is None

    populated = Trial(trial_id="t-7", number=6, params={"x": 1.0}, value=2.5)
    assert populated.value == 2.5
