"""Client-side ``Trial`` — ``sampled_*`` accessors and dict-like access."""

from __future__ import annotations

from typing import Any

import pytest

from intelloptim.client.trial import Trial


def _trial(**params: Any) -> Trial:
    return Trial(trial_id="t-1", number=0, params=params)


class TestSampledFloat:
    def test_returns_server_value(self) -> None:
        assert _trial(x=0.42).sampled_float("x") == 0.42

    def test_converts_an_integer_value(self) -> None:
        value = _trial(x=3).sampled_float("x")
        assert value == 3.0
        assert isinstance(value, float)

    @pytest.mark.parametrize("bad", [True, "0.5", None, [1.0]])
    def test_rejects_non_numbers(self, bad: Any) -> None:
        with pytest.raises(TypeError, match="parameter 'x' holds .* not a float"):
            _trial(x=bad).sampled_float("x")


class TestSampledInt:
    def test_returns_server_value(self) -> None:
        value = _trial(n=7).sampled_int("n")
        assert value == 7
        assert isinstance(value, int)

    @pytest.mark.parametrize("bad", [0.42, 3.0, True, "7", None])
    def test_rejects_anything_but_an_int(self, bad: Any) -> None:
        # In particular a float is never truncated: 0.42 must not become 0.
        with pytest.raises(TypeError, match="parameter 'n' holds .* not an int"):
            _trial(n=bad).sampled_int("n")


class TestSampledCategorical:
    @pytest.mark.parametrize("value", ["relu", 3, 2.5, True, False, None])
    def test_returns_any_scalar_as_is(self, value: Any) -> None:
        assert _trial(act=value).sampled_categorical("act") is value


class TestUnknownParameter:
    @pytest.mark.parametrize(
        "accessor", ["sampled_float", "sampled_int", "sampled_categorical"]
    )
    def test_raises_keyerror_listing_the_declared_names(self, accessor: str) -> None:
        trial = _trial(x=0.0, k="a")
        with pytest.raises(KeyError, match="not declared in this study's spec") as info:
            getattr(trial, accessor)("y")
        assert "['k', 'x']" in str(info.value)


class TestNoSearchSpaceInTheObjective:
    @pytest.mark.parametrize(
        "accessor,args",
        [
            ("sampled_float", (0.0, 1.0)),
            ("sampled_int", (0, 10)),
            ("sampled_categorical", (["a", "b"],)),
        ],
    )
    def test_bounds_and_choices_are_not_accepted(
        self, accessor: str, args: tuple  # type: ignore[type-arg]
    ) -> None:
        trial = _trial(x=0.5, n=3, c="a")
        name = {"sampled_float": "x", "sampled_int": "n"}.get(accessor, "c")
        with pytest.raises(TypeError):
            getattr(trial, accessor)(name, *args)

    @pytest.mark.parametrize(
        "old", ["suggest_float", "suggest_int", "suggest_categorical"]
    )
    def test_optuna_style_methods_are_gone(self, old: str) -> None:
        assert not hasattr(_trial(x=0.5), old)


def test_dict_like_access() -> None:
    trial = Trial(trial_id="t-5", number=4, params={"x": 1.5, "k": "a"})
    assert trial["x"] == 1.5
    assert "x" in trial
    assert "z" not in trial
    assert sorted(trial.keys()) == ["k", "x"]


def test_params_are_copied() -> None:
    source = {"x": 1.0}
    trial = Trial(trial_id="t-8", number=0, params=source)
    source["x"] = 2.0
    assert trial.params == {"x": 1.0}


def test_value_attribute_defaults_to_none() -> None:
    trial = Trial(trial_id="t-6", number=5, params={"x": 0.0})
    assert trial.value is None

    populated = Trial(trial_id="t-7", number=6, params={"x": 1.0}, value=2.5)
    assert populated.value == 2.5
