"""Objective registry — register, attach study specs, and reject duplicates."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from intelloptim.core.objective import (
    ObjectiveAlreadyRegisteredError,
    _clear_registry_for_tests,
    get_study_spec,
    get_study_spec_for,
    objective,
)

_SPEC: Dict[str, Any] = {
    "study_name": "demo",
    "direction": "minimize",
    "sampler_spec": {"kind": "tpe"},
    "parameters": [{"kind": "float", "name": "x", "low": -1.0, "high": 1.0}],
    "pruner_spec": {"kind": "nop"},
}


@pytest.fixture(autouse=True)
def _isolate_registry():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def test_decorator_returns_the_function_unmodified() -> None:
    def f(trial):  # noqa: ARG001 - signature placeholder
        return 1.0

    assert objective("toy")(f) is f


def test_duplicate_name_rejected() -> None:
    @objective("dup")
    def f1():
        return 0.0

    with pytest.raises(ObjectiveAlreadyRegisteredError, match="dup"):

        @objective("dup")
        def f2():
            return 1.0


def test_error_names_the_existing_callable() -> None:
    @objective("taken")
    def first():
        return 0.0

    with pytest.raises(ObjectiveAlreadyRegisteredError, match="first"):
        objective("taken")(lambda: 1.0)


def test_re_registering_same_callable_is_a_noop() -> None:
    @objective("idem")
    def f():
        return 0.0

    # Decorating the *same* function twice (e.g. from a re-imported
    # module) must not raise.
    assert objective("idem")(f) is f


def test_study_spec_is_attached_and_returned() -> None:
    @objective("with_spec", study_spec=_SPEC)
    def f():
        return 0.0

    assert get_study_spec("with_spec") == _SPEC


def test_study_spec_is_none_without_one_or_for_unknown_names() -> None:
    @objective("plain")
    def f():
        return 0.0

    assert get_study_spec("plain") is None
    assert get_study_spec("never-registered") is None


def test_re_registering_replaces_the_study_spec() -> None:
    @objective("replace", study_spec=_SPEC)
    def f():
        return 0.0

    newer = {**_SPEC, "study_name": "newer"}
    objective("replace", study_spec=newer)(f)
    assert get_study_spec("replace") == newer


def test_re_registering_without_a_spec_keeps_the_previous_one() -> None:
    @objective("keep", study_spec=_SPEC)
    def f():
        return 0.0

    objective("keep")(f)
    assert get_study_spec("keep") == _SPEC


def test_clear_registry_for_tests() -> None:
    @objective("gone", study_spec=_SPEC)
    def f():
        return 0.0

    _clear_registry_for_tests()
    assert get_study_spec("gone") is None

    # The name is free again, for a different callable.
    objective("gone")(lambda: 1.0)


def test_get_study_spec_for_finds_the_spec_whatever_the_registered_name() -> None:
    @objective("registered-name", study_spec=_SPEC)
    def fn(trial) -> float:  # type: ignore[no-untyped-def]
        return 0.0

    assert fn.__name__ != "registered-name"
    assert get_study_spec_for(fn) == _SPEC


def test_get_study_spec_for_is_none_without_a_spec_or_registration() -> None:
    @objective("no-spec")
    def registered(trial) -> float:  # type: ignore[no-untyped-def]
        return 0.0

    def unregistered(trial) -> float:  # type: ignore[no-untyped-def]
        return 0.0

    assert get_study_spec_for(registered) is None
    assert get_study_spec_for(unregistered) is None
