"""Objective registry — register, look up, and reject duplicates."""

from __future__ import annotations

import pytest

from intelloptim.core.objective import (
    ObjectiveAlreadyRegisteredError,
    ObjectiveNotFoundError,
    _clear_registry_for_tests,
    get_objective,
    objective,
    registered_objectives,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def test_register_and_lookup() -> None:
    @objective("toy")
    def f(trial):  # noqa: ARG001 - signature placeholder
        return 1.0

    assert get_objective("toy") is f


def test_registered_objectives_returns_copy() -> None:
    @objective("a")
    def fa():
        return 0.0

    snapshot = registered_objectives()
    assert "a" in snapshot
    snapshot["b"] = lambda: None
    # Mutating the snapshot must not leak into the real registry.
    assert "b" not in registered_objectives()


def test_duplicate_name_rejected() -> None:
    @objective("dup")
    def f1():
        return 0.0

    with pytest.raises(ObjectiveAlreadyRegisteredError):

        @objective("dup")
        def f2():
            return 1.0


def test_re_registering_same_callable_is_a_noop() -> None:
    @objective("idem")
    def f():
        return 0.0

    # Decorating the *same* function twice (e.g. from a re-imported
    # module) must not raise.
    objective("idem")(f)
    assert get_objective("idem") is f


def test_lookup_unknown_raises() -> None:
    with pytest.raises(ObjectiveNotFoundError):
        get_objective("nope")
