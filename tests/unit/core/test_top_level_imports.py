"""Top-level public surface — ``import intelloptim as intopt`` works.

Sanity that every symbol promised by the public API is reachable via the
``into`` alias without optional extras installed.
"""

from __future__ import annotations

import pytest


def test_alias_import_works() -> None:
    import intelloptim as intopt

    assert intopt.__version__


def test_params_namespace() -> None:
    import intelloptim as intopt

    p = intopt.params.Float("lr", 1e-5, 1.0, log=True)
    assert p.name == "lr"
    assert intopt.params.Int("n", 1, 5).low == 1
    assert intopt.params.Categorical("c", ["a", "b"]).choices == ["a", "b"]


def test_pruners_namespace() -> None:
    import intelloptim as intopt

    assert intopt.pruners.NopPruner().kind.value == "nop"
    assert intopt.pruners.MedianPruner(n_warmup_steps=3).n_warmup_steps == 3


def test_samplers_namespace_classical() -> None:
    import intelloptim as intopt

    s = intopt.samplers.TPE(seed=42)
    assert s.seed == 42
    assert s.runs_remote is False


def test_samplers_namespace_quantum_constructible() -> None:
    import intelloptim as intopt

    q = intopt.samplers.QuantumQAOA(reps=3)
    assert q.reps == 3
    assert q.runs_remote is True
    v = intopt.samplers.QuantumVQE(ansatz="EfficientSU2")
    assert v.ansatz == "EfficientSU2"


def test_objective_decorator() -> None:
    import intelloptim as intopt
    from intelloptim.core.objective import _clear_registry_for_tests

    _clear_registry_for_tests()

    @intopt.objective("toplevel_toy")
    def f(trial):  # noqa: ARG001 - signature placeholder
        return 0.0

    from intelloptim.core.objective import get_objective

    assert get_objective("toplevel_toy") is f
    _clear_registry_for_tests()


def test_trial_is_client_side_wrapper() -> None:
    """``intopt.Trial`` resolves lazily to the client-side wrapper that
    exposes ``suggest_*`` methods for use inside user objectives."""
    import intelloptim as intopt

    t = intopt.Trial(trial_id="t1", number=0, params={"x": 0.5})
    assert t.suggest_float("x", -1.0, 1.0) == 0.5
    assert t.trial_id == "t1"


def test_client_attribute_lazy() -> None:
    """``intopt.Client`` is lazily imported via ``__getattr__``; an unrelated
    attribute access still raises :class:`AttributeError`."""
    import intelloptim as intopt

    cls = intopt.Client
    assert cls.__name__ == "Client"
    with pytest.raises(AttributeError):
        _ = intopt.does_not_exist
