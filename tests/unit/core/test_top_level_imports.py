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

    assert intopt.pruners.NopPrunerSpec().kind.value == "nop"
    assert intopt.pruners.MedianPrunerSpec(n_warmup_steps=3).n_warmup_steps == 3


def test_params_aliases_point_to_the_core_specs() -> None:
    import intelloptim as intopt
    from intelloptim.core import parameters

    assert intopt.params.Float is parameters.FloatSpec
    assert intopt.params.Int is parameters.IntSpec
    assert intopt.params.Categorical is parameters.CategoricalSpec
    public = {n for n in dir(intopt.params) if not n.startswith("_")}
    assert public == set(intopt.params.__all__)


def test_samplers_and_pruners_are_the_core_packages() -> None:
    import intelloptim as intopt
    from intelloptim.core import pruners, samplers

    assert intopt.samplers is samplers
    assert intopt.pruners is pruners


def test_samplers_namespace_classical() -> None:
    import intelloptim as intopt

    s = intopt.samplers.TPESamplerSpec(seed=42)
    assert s.seed == 42
    assert s.runs_remote is False


def test_samplers_namespace_quantum_constructible() -> None:
    import intelloptim as intopt

    sim = intopt.samplers.SimQuantumGausSamplerSpec(n_qubits=4)
    assert sim.n_qubits == 4 and sim.runs_remote is True
    hardware = intopt.samplers.QuantumMixtureSamplerSpec(
        provider={"provider": "ibm_runtime"}, shots=100, max_total_shots=10**7
    )
    assert hardware.runs_remote is True


def test_objective_decorator() -> None:
    import intelloptim as intopt
    from intelloptim.core.objective import (
        ObjectiveAlreadyRegisteredError,
        _clear_registry_for_tests,
        get_study_spec,
    )

    _clear_registry_for_tests()
    spec = {"study_name": "toplevel", "direction": "minimize"}

    @intopt.objective("toplevel_toy", study_spec=spec)
    def f(trial):  # noqa: ARG001 - signature placeholder
        return 0.0

    assert get_study_spec("toplevel_toy") == spec
    with pytest.raises(ObjectiveAlreadyRegisteredError):
        intopt.objective("toplevel_toy")(lambda trial: 1.0)
    _clear_registry_for_tests()


def test_trial_is_client_side_wrapper() -> None:
    """``intopt.client.Trial`` is the client-side wrapper that exposes
    ``sampled_*`` accessors for use inside user objectives."""
    import intelloptim as intopt

    t = intopt.client.Trial(trial_id="t1", number=0, params={"x": 0.5})
    assert t.sampled_float("x") == 0.5
    assert t.trial_id == "t1"


@pytest.mark.parametrize("name", ["client", "algorithms"])
def test_submodules_are_imported_on_first_access(name: str) -> None:
    """``intopt.client`` and ``intopt.algorithms`` are loaded by
    ``__getattr__``; an unrelated attribute still raises
    :class:`AttributeError`."""
    import intelloptim as intopt

    module = getattr(intopt, name)
    assert module.__name__ == f"intelloptim.{name}"
    with pytest.raises(AttributeError):
        _ = intopt.does_not_exist


def test_client_objects_are_not_on_the_top_level() -> None:
    import intelloptim as intopt

    for name in ("Client", "Trial", "TrialPruned", "TrialState"):
        assert not hasattr(intopt, name)
        assert name not in intopt.__all__
    assert intopt.client.Client.__name__ == "Client"


def test_importing_intelloptim_does_not_load_the_client_or_algorithms() -> None:
    import subprocess
    import sys

    code = (
        "import sys, intelloptim; "
        "print(any(m.startswith(('intelloptim.client', 'intelloptim.algorithms', "
        "'httpx', 'optuna')) for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
