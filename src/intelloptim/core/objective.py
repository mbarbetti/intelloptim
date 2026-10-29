"""Objective registry.

The :func:`objective` decorator registers an objective callable under a name,
optionally together with the study specification it belongs to. The
convenience evaluator CLI (``intelloptim-worker evaluator``) uses the attached
specification to create or join the study without a separate spec file.

The registry is local to the process and is filled when the module that holds
the decorated functions is imported. Direct SDK users — those calling
``study.optimize(my_func, ...)`` from their own script — do not need it: they
pass the callable in directly.

Besides :func:`objective` itself, the module offers :func:`get_study_spec` and
:func:`get_study_spec_for` to read back the attached specification, by name or
from the callable. Registering a different callable under a
name that is already taken raises :class:`ObjectiveAlreadyRegisteredError`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

_REGISTRY: Dict[str, Callable[..., Any]] = {}
# Raw study-spec dicts keyed by objective name, set via the study_spec kwarg.
# Left as plain dicts so core/ stays free of _internal schema imports.
_SPEC_REGISTRY: Dict[str, Dict[str, Any]] = {}


class ObjectiveAlreadyRegisteredError(RuntimeError):
    """Raised when a different callable is registered under a taken name.

    Registering the same callable twice under the same name is not an error.
    """


def objective(
    name: str,
    *,
    study_spec: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a callable as the objective known as *name*.

    Usage::

        import intelloptim as intopt

        @intopt.objective("xgb_classifier")
        def my_objective(trial: intopt.client.Trial) -> float:
            ...

    An optional *study_spec* can be attached so that the evaluator CLI can
    create or join the study without a separate spec file. It has the same
    shape as the ``StudyResolveRequest`` wire schema and is stored as given,
    so a malformed one only surfaces when the CLI uses it::

        @intopt.objective(
            "pump_calibration",
            study_spec={
                "study_name": "pump-cal-2026-Q2",
                "direction": "minimize",
                "sampler_spec": {"kind": "tpe"},
                "parameters": [
                    {"kind": "float", "name": "x", "low": -5.0, "high": 5.0},
                ],
                "pruner_spec": {"kind": "nop"},
            },
        )
        def pump_calibration(trial: intopt.client.Trial) -> float:
            ...

    Registering the same callable again under the same name is allowed, so
    re-importing a module is harmless; a *study_spec* given the second time
    replaces the first.

    Args:
        name (str): Identifier of the objective. It must be unique within the
            process. The evaluator CLI finds the attached study spec from the
            function it imports, whatever name it was registered under.
        study_spec (Optional[Dict[str, Any]]): Raw study configuration to
            attach to the objective, returned later by :func:`get_study_spec`.
            Defaults to ``None``.

    Returns:
        Callable[[Callable[..., Any]], Callable[..., Any]]: A decorator that
        registers the function and returns it unmodified.

    Raises:
        ObjectiveAlreadyRegisteredError: When the decorator is applied and a
            different callable is already registered under *name*.
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY and _REGISTRY[name] is not func:
            raise ObjectiveAlreadyRegisteredError(
                f"objective {name!r} is already registered to "
                f"{_REGISTRY[name].__module__}.{_REGISTRY[name].__qualname__}"
            )
        _REGISTRY[name] = func
        if study_spec is not None:
            _SPEC_REGISTRY[name] = study_spec
        return func

    return _decorator


def get_study_spec(name: str) -> Optional[Dict[str, Any]]:
    """Return the study spec attached to the objective called *name*.

    Args:
        name (str): Name the objective was registered under.

    Returns:
        Optional[Dict[str, Any]]: The raw spec passed as ``study_spec``, or
        ``None`` if the objective is unknown or has no spec attached.
    """
    return _SPEC_REGISTRY.get(name)


def get_study_spec_for(func: Callable[..., Any]) -> Optional[Dict[str, Any]]:
    """Return the study spec attached to the registered callable *func*.

    Args:
        func (Callable[..., Any]): A callable registered with
            :func:`objective`, under any name.

    Returns:
        Optional[Dict[str, Any]]: The raw spec passed as ``study_spec``, or
        ``None`` if *func* is not registered or has no spec attached.
    """
    for name, registered in _REGISTRY.items():
        if registered is func:
            return _SPEC_REGISTRY.get(name)
    return None


def _clear_registry_for_tests() -> None:
    """Test-only helper to wipe the registry between cases."""
    _REGISTRY.clear()
    _SPEC_REGISTRY.clear()


__all__ = [
    "get_study_spec",
    "get_study_spec_for",
    "objective",
]
