"""Objective registry.

Evaluator-side users register their objective callables once with the
:func:`objective` decorator. The convenience evaluator CLI
(``intelloptim-worker evaluator``) and the test harness look up
objectives by their registered name.

Direct SDK users — those calling ``study.optimize(my_func, ...)`` from
their own Python script — do not need the registry; they pass the callable
in directly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

_REGISTRY: Dict[str, Callable[..., Any]] = {}
# Raw study-spec dicts keyed by objective name, set via the study_spec kwarg.
# Left as plain dicts so core/ stays free of _internal schema imports.
_SPEC_REGISTRY: Dict[str, Dict[str, Any]] = {}


class ObjectiveAlreadyRegisteredError(RuntimeError):
    """Raised when two callables register under the same name."""


class ObjectiveNotFoundError(LookupError):
    """Raised when a name is requested that no registered objective matches."""


def objective(
    name: str,
    *,
    study_spec: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register *func* as the objective callable known as *name*.

    Usage::

        import intelloptim as intopt

        @intopt.objective("xgb_classifier")
        def my_objective(trial: intopt.Trial) -> float:
            ...

    An optional *study_spec* dict (matching the ``StudyResolveRequest``
    wire schema) can be attached so the evaluator CLI can create/join the
    study without a separate spec file::

        @intopt.objective(
            "pump-calibration",
            study_spec={
                "name": "pump-cal-2026-Q2",
                "direction": "minimize",
                "sampler": {"kind": "tpe"},
                "parameters": [
                    {"kind": "float", "name": "x", "low": -5.0, "high": 5.0},
                ],
                "pruner": {"kind": "nop"},
            },
        )
        def my_objective(trial: intopt.Trial) -> float:
            ...

    Args:
        name: The string identifier the orchestrator uses to dispatch to
            this callable. Must be unique within the worker process.
        study_spec: Optional raw dict representing the study configuration
            (same shape as ``StudyResolveRequest``). Stored in a parallel
            registry and returned by :func:`get_study_spec`.

    Returns:
        The original callable, unmodified.

    Raises:
        ObjectiveAlreadyRegisteredError: If another callable is already
            registered under the same name.
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


def get_objective(name: str) -> Callable[..., Any]:
    """Return the callable registered under *name*.

    Raises:
        ObjectiveNotFoundError: If no callable matches.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ObjectiveNotFoundError(
            f"no objective registered under {name!r}; "
            f"available: {sorted(_REGISTRY)}"
        ) from exc


def get_study_spec(name: str) -> Optional[Dict[str, Any]]:
    """Return the study-spec dict attached to objective *name*, or ``None``."""
    return _SPEC_REGISTRY.get(name)


def registered_objectives() -> Dict[str, Callable[..., Any]]:
    """Return a copy of the current registry, keyed by name."""
    return dict(_REGISTRY)


def _clear_registry_for_tests() -> None:
    """Test-only helper to wipe the registry between cases."""
    _REGISTRY.clear()
    _SPEC_REGISTRY.clear()


__all__ = [
    "ObjectiveAlreadyRegisteredError",
    "ObjectiveNotFoundError",
    "get_objective",
    "get_study_spec",
    "objective",
    "registered_objectives",
]
