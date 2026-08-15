"""Resolve a :class:`~intelloptim.core.samplers.base.Sampler` spec to a
concrete :class:`optuna.samplers.BaseSampler` instance.

The resolver lives in the public algorithm layer (not in ``core/``) — it
imports Optuna's algorithm classes, which we intentionally keep out of
``core/``. Each spec ``kind`` maps to one Optuna sampler class via the
registry below; adding a new built-in sampler is a single-line change here
plus a spec class in ``core.samplers.classical``.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Callable, Dict, Union

import optuna

from intelloptim._internal.logging import get_logger
from intelloptim.core.samplers.base import SamplerKind
from intelloptim.core.samplers.classical import TPE
from intelloptim.core.samplers.quantum._stubs import QuantumQAOA, QuantumVQE
from intelloptim.core.samplers.remote import RemoteRandom

_log = get_logger(__name__)

ConcreteSampler = Union[TPE, RemoteRandom, QuantumQAOA, QuantumVQE]
"""Concrete (non-abstract) sampler spec union accepted by :func:`resolve_sampler`.

Mirrors :data:`intelloptim._internal.schemas.SamplerSpec` but without the
pydantic ``Annotated[..., Field(discriminator=...)]`` wrapping, so the
type is usable in this module's function signatures without pulling in
schemas.
"""


class SamplerNotImplementedError(NotImplementedError):
    """Raised when no plugin is registered for a sampler kind.

    Carries a pointer to the plugin documentation so the user knows what
    to install or implement next.
    """


def _resolve_tpe(spec: TPE) -> optuna.samplers.BaseSampler:
    return optuna.samplers.TPESampler(
        seed=spec.seed,
        n_startup_trials=spec.n_startup_trials,
        n_ei_candidates=spec.n_ei_candidates,
        multivariate=spec.multivariate,
    )


def _resolve_remote_random(spec: RemoteRandom) -> optuna.samplers.BaseSampler:
    """RemoteRandom is algorithmically Random; the "remote" part is the
    Celery dispatch, not the sampling logic. The orchestrator and the
    optimizer worker both call this resolver — the only difference is
    *where* the resulting sampler runs.
    """
    return optuna.samplers.RandomSampler(seed=spec.seed)


def _resolve_quantum_unimplemented(
    spec: QuantumQAOA | QuantumVQE,
) -> optuna.samplers.BaseSampler:
    raise SamplerNotImplementedError(
        f"No runtime implementation registered for sampler kind "
        f"{spec.kind.value!r}. Quantum samplers ship as a plugin "
        f"contract only — see docs/samplers/quantum_plugin.md to "
        f"install or implement one, then register it under the "
        f"'intelloptim.samplers' entry-point group."
    )


_RESOLVERS: Dict[SamplerKind, Callable[[Any], optuna.samplers.BaseSampler]] = {
    SamplerKind.TPE: _resolve_tpe,
    SamplerKind.REMOTE_RANDOM: _resolve_remote_random,
    SamplerKind.QUANTUM_QAOA: _resolve_quantum_unimplemented,
    SamplerKind.QUANTUM_VQE: _resolve_quantum_unimplemented,
}


def load_sampler_plugins() -> None:
    """Discover and register sampler plugins from the ``intelloptim.samplers``
    entry-point group.

    Each entry point must be a callable with the signature
    ``(spec: Any) -> optuna.samplers.BaseSampler``. The entry-point name must
    match a :class:`~intelloptim.core.samplers.base.SamplerKind` value
    (e.g. ``"quantum_qaoa"``); unrecognised names are silently skipped so that
    a newer plugin package installed against an older server does not crash
    startup.

    Call once during server startup (the private ``intelloptim-server``
    invokes this from its FastAPI lifespan automatically).
    """
    eps = entry_points(group="intelloptim.samplers")
    for ep in eps:
        try:
            kind = SamplerKind(ep.name)
        except ValueError:
            _log.warning(
                "skipping sampler plugin with unknown kind",
                extra={"entry_point": ep.name},
            )
            continue
        try:
            resolver = ep.load()
        except Exception as exc:
            _log.warning(
                "failed to load sampler plugin",
                extra={"entry_point": ep.name, "error": str(exc)},
            )
            continue
        _RESOLVERS[kind] = resolver
        _log.info("sampler plugin registered", extra={"kind": ep.name})


def resolve_sampler(spec: ConcreteSampler) -> optuna.samplers.BaseSampler:
    """Instantiate the Optuna sampler corresponding to *spec*.

    Args:
        spec: A declarative sampler spec (built-in or registered plugin).

    Returns:
        A fresh Optuna :class:`~optuna.samplers.BaseSampler` instance
        ready to plug into ``optuna.create_study(...)``.

    Raises:
        SamplerNotImplementedError: If the spec is for a sampler kind
            with no registered runtime implementation (currently the
            quantum stubs).
    """
    try:
        resolver = _RESOLVERS[spec.kind]
    except KeyError as exc:
        raise SamplerNotImplementedError(
            f"Unknown sampler kind: {spec.kind!r}"
        ) from exc
    return resolver(spec)


__all__ = ["SamplerNotImplementedError", "load_sampler_plugins", "resolve_sampler"]
