"""Spec-to-Optuna resolution and plugin discovery for samplers.

Two public names are exported:

- :func:`resolve_sampler` — instantiate an Optuna sampler from a
  :class:`~intelloptim.core.samplers.base.BaseSampler` spec.
- :func:`load_sampler_plugins` — discover and register plugins from the
  ``intelloptim.samplers`` entry-point group (called once at server startup).

The plugin protocol (:class:`~intelloptim.algorithms.protocols.SamplerPlugin`)
and runtime errors (:class:`~intelloptim.algorithms.errors.SamplerNotImplementedError`)
live in their own modules.

Adding a built-in sampler requires a spec class in
:mod:`intelloptim.core.samplers.optuna` and a single resolver entry in
``_RESOLVERS`` below.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Callable, Dict

import optuna

from intelloptim._internal.logging import get_logger
from intelloptim.algorithms.backend import SASampler
from intelloptim.algorithms.errors import SamplerNotImplementedError
from intelloptim.core.samplers import (
    GPSamplerSpec,
    NSGAIISamplerSpec,
    QMCSamplerSpec,
    RandomSamplerSpec,
    SamplerSpec,
    SamplerKind,
    SASamplerSpec,
    TPESamplerSpec,
    QuantumQAOA,
    QuantumVQE,
)

_log = get_logger(__name__)


def _resolve_random(spec: RandomSamplerSpec) -> optuna.samplers.BaseSampler:
    return optuna.samplers.RandomSampler(seed=spec.seed)


def _resolve_tpe(spec: TPESamplerSpec) -> optuna.samplers.BaseSampler:
    return optuna.samplers.TPESampler(
        seed=spec.seed,
        n_startup_trials=spec.n_startup_trials,
        n_ei_candidates=spec.n_ei_candidates,
        multivariate=spec.multivariate,
        group=spec.group,
    )


def _resolve_gp(spec: GPSamplerSpec) -> optuna.samplers.BaseSampler:
    return optuna.samplers.GPSampler(
        seed=spec.seed,
        n_startup_trials=spec.n_startup_trials,
        deterministic_objective=spec.deterministic_objective,
    )


def _resolve_nsga2(spec: NSGAIISamplerSpec) -> optuna.samplers.BaseSampler:
    return optuna.samplers.NSGAIISampler(
        population_size=spec.population_size,
        mutation_prob=spec.mutation_prob,
        crossover_prob=spec.crossover_prob,
        swapping_prob=spec.swapping_prob,
        seed=spec.seed,
    )


def _resolve_qmc(spec: QMCSamplerSpec) -> optuna.samplers.BaseSampler:
    return optuna.samplers.QMCSampler(
        qmc_type=spec.qmc_type,
        scramble=spec.scramble,
        seed=spec.seed,
    )


def _resolve_sa(spec: SASamplerSpec) -> optuna.samplers.BaseSampler:
    return SASampler(
        temperature=spec.temperature,
        cooldown_factor=spec.cooldown_factor,
        neighbor_range_factor=spec.neighbor_range_factor,
        seed=spec.seed,
    )


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
    SamplerKind.RANDOM: _resolve_random,
    SamplerKind.TPE: _resolve_tpe,
    SamplerKind.GP: _resolve_gp,
    SamplerKind.NSGA2: _resolve_nsga2,
    SamplerKind.QMC: _resolve_qmc,
    SamplerKind.SA: _resolve_sa,
    SamplerKind.QUANTUM_QAOA: _resolve_quantum_unimplemented,
    SamplerKind.QUANTUM_VQE: _resolve_quantum_unimplemented,
}


def resolve_sampler(spec: SamplerSpec) -> optuna.samplers.BaseSampler:
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


__all__ = ["resolve_sampler", "load_sampler_plugins"]
