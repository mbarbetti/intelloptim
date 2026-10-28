"""Spec-to-Optuna resolution and plugin discovery for samplers.

Two public names are exported:

- :func:`resolve_sampler` — instantiate an Optuna sampler from a
  :data:`~intelloptim.core.samplers.SamplerSpec`.
- :func:`load_sampler_plugins` — discover and register plugins from the
  ``intelloptim.samplers`` entry-point group (called once at server startup).

Each :class:`~intelloptim.core.samplers.SamplerKind` maps to a private resolver
that forwards the spec's validated fields to the matching strategy;
``_RESOLVERS`` and the ``_resolve_*`` functions are implementation details.
The server calls :func:`resolve_sampler` when it builds a study.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from importlib.util import find_spec
from typing import Any, Callable, Dict, Tuple

import optuna

from intelloptim._internal.logging import get_logger
from intelloptim.algorithms.backend import (
    OWGSampler,
    SASampler,
)
from intelloptim.algorithms.errors import (
    SamplerDependencyError,
    SamplerNotImplementedError,
)
from intelloptim.core.samplers import (
    GPSamplerSpec,
    NSGAIISamplerSpec,
    OWGSamplerSpec,
    QMCSamplerSpec,
    RandomSamplerSpec,
    SamplerKind,
    SamplerSpec,
    SASamplerSpec,
    TPESamplerSpec,
)

_log = get_logger(__name__)

# Optional packages that some Optuna-backed samplers import lazily, with the
# extra of ``intelloptim`` that installs them.
_OPTIONAL_DEPENDENCIES: Dict[SamplerKind, Tuple[Tuple[str, ...], str]] = {
    SamplerKind.GP: (("torch", "scipy"), "gp"),
    SamplerKind.QMC: (("scipy",), "qmc"),
    SamplerKind.SIM_QUANTUM_GAUS: (("qiskit",), "quantum"),
    SamplerKind.SIM_QUANTUM_MIXTURE: (("qiskit",), "quantum"),
}

# Hardware samplers depend on their provider, not on their kind: a plugin
# provider brings (and reports) its own requirements.
_PROVIDER_DEPENDENCIES: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "ibm_runtime": (("qiskit", "qiskit_ibm_runtime"), "quantum-ibm"),
}


def _check_dependencies(spec: Any) -> None:
    """Raise :class:`SamplerDependencyError` if the sampler lacks a package."""
    kind = spec.kind
    modules, extra = _OPTIONAL_DEPENDENCIES.get(kind, ((), ""))
    provider = getattr(spec, "provider", None)
    if provider is not None:
        modules, extra = _PROVIDER_DEPENDENCIES.get(provider.provider, ((), ""))
    missing = [m for m in modules if find_spec(m) is None]
    if missing:
        raise SamplerDependencyError(
            f"The {kind.value!r} sampler requires {', '.join(missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} not installed. "
            f'Install them with: pip install "intelloptim[{extra}]"'
        )


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
        n_startup_trials=spec.n_startup_trials,
        seed=spec.seed,
    )


def _resolve_owg(
    spec: OWGSamplerSpec,
) -> optuna.samplers.BaseSampler:
    return OWGSampler(
        rebuild_every=spec.rebuild_every,
        initial_design=spec.initial_design,
        sharpness=spec.sharpness,
        min_std=spec.min_std,
        exploration_prob=spec.exploration_prob,
        rebuild_delay_seconds=spec.rebuild_delay_seconds,
        seed=spec.seed,
    )


def _resolve_quantum(cls_name: str) -> Callable[[Any], optuna.samplers.BaseSampler]:
    """Return a resolver that builds the named quantum sampler from its spec."""

    def resolve(spec: Any) -> optuna.samplers.BaseSampler:
        from intelloptim.algorithms.backend import quantum

        return getattr(quantum, cls_name).from_spec(spec)  # type: ignore[no-any-return]

    return resolve


_RESOLVERS: Dict[SamplerKind, Callable[[Any], optuna.samplers.BaseSampler]] = {
    SamplerKind.RANDOM: _resolve_random,
    SamplerKind.TPE: _resolve_tpe,
    SamplerKind.GP: _resolve_gp,
    SamplerKind.NSGA2: _resolve_nsga2,
    SamplerKind.QMC: _resolve_qmc,
    SamplerKind.SA: _resolve_sa,
    SamplerKind.OWG: _resolve_owg,
    SamplerKind.SIM_QUANTUM_GAUS: _resolve_quantum("SimQuantumGausSampler"),
    SamplerKind.QUANTUM_GAUS: _resolve_quantum("QuantumGausSampler"),
    SamplerKind.SIM_QUANTUM_MIXTURE: _resolve_quantum("SimQuantumMixtureSampler"),
    SamplerKind.QUANTUM_MIXTURE: _resolve_quantum("QuantumMixtureSampler"),
}


def resolve_sampler(spec: SamplerSpec) -> optuna.samplers.BaseSampler:
    """Instantiate the sampler described by *spec*.

    Args:
        spec (SamplerSpec): A validated sampler spec of any supported kind,
            built-in or provided by a registered plugin.

    Returns:
        optuna.samplers.BaseSampler: A fresh sampler instance, ready to plug
        into ``optuna.create_study(...)``.

    Raises:
        SamplerNotImplementedError: If ``spec.kind`` has no runtime
            implementation registered (currently the quantum stubs, unless a
            plugin provides one).
        SamplerDependencyError: If the sampler needs an optional package that
            is not installed (``GP``, ``QMC``, the quantum policy samplers).
    """
    try:
        resolver = _RESOLVERS[spec.kind]
    except KeyError as exc:
        raise SamplerNotImplementedError(
            f"Unknown sampler kind: {spec.kind!r}"
        ) from exc
    _check_dependencies(spec)
    return resolver(spec)


def load_sampler_plugins() -> None:
    """Register sampler plugins found in the ``intelloptim.samplers`` entry points.

    Each entry point must be a callable with the signature
    ``(spec: Any) -> optuna.samplers.BaseSampler``, and its name must match a
    :class:`~intelloptim.core.samplers.SamplerKind` value (e.g.
    ``"quantum_qaoa"``). A plugin registered for a kind that already has a
    resolver replaces it.

    Entry points with an unrecognised name or that fail to load are skipped
    with a warning, so a newer plugin package installed against an older
    server does not crash startup.

    Call once during server startup; the private ``intelloptim-server`` does
    so from its FastAPI lifespan.
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
