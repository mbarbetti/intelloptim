"""Runtime protocols (structural interfaces) for the algorithm layer.

Plugin authors implement these protocols to register custom samplers via the
``intelloptim.samplers`` entry-point group. See
``docs/samplers/quantum_plugin.md`` for the full plugin contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Protocol

from intelloptim.core.samplers.base import SamplerKind

if TYPE_CHECKING:
    from intelloptim.core.parameters import Parameter


class SamplerPlugin(Protocol):
    """Runtime contract every sampler implementation must satisfy.

    Implementations are discovered via the ``intelloptim.samplers``
    entry-point group; the server resolves a
    :class:`~intelloptim.core.samplers.base.BaseSampler` spec to a
    :class:`SamplerPlugin` instance at study-resolution time.

    Methods are intentionally narrow: :func:`ask` returns the next parameter
    dict; :func:`tell` reports the resulting value back so the sampler can
    update; :func:`rebuild` recomputes internal state from the accumulated
    history (may be a no-op for stateless samplers).
    """

    kind: SamplerKind
    runs_remote: bool

    def ask(self, parameters: "List[Parameter]") -> Dict[str, Any]:
        """Return the next set of parameter values to evaluate."""
        ...

    def tell(self, params: Dict[str, Any], value: float) -> None:
        """Report the objective value for *params* to update internal state."""
        ...

    def rebuild(
        self,
        history: "List[tuple[Dict[str, Any], float]]",
    ) -> None:
        """Recompute internal state from scratch using *history*."""
        ...


__all__ = ["SamplerPlugin"]
