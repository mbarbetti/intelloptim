"""Runtime protocols (structural interfaces) for the algorithm layer.

A protocol describes the methods and attributes an object must provide, without
requiring it to inherit from any intelloptim class. Sampler implementations that
manage their own state, such as quantum or other custom strategies, satisfy
:class:`SamplerPlugin` to take part in the ask/tell loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Protocol

from intelloptim.core.samplers.base import SamplerKind

if TYPE_CHECKING:
    from intelloptim.core.parameters import ParameterSpec


class SamplerPlugin(Protocol):
    """Runtime contract for a stateful sampler implementation.

    A sampler that satisfies this protocol proposes parameter values on request
    (:meth:`ask`), learns from the results it is given (:meth:`tell`), and can
    reconstruct its internal state from past trials (:meth:`rebuild`).

    Attributes:
        kind (SamplerKind): Identifier of the sampling strategy, matching the
            ``kind`` of the corresponding sampler spec.
        runs_remote (bool): Whether the sampler runs on an optimizer worker
            rather than inline in the orchestrator process.
    """

    kind: SamplerKind
    runs_remote: bool

    def ask(self, parameters: "List[ParameterSpec]") -> Dict[str, Any]:
        """Propose the next set of parameter values to evaluate.

        Args:
            parameters (List[ParameterSpec]): The search space to sample from.

        Returns:
            Dict[str, Any]: Mapping from parameter name to suggested value.
        """
        ...

    def tell(self, params: Dict[str, Any], value: float) -> None:
        """Report an evaluated trial so the sampler can update its state.

        Args:
            params (Dict[str, Any]): ParameterSpec values that were evaluated.
            value (float): Objective value obtained for *params*.
        """
        ...

    def rebuild(
        self,
        history: "List[tuple[Dict[str, Any], float]]",
    ) -> None:
        """Recompute internal state from scratch using past trials.

        May be a no-op for stateless samplers.

        Args:
            history (List[tuple[Dict[str, Any], float]]): Pairs of evaluated
                parameter values and their objective value.
        """
        ...


__all__ = ["SamplerPlugin"]
