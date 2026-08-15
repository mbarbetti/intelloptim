"""Client-side ``Trial`` with Optuna-shaped ``suggest_*`` methods.

In intelloptim's distributed model the *server* runs the sampler and
ships an already-decided parameter dict to the evaluator. The client
``Trial`` is therefore a passive wrapper around that dict; its
``suggest_*`` methods exist only so the user's objective code can look
like Optuna code::

    def objective(trial: intopt.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        return train_and_score(lr)

The arguments to ``suggest_*`` are kept for API symmetry with Optuna but
are not used to re-sample — the value the server proposed is already in
``trial.params``. The orchestrator validates the declared distribution
against the stored study spec on every request, so mismatches show up as
a ``409 Conflict`` rather than as silent data drift.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


class Trial:
    """One optimization trial as seen by the evaluator.

    Args:
        trial_id: Server-assigned opaque identifier; pass back unchanged
            in :meth:`StudyHandle.tell`.
        number: Zero-based ordinal within the study.
        params: Mapping of parameter name to suggested value, as
            returned by the server.
    """

    def __init__(
        self,
        *,
        trial_id: str,
        number: int,
        params: Dict[str, Any],
        value: Optional[float] = None,
    ) -> None:
        self.trial_id = trial_id
        self.number = number
        self.params: Dict[str, Any] = dict(params)
        self.value: Optional[float] = value
        """Final objective value once the trial completes (best-trial reads
        populate it, fresh asks leave it ``None``)."""

    # ------------------------------------------------------------------
    # Optuna-shaped suggest_* methods
    # ------------------------------------------------------------------

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: Optional[float] = None,
    ) -> float:
        """Return the server-suggested value for the float parameter *name*.

        ``low`` / ``high`` / ``log`` / ``step`` are accepted for symmetry
        with Optuna's API; the suggestion has already been made by the
        server using the spec registered at study-resolve time.
        """
        del low, high, log, step  # symmetry only
        return float(self._lookup(name))

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
        step: int = 1,
    ) -> int:
        """Return the server-suggested value for the integer parameter *name*."""
        del low, high, log, step
        return int(self._lookup(name))

    def suggest_categorical(
        self,
        name: str,
        choices: Sequence[Any],
    ) -> Any:
        """Return the server-suggested value for the categorical parameter *name*."""
        del choices
        return self._lookup(name)

    # ------------------------------------------------------------------
    # debugging / dict-like access
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Trial(number={self.number}, trial_id={self.trial_id!r}, params={self.params!r})"

    def __getitem__(self, name: str) -> Any:
        return self._lookup(name)

    def __contains__(self, name: str) -> bool:
        return name in self.params

    def keys(self) -> List[str]:
        return list(self.params.keys())

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _lookup(self, name: str) -> Any:
        try:
            return self.params[name]
        except KeyError as exc:
            raise KeyError(
                f"parameter {name!r} was not declared in this study's spec; "
                f"declared: {sorted(self.params)}"
            ) from exc


__all__ = ["Trial"]
