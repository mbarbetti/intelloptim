"""Client-side ``Trial``: the values the server sampled for one evaluation.

In intelloptim's distributed model the search space is declared once, when the
study is created, and the *server* runs the sampler and ships an already
decided set of parameter values to the evaluator. The client ``Trial`` is
therefore a passive record of those values: reading them does not sample
anything, and calling an accessor twice returns the same value. The objective
only needs the name of a parameter, because its range, scale and choices are
already part of the study::

    def objective(trial: intopt.client.Trial) -> float:
        lr = trial.sampled_float("lr")
        n_layers = trial.sampled_int("n_layers")
        activation = trial.sampled_categorical("activation")
        return train_and_score(lr, n_layers, activation)

An objective that trains in steps can also stop early. It reports the
intermediate objective value of each step and asks whether the study's pruner
wants to continue; the answer comes from the server, which holds the history
of every other trial::

    for step in range(n_epochs):
        loss = train_one_epoch(...)
        trial.report(loss, step)
        if trial.should_prune():
            raise intopt.client.errors.TrialPruned

Coming from Optuna? Here the search space is defined once, when the study is
created, and not dynamically inside the objective as Optuna's ``suggest_*``
methods allow. Declare the parameters in
:meth:`~intelloptim.client.api.Client.create_study` and read them back with the
``sampled_*`` accessors.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from intelloptim.client.study import StudyHandle


class Trial:
    """One optimization trial as seen by the evaluator.

    Args:
        trial_id (str): Server-assigned opaque identifier; pass it back
            unchanged in :meth:`~intelloptim.client.study.StudyHandle.tell`.
        number (int): Zero-based ordinal within the study.
        params (Dict[str, Any]): Mapping of parameter name to the value the
            server sampled for it.
        value (Optional[float]): Final objective value once the trial has
            completed. Best-trial reads populate it; a fresh ask leaves it
            ``None``. Defaults to ``None``.
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
        # Set by ``StudyHandle.ask``; ``None`` for trials that were not asked
        # from a study (enqueued, best trial, built by hand), which cannot prune.
        self._study: Optional["StudyHandle"] = None
        self._reported: Dict[int, float] = {}
        self._sent_steps: Set[int] = set()
        self._pruned = False

    # ------------------------------------------------------------------
    # sampled_* accessors
    # ------------------------------------------------------------------

    def sampled_float(self, name: str) -> float:
        """Return the value sampled for the float parameter *name*.

        Args:
            name (str): Name of the parameter, as declared in the study.

        Returns:
            float: The sampled value. An integer value is converted.

        Raises:
            KeyError: If the study declares no parameter called *name*.
            TypeError: If the value is not a number, or is a ``bool``.
        """
        value = self._lookup(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(self._type_message(name, value, "a float"))
        return float(value)

    def sampled_int(self, name: str) -> int:
        """Return the value sampled for the integer parameter *name*.

        Args:
            name (str): Name of the parameter, as declared in the study.

        Returns:
            int: The sampled value.

        Raises:
            KeyError: If the study declares no parameter called *name*.
            TypeError: If the value is not an ``int``, which includes floats
                and ``bool``. Nothing is truncated or rounded.
        """
        value = self._lookup(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(self._type_message(name, value, "an int"))
        return value

    def sampled_categorical(self, name: str) -> Any:
        """Return the value sampled for the categorical parameter *name*.

        Args:
            name (str): Name of the parameter, as declared in the study.

        Returns:
            Any: The chosen value, as it was declared among the choices.

        Raises:
            KeyError: If the study declares no parameter called *name*.
        """
        return self._lookup(name)

    # ------------------------------------------------------------------
    # pruning
    # ------------------------------------------------------------------

    @property
    def intermediate_values(self) -> Dict[int, float]:
        """The values reported so far, keyed by step (a copy)."""
        return dict(self._reported)

    def report(self, value: float, step: int) -> None:
        """Record an intermediate objective value of this trial.

        The value is only stored locally; it reaches the server with the next
        :meth:`should_prune`, so reporting at every step costs no request.

        Args:
            value (float): Intermediate objective value, finite.
            step (int): Step at which the value was observed, non-negative.
                Each step can be reported once.

        Raises:
            ValueError: If *step* is negative or already reported, or *value*
                is not finite.
            TypeError: If *step* is not an ``int`` or *value* is not a number
                (``bool`` is rejected for both).
        """
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError(f"step must be an int, got {step!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"value must be a number, got {value!r}")
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        if not math.isfinite(value):
            raise ValueError(f"value must be finite, got {value!r}")
        if step in self._reported:
            raise ValueError(f"step {step} was already reported for this trial")
        self._reported[step] = float(value)

    def should_prune(self) -> bool:
        """Ask the server whether the study's pruner wants to stop this trial.

        The values reported since the previous call are sent in a single
        request, and the answer is the pruner's decision on all of them. If
        nothing new was reported no request is made and the previous answer
        (``False`` before the first call) is returned. The call does not count
        against the call quota.

        Returns:
            bool: ``True`` if the trial should stop. The usual response is to
            raise :class:`~intelloptim.client.errors.TrialPruned`.

        Raises:
            RuntimeError: If the trial did not come from
                :meth:`~intelloptim.client.study.StudyHandle.ask`.
            TrialNotFoundError: If the server does not know the trial, or it
                is no longer running.
            ConfigDigestMismatchError: If the study configuration on the
                server differs from this client's.
            IntelloptimAPIError: For any other error response.
        """
        pending = {
            step: value
            for step, value in self._reported.items()
            if step not in self._sent_steps
        }
        if not pending:
            return self._pruned
        if self._study is None:
            raise RuntimeError(
                "should_prune() needs a trial returned by StudyHandle.ask(); "
                "this trial is not attached to a study"
            )
        self._pruned = self._study._should_prune(self, pending)
        self._sent_steps.update(pending)
        return self._pruned

    # ------------------------------------------------------------------
    # debugging / dict-like access
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Trial(number={self.number}, trial_id={self.trial_id!r}, "
            f"params={self.params!r})"
        )

    def __getitem__(self, name: str) -> Any:
        return self._lookup(name)

    def __contains__(self, name: str) -> bool:
        return name in self.params

    def keys(self) -> List[str]:
        """Return the names of the parameters of this trial."""
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

    @staticmethod
    def _type_message(name: str, value: Any, expected: str) -> str:
        return (
            f"parameter {name!r} holds {value!r} ({type(value).__name__}), "
            f"which is not {expected}; use the sampled_* accessor that matches "
            "how the parameter was declared in the study"
        )


__all__ = ["Trial"]
