"""Client-side ``Trial``: the values the server sampled for one evaluation.

In intelloptim's distributed model the search space is declared once, when the
study is created, and the *server* runs the sampler and ships an already
decided set of parameter values to the evaluator. The client ``Trial`` is
therefore a passive record of those values: reading them does not sample
anything, and calling an accessor twice returns the same value. The objective
only needs the name of a parameter, because its range, scale and choices are
already part of the study::

    def objective(trial: intopt.Trial) -> float:
        lr = trial.sampled_float("lr")
        n_layers = trial.sampled_int("n_layers")
        activation = trial.sampled_categorical("activation")
        return train_and_score(lr, n_layers, activation)

Coming from Optuna? Here the search space is defined once, when the study is
created, and not dynamically inside the objective as Optuna's ``suggest_*``
methods allow. Declare the parameters in
:meth:`~intelloptim.client.api.Client.create_study` and read them back with the
``sampled_*`` accessors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


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
