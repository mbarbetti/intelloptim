"""``StudyHandle`` — the Optuna-shaped per-study object.

A :class:`StudyHandle` wraps the client-server REST surface for a single
study. It is constructed via :meth:`Client.create_study`; users do not
instantiate it directly.

Two usage flavours coexist:

* **Explicit ask/tell** loop, useful when the evaluator code is
  imperative or needs custom error handling::

      for _ in range(50):
          trial = study.ask()
          value = my_objective(trial.params["lr"])
          study.tell(trial, value)

* **Optuna-shaped** :meth:`optimize` helper that runs the loop on the
  user's behalf and catches exceptions to mark trials ``FAILED``::

      def objective(trial):
          lr = trial.suggest_float("lr", 1e-5, 1.0, log=True)
          return train(lr)

      study.optimize(objective, n_trials=200)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from intelloptim._internal.logging import get_logger
from intelloptim._internal.schemas import (
    AskRequest,
    AskResponse,
    EnqueueRequest,
    EnqueueResponse,
    TellRequest,
)
from intelloptim.client.errors import InsufficientQuotaError
from intelloptim.client.trial import Trial
from intelloptim.core.trial import TrialState

if TYPE_CHECKING:
    from intelloptim.client.api import Client


_log = get_logger(__name__)


class StudyHandle:
    """A handle to a server-side study.

    Args:
        client: The :class:`Client` that resolved this study.
        study_id: Server-assigned content-addressed identifier.
        existed: ``True`` if the study already existed when resolved
            (i.e. another caller had already created it), ``False`` on
            first-touch creation.
        n_trials: Trials present on the server at resolve time.
        config_digest: Locally-computed digest of the canonical config;
            attached to every request so the server can detect drift.
    """

    def __init__(
        self,
        *,
        client: "Client",
        study_id: str,
        existed: bool,
        n_trials: int,
        config_digest: str,
    ) -> None:
        self._client = client
        self._study_id = study_id
        self._existed = existed
        self._initial_n_trials = n_trials
        self._config_digest = config_digest

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        return self._study_id

    @property
    def existed(self) -> bool:
        """``True`` if the study was already on the server at resolve time."""
        return self._existed

    @property
    def config_digest(self) -> str:
        return self._config_digest

    # ------------------------------------------------------------------
    # ask / tell / optimize / enqueue / stop
    # ------------------------------------------------------------------

    def ask(self, worker_id: Optional[str] = None) -> Trial:
        """Request the next trial from the server.

        Args:
            worker_id: Optional caller identifier surfaced in server
                logs and (later) the dashboard's workers view.

        Returns:
            A :class:`Trial` whose ``params`` dict carries the values
            the server's sampler proposed.

        Raises:
            StudyStoppedError: The study is stopped (HTTP 410).
            StudyRebuildingError: The study is rebuilding; the wrapped
                exception carries the server's ``Retry-After`` hint.
            ConfigDigestMismatchError: Local and server configs diverged.
        """
        body = AskRequest(
            config_digest=self._config_digest, worker_id=worker_id
        ).model_dump(mode="json")
        response = self._client._post_json(f"/studies/{self._study_id}/ask", body)
        payload = AskResponse.model_validate(response.json())
        return Trial(
            trial_id=payload.trial_id,
            number=payload.number,
            params=payload.params,
        )

    def tell(
        self,
        trial: Trial,
        value: Optional[float] = None,
        *,
        state: TrialState = TrialState.COMPLETE,
        intermediate_values: Optional[Dict[int, float]] = None,
    ) -> None:
        """Report the outcome of a trial.

        Args:
            trial: The :class:`Trial` previously returned by
                :meth:`ask`.
            value: Objective value. Required when *state* is ``COMPLETE``;
                optional for ``PRUNED`` or ``FAILED``.
            state: Final lifecycle state of the trial.
            intermediate_values: Optional per-step intermediate values
                used by pruners (Phase 1 stores but does not act on them).
        """
        if state == TrialState.COMPLETE and value is None:
            raise ValueError(
                "tell(state=COMPLETE) requires a value; pass state=TrialState.FAILED "
                "or state=TrialState.PRUNED if the trial did not produce one"
            )

        body = TellRequest(
            config_digest=self._config_digest,
            value=value,
            state=state,
            intermediate_values=intermediate_values or {},
        ).model_dump(mode="json")
        self._client._post_json(
            f"/studies/{self._study_id}/trials/{trial.trial_id}/tell", body
        )

    def enqueue(
        self,
        params: Dict[str, Any],
        value: Optional[float] = None,
    ) -> Trial:
        """Inject an externally-known trial.

        If *value* is provided the trial is recorded as ``COMPLETE``
        immediately; otherwise it is placed in ``WAITING`` and the next
        :meth:`ask` returns its parameters.
        """
        body = EnqueueRequest(
            config_digest=self._config_digest,
            params=params,
            value=value,
        ).model_dump(mode="json")
        response = self._client._post_json(f"/studies/{self._study_id}/enqueue", body)
        payload = EnqueueResponse.model_validate(response.json())
        return Trial(
            trial_id=payload.trial_id,
            number=int(payload.trial_id),
            params=dict(params),
        )

    def stop(self) -> None:
        """Transition the study to the ``STOPPED`` state."""
        self._client._post_json(f"/studies/{self._study_id}/stop", {})

    def optimize(
        self,
        objective: Callable[[Trial], float],
        n_trials: int,
        *,
        catch: Optional[type] = Exception,
    ) -> None:
        """Run *n_trials* of ``ask -> objective -> tell``.

        Exceptions raised inside *objective* are reported back to the
        server as ``FAILED`` and then re-raised unless caught by *catch*.
        Pass ``catch=None`` to disable the safety net and let trial
        failures propagate without a final ``/tell``.

        Args:
            objective: Callable taking a :class:`Trial` and returning
                the objective value (a float).
            n_trials: Number of ask/tell iterations to run from *this*
                client. Other clients sharing the same study contribute
                their own trials concurrently.
            catch: Exception class to treat as a recoverable trial
                failure. Default :class:`Exception`. Use ``catch=None``
                to disable.

        Raises:
            InsufficientQuotaError: ``n_trials`` exceeds the remaining
                monthly budget (pre-flight check; no server calls made).
        """
        who = self._client._who_am_i
        if who is not None:
            remaining = (
                who.group_calls_remaining
                if who.is_group_context
                else who.calls_remaining
            )
            if remaining is not None and n_trials > remaining:
                ctx = (
                    f"group '{who.group_name}'"
                    if who.is_group_context
                    else f"role '{who.role_name}'"
                )
                resets = (
                    f" Quota resets on {who.period_resets_at}."
                    if who.period_resets_at
                    else ""
                )
                raise InsufficientQuotaError(
                    f"Requested {n_trials} trials but only {remaining} "
                    f"ask/tell call(s) remain in your {ctx} budget this "
                    f"month.{resets} Reduce n_trials or contact your admin "
                    "to raise the quota."
                )

        for _ in range(n_trials):
            trial = self.ask()
            try:
                value = float(objective(trial))
            except Exception as exc:  # noqa: BLE001 - re-raised below
                if catch is None or not isinstance(exc, catch):
                    raise
                _log.warning("trial %s failed: %s", trial.trial_id, exc, exc_info=True)
                self.tell(trial, value=None, state=TrialState.FAILED)
                continue
            self.tell(trial, value, state=TrialState.COMPLETE)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def best_trial(self) -> Optional[Trial]:
        """Return the best completed trial, or ``None`` if none exists yet."""
        response = self._client._get(f"/studies/{self._study_id}/best_trial")
        payload = response.json()
        if not payload.get("available", False):
            return None
        return Trial(
            trial_id=payload["trial_id"],
            number=payload["number"],
            params=payload["params"],
            value=payload["value"],
        )


__all__ = ["StudyHandle"]
