"""Handle to a study on the server.

A :class:`StudyHandle` wraps the client-server REST surface of one study. It is
returned by :meth:`~intelloptim.client.api.Client.create_study`; users do not
instantiate it directly. The study, including its search space, is already
defined by then: the handle only asks the server for trials and reports their
results.

Two ways of running the loop are available:

- **Explicit ask/tell**, useful when the evaluator code is imperative or needs
  custom error handling::

      for _ in range(50):
          trial = study.ask()
          value = my_objective(trial.sampled_float("lr"))
          study.tell(trial, value)

- :meth:`StudyHandle.optimize`, which runs the same loop on the user's behalf
  and marks trials whose objective raised as ``FAILED``::

      def objective(trial):
          lr = trial.sampled_float("lr")
          return train(lr)

      study.optimize(objective, n_trials=200)

A trial that trains in steps can be stopped early: the objective reports each
intermediate value and checks the study's pruner, and
:meth:`StudyHandle.optimize` records the trial as ``PRUNED``::

    def objective(trial):
        for step in range(20):
            trial.report(train_one_epoch(trial.sampled_float("lr")), step)
            if trial.should_prune():
                raise intopt.TrialPruned
        return final_loss

Every request carries the study's ``config_digest``, so the server can detect
that this client and the study disagree on its configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from intelloptim._internal.logging import get_logger
from intelloptim._internal.schemas import (
    AskRequest,
    AskResponse,
    EnqueueRequest,
    EnqueueResponse,
    ShouldPruneRequest,
    ShouldPruneResponse,
    TellRequest,
    TrialState,
)
from intelloptim.client.errors import InsufficientQuotaError, TrialPruned
from intelloptim.client.trial import Trial

if TYPE_CHECKING:
    from intelloptim.client.api import Client


_log = get_logger(__name__)


class StudyHandle:
    """A handle to a study that lives on the server.

    Args:
        client (Client): The client that resolved this study, used to send
            every request.
        study_id (str): Server-assigned, content-addressed identifier of the
            study.
        existed (bool): ``True`` if the study already existed when it was
            resolved, because another caller had created it; ``False`` if
            this call created it.
        n_trials (int): Number of trials the study already had at resolve time.
        config_digest (str): Digest of the study configuration, computed
            locally and attached to every request so that the server can
            detect drift.
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
        """The content-addressed identifier of the study."""
        return self._study_id

    @property
    def existed(self) -> bool:
        """``True`` if the study was already on the server at resolve time."""
        return self._existed

    @property
    def config_digest(self) -> str:
        """The digest of the study configuration sent with every request."""
        return self._config_digest

    # ------------------------------------------------------------------
    # ask / tell / optimize / enqueue / stop
    # ------------------------------------------------------------------

    def ask(self, worker_id: Optional[str] = None) -> Trial:
        """Request the next trial from the server.

        The trial carries the parameter values the server's sampler chose; read
        them with the ``sampled_*`` accessors of :class:`Trial`.

        Args:
            worker_id (Optional[str]): Identifier of the caller, shown in the
                server logs and, later, in the dashboard's workers view.
                Defaults to ``None``.

        Returns:
            Trial: The new trial, in the running state until it is reported
            with :meth:`tell`.

        Raises:
            StudyNotFoundError: If the server has no such study.
            StudyStoppedError: If the study has been stopped.
            StudyRebuildingError: If the study is rebuilding its sampler. The
                error carries the server's ``Retry-After`` hint in
                ``retry_after_seconds``.
            ConfigDigestMismatchError: If the study configuration on the server
                differs from this client's.
            IntelloptimAPIError: For any other error response, such as an
                exhausted quota.
        """
        body = AskRequest(
            config_digest=self._config_digest, worker_id=worker_id
        ).model_dump(mode="json")
        response = self._client._post_json(f"/studies/{self._study_id}/ask", body)
        payload = AskResponse.model_validate(response.json())
        trial = Trial(
            trial_id=payload.trial_id,
            number=payload.number,
            params=payload.params,
        )
        trial._study = self
        return trial

    def tell(
        self,
        trial: Trial,
        value: Optional[float] = None,
        *,
        state: TrialState = TrialState.COMPLETE,
    ) -> None:
        """Report the outcome of a trial.

        Args:
            trial (Trial): The trial previously returned by :meth:`ask`.
            value (Optional[float]): Objective value. Required when *state* is
                ``COMPLETE``; optional for ``PRUNED`` and ``FAILED``. Defaults
                to ``None``.
            state (TrialState): Final state of the trial. Defaults to
                ``TrialState.COMPLETE``.

        Raises:
            ValueError: If *state* is ``COMPLETE`` and no *value* is given.
            TrialNotFoundError: If the server does not know the trial.
            ConfigDigestMismatchError: If the study configuration on the server
                differs from this client's.
            IntelloptimAPIError: For any other error response, such as an
                exhausted quota.
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
        ).model_dump(mode="json")
        self._client._post_json(
            f"/studies/{self._study_id}/trials/{trial.trial_id}/tell", body
        )

    def _should_prune(self, trial: Trial, values: Dict[int, float]) -> bool:
        """Send the new intermediate *values* of *trial* and return the verdict.

        Called by :meth:`Trial.should_prune`; users do not call it directly.
        """
        body = ShouldPruneRequest(
            config_digest=self._config_digest, intermediate_values=values
        ).model_dump(mode="json")
        response = self._client._post_json(
            f"/studies/{self._study_id}/trials/{trial.trial_id}/should_prune", body
        )
        return ShouldPruneResponse.model_validate(response.json()).should_prune

    def enqueue(
        self,
        params: Dict[str, Any],
        value: Optional[float] = None,
    ) -> Trial:
        """Add a trial whose parameter values are already known.

        Use it to seed a study with points evaluated elsewhere, or to have a
        given point evaluated next. The values should lie inside the search
        space declared for the study; when an objective *value* is given, the
        server rejects a point outside it with an error response.

        Args:
            params (Dict[str, Any]): Mapping of parameter name to value.
            value (Optional[float]): Objective value, if the point has already
                been evaluated. With a value the trial is recorded as
                ``COMPLETE`` at once; without one it is queued as ``WAITING``
                and returned by the next :meth:`ask` of any client of the
                study. Defaults to ``None``.

        Returns:
            Trial: The enqueued trial, with the given *params*.

        Raises:
            StudyNotFoundError: If the server has no such study.
            StudyStoppedError: If the study has been stopped.
            ConfigDigestMismatchError: If the study configuration on the server
                differs from this client's.
            IntelloptimAPIError: For any other error response, such as a point
                outside the search space.
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
        """Stop the study.

        Afterwards ``ask`` and ``enqueue`` raise
        :class:`~intelloptim.client.errors.StudyStoppedError` for every client
        of the study, while ``tell`` still accepts trials that were already
        running.

        Raises:
            StudyNotFoundError: If the server has no such study.
        """
        self._client._post_json(f"/studies/{self._study_id}/stop", {})

    def optimize(
        self,
        objective: Callable[[Trial], float],
        n_trials: int,
        *,
        catch: Optional[type] = Exception,
    ) -> None:
        """Run *n_trials* iterations of ask, objective and tell.

        For each trial, the objective is called with the trial the server
        provided and its return value is reported as the result. If the
        objective raises an exception that matches *catch*, a warning is
        logged, the trial is reported as ``FAILED`` and the loop moves on to the
        next one. Any other exception is raised immediately, and the trial that
        was running is left without a result on the server. If the objective
        raises :class:`~intelloptim.client.errors.TrialPruned`, the trial is
        reported as ``PRUNED`` with the last intermediate value it reported
        (none if it reported nothing); this is not a failure and is unaffected
        by *catch*.

        Before the first trial, the number of trials is checked against the
        remaining call budget, using the quota snapshot fetched when the study
        was created. The check is skipped if no snapshot is available.

        Args:
            objective (Callable[[Trial], float]): Function that reads the
                parameters from the trial and returns the objective value.
            n_trials (int): Number of trials to run from this client. Other
                clients of the same study contribute their own trials
                concurrently.
            catch (Optional[type]): Exception class whose instances mark the
                trial as ``FAILED`` instead of stopping the run. ``None``
                disables this, so every exception stops the run. Defaults to
                ``Exception``.

        Raises:
            InsufficientQuotaError: If *n_trials* exceeds the remaining call
                budget. No request is made in this case.
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
            except TrialPruned:
                reported = trial.intermediate_values
                last = reported[max(reported)] if reported else None
                _log.info(f"trial {trial.trial_id} pruned")
                self.tell(trial, value=last, state=TrialState.PRUNED)
                continue
            except Exception as exc:  # noqa: BLE001 - re-raised below
                if catch is None or not isinstance(exc, catch):
                    raise
                _log.warning(
                    f"trial {trial.trial_id} failed: {exc}",
                    exc_info=True,
                )
                self.tell(trial, value=None, state=TrialState.FAILED)
                continue
            self.tell(trial, value, state=TrialState.COMPLETE)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def best_trial(self) -> Optional[Trial]:
        """The best completed trial of the study, if there is one.

        Each access asks the server, so the result reflects the trials of every
        client up to that moment.

        Returns:
            Optional[Trial]: The best trial, with its ``value`` set, or ``None``
            if no trial has completed yet.
        """
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
