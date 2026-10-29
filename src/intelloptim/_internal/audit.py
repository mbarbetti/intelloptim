"""Audit events: a small, stable record of security-relevant actions.

Each function below records one kind of event (a login, a token change, a study
stopped or rebuilt) as a structured log record on the ``intelloptim.audit``
logger. Event and field names do not change between releases, so operators can
filter and alert on them.

Every record carries:

* ``event_category="audit"`` — tells audit records apart from operational ones.
* ``event`` — short, stable name such as ``"login.success"``,
  ``"token.issued"`` or ``"study.stopped"``.
* ``outcome`` — ``"success"`` or ``"failure"``.
* ``actor`` — who acted: a user id, a token prefix, ``"system"`` for background
  work or ``"anonymous"``.
* extra fields specific to the event, such as ``study_id`` or ``reason``.

Records are written to the main log stream, marked with ``event_category=audit``,
unless ``INTELLOPTIM_AUDIT_FILE`` sends them to a file of their own (see
:mod:`intelloptim._internal.logging`). Tokens are identified by their prefix
only, never in full.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from intelloptim._internal.logging import ROOT_LOGGER_NAME, get_logger

_STANDARD_FIELDS = frozenset({"event_category", "event", "outcome", "actor"})

# Names ``extra=`` cannot use: the standard fields above and the attributes of a
# LogRecord (``logging`` raises ``KeyError`` on those, and ``message`` and
# ``asctime`` are set by the formatters).
_RESERVED_FIELDS = (
    _STANDARD_FIELDS
    | frozenset(logging.makeLogRecord({}).__dict__)
    | {"message", "asctime"}
)


def get_audit_logger() -> logging.Logger:
    """Return the ``intelloptim.audit`` logger, configuring logging if needed.

    Returns:
        logging.Logger: The logger all audit records are written to.
    """
    get_logger("audit")
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.audit")


def emit(
    event: str,
    *,
    outcome: str,
    actor: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    message: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """Write one audit record.

    The functions below call this with the right event name and fields; call it
    directly only for an event they do not cover.

    Args:
        event (str): Short stable name, for example ``"login.success"``. Dots
            group related events.
        outcome (str): ``"success"`` or ``"failure"``.
        actor (Optional[str]): Who acted: a user id or a token prefix.
            ``None`` means ``"anonymous"``.
        extra (Optional[Mapping[str, Any]]): Additional structured fields, for
            example ``{"study_id": "..."}``. They are merged into the record
            and must not reuse the names of the standard fields or of
            :class:`logging.LogRecord` attributes.
        message (Optional[str]): Human-readable summary; defaults to *event*.
        level (int): Log level. Use :data:`logging.WARNING` for failures and
            suspicious activity.

    Raises:
        ValueError: If *extra* uses a reserved field name. Nothing is logged.
    """
    if extra:
        clashes = sorted(_RESERVED_FIELDS & set(extra))
        if clashes:
            raise ValueError(
                f"audit extra fields use reserved names: {', '.join(clashes)}"
            )
    payload: Dict[str, Any] = {
        "event_category": "audit",
        "event": event,
        "outcome": outcome,
        "actor": actor or "anonymous",
    }
    if extra:
        payload.update(extra)
    get_audit_logger().log(level, message or event, extra=payload)


def login_success(*, user_id: str, email: str) -> None:
    """Record a successful login (event ``login.success``).

    Args:
        user_id (str): Id of the user who logged in; becomes the actor.
        email (str): Email the user logged in with.
    """
    emit("login.success", outcome="success", actor=user_id, extra={"email": email})


def login_failure(*, email: str, reason: str) -> None:
    """Record a failed login (event ``login.failure``, level ``WARNING``).

    Args:
        email (str): Email the attempt used. The actor is ``"anonymous"``, since
            the user is not authenticated.
        reason (str): Why the login failed, for example ``"invalid-credentials"``.
    """
    emit(
        "login.failure",
        outcome="failure",
        actor="anonymous",
        extra={"email": email, "reason": reason},
        level=logging.WARNING,
    )


def logout(*, user_id: str) -> None:
    """Record a logout (event ``logout``).

    Args:
        user_id (str): Id of the user who logged out; becomes the actor.
    """
    emit("logout", outcome="success", actor=user_id)


def token_issued(*, user_id: str, token_prefix: str, name: str) -> None:
    """Record the creation of an API token (event ``token.issued``).

    Args:
        user_id (str): Id of the user who owns the token; becomes the actor.
        token_prefix (str): First characters of the token, enough to identify it
            in a log. Never pass the full token.
        name (str): Label the user gave the token.
    """
    emit(
        "token.issued",
        outcome="success",
        actor=user_id,
        extra={"token_prefix": token_prefix, "token_name": name},
    )


def token_revoked(*, user_id: str, token_prefix: str) -> None:
    """Record that an API token was revoked (event ``token.revoked``).

    Args:
        user_id (str): Id of the user who owns the token; becomes the actor.
        token_prefix (str): Prefix identifying the token. Never pass the full
            token.
    """
    emit(
        "token.revoked",
        outcome="success",
        actor=user_id,
        extra={"token_prefix": token_prefix},
    )


def token_deleted(*, user_id: str, token_prefix: str) -> None:
    """Record that an API token was deleted (event ``token.deleted``).

    Args:
        user_id (str): Id of the user who owns the token; becomes the actor.
        token_prefix (str): Prefix identifying the token. Never pass the full
            token.
    """
    emit(
        "token.deleted",
        outcome="success",
        actor=user_id,
        extra={"token_prefix": token_prefix},
    )


def token_authentication_failure(*, prefix: str, reason: str) -> None:
    """Record a request rejected for its token (event ``token.auth_failed``).

    The record has level ``WARNING``.

    Args:
        prefix (str): Prefix of the token that was presented. The actor is
            ``"prefix:<prefix>"``, or ``"anonymous"`` if the prefix is empty.
        reason (str): Why authentication failed, for example ``"invalid-or-revoked"``.
    """
    emit(
        "token.auth_failed",
        outcome="failure",
        actor=f"prefix:{prefix}" if prefix else "anonymous",
        extra={"reason": reason},
        level=logging.WARNING,
    )


def study_created(*, study_id: str, user_id: str, name: str) -> None:
    """Record the creation of a study (event ``study.created``).

    Args:
        study_id (str): Id of the new study.
        user_id (str): Id of the user who created it; becomes the actor.
        name (str): Study name, stored in the record as ``study_name``.
    """
    emit(
        "study.created",
        outcome="success",
        actor=user_id,
        extra={"study_id": study_id, "study_name": name},
    )


def study_stopped(*, study_id: str, user_id: str) -> None:
    """Record that a study was stopped (event ``study.stopped``).

    Args:
        study_id (str): Id of the stopped study.
        user_id (str): Id of the user who stopped it; becomes the actor.
    """
    emit(
        "study.stopped",
        outcome="success",
        actor=user_id,
        extra={"study_id": study_id},
    )


def trial_enqueued(*, study_id: str, user_id: str, trial_number: int) -> None:
    """Record a trial injected with a known result (event ``trial.enqueued``).

    Only trials added together with an objective value are recorded: they enter
    the study as completed and shape what the sampler learns. The value and the
    parameters are not part of the record.

    Args:
        study_id (str): Id of the study that received the trial.
        user_id (str): Id of the user who injected it; becomes the actor.
        trial_number (int): Zero-based position of the new trial in the study.
    """
    emit(
        "trial.enqueued",
        outcome="success",
        actor=user_id,
        extra={"study_id": study_id, "trial_number": trial_number},
    )


def quota_exhausted(*, user_id: str, calls_per_month: int, refused: int = 1) -> None:
    """Record calls refused because the monthly quota is used up.

    The event is ``quota.exhausted``, with outcome ``failure`` and level
    ``WARNING``. A client that keeps retrying makes many refusals, so the caller
    may log one record for several of them and report their number in *refused*.

    Args:
        user_id (str): Id of the user whose quota is exhausted; becomes the
            actor.
        calls_per_month (int): The user's effective monthly limit.
        refused (int): Number of refused calls this record stands for.
    """
    emit(
        "quota.exhausted",
        outcome="failure",
        actor=user_id,
        extra={"calls_per_month": calls_per_month, "refused": refused},
        level=logging.WARNING,
    )


def study_access_denied(*, study_id: str, user_id: str, action: str) -> None:
    """Record a request for a study that belongs to someone else.

    The event is ``study.access_denied``, with outcome ``failure`` and level
    ``WARNING``. The caller answers 404 as if the study did not exist, so this
    record is the only trace that the study does exist.

    Args:
        study_id (str): Id of the study that was requested.
        user_id (str): Id of the user who asked; becomes the actor.
        action (str): What was attempted: ``"read"``, ``"stop"``, ``"delete"``
            or ``"rebuild"``.
    """
    emit(
        "study.access_denied",
        outcome="failure",
        actor=user_id,
        extra={"study_id": study_id, "action": action},
        level=logging.WARNING,
    )


def study_rebuild_started(*, study_id: str, user_id: Optional[str] = None) -> None:
    """Record the start of a sampler rebuild (event ``study.rebuild.started``).

    Args:
        study_id (str): Id of the study being rebuilt.
        user_id (Optional[str]): Id of the user who triggered the rebuild, or
            ``None`` if the server started it, in which case the actor is
            ``"system"``.
    """
    emit(
        "study.rebuild.started",
        outcome="success",
        actor=user_id or "system",
        extra={"study_id": study_id},
    )


def study_rebuild_finished(*, study_id: str, user_id: Optional[str] = None) -> None:
    """Record the end of a sampler rebuild (event ``study.rebuild.finished``).

    Args:
        study_id (str): Id of the rebuilt study.
        user_id (Optional[str]): Id of the user who triggered the rebuild, or
            ``None`` if the server started it, in which case the actor is
            ``"system"``.
    """
    emit(
        "study.rebuild.finished",
        outcome="success",
        actor=user_id or "system",
        extra={"study_id": study_id},
    )


__all__ = [
    "login_failure",
    "login_success",
    "logout",
    "quota_exhausted",
    "study_access_denied",
    "study_created",
    "study_rebuild_finished",
    "study_rebuild_started",
    "study_stopped",
    "token_authentication_failure",
    "token_deleted",
    "token_issued",
    "token_revoked",
    "trial_enqueued",
]
