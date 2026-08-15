"""Audit event helpers.

These produce a small, **stable, machine-parseable** vocabulary of
security-relevant events. Operators and (future) the dashboard rely on
the field names here to filter and surface activity; treat them as part
of the wire contract.

Every audit record carries:

* ``event_category="audit"`` — distinguishes from operational logs.
* ``event`` — short stable name (e.g. ``"login.success"``,
  ``"token.issued"``, ``"study.stopped"``).
* ``outcome`` — ``"success"`` or ``"failure"``.
* ``actor`` — opaque identifier (user id, token prefix, or ``"anonymous"``).
* arbitrary additional structured fields per event type.

Audit records are routed by :mod:`intelloptim._internal.logging` to the
``intelloptim.audit`` logger, which by default inherits the operational
handlers and is therefore visible in the main log stream with
``event_category=audit``. Operators can split it off with
``INTELLOPTIM_AUDIT_FILE``.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from intelloptim._internal.logging import ROOT_LOGGER_NAME, get_logger


def get_audit_logger() -> logging.Logger:
    """Return the canonical ``intelloptim.audit`` logger."""
    # get_logger ensures configure_logging has run.
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
    """Emit a single audit record.

    Args:
        event: Short stable identifier, e.g. ``"login.success"``,
            ``"token.revoked"``. Use dots to group related events.
        outcome: ``"success"`` or ``"failure"``.
        actor: Who took the action — user id, token prefix, etc. Use
            ``"anonymous"`` for unauthenticated attempts.
        extra: Additional structured fields (e.g. ``{"study_id": "..."}``).
        message: Optional human-readable summary. Defaults to the event
            name.
        level: Log level. Use :data:`logging.WARNING` for failures or
            otherwise suspicious activity.
    """
    payload: dict[str, Any] = {
        "event_category": "audit",
        "event": event,
        "outcome": outcome,
        "actor": actor or "anonymous",
    }
    if extra:
        payload.update(extra)
    get_audit_logger().log(level, message or event, extra=payload)


# Convenience helpers — keep the call sites short and the vocabulary
# discoverable from this one module.


def login_success(*, user_id: str, email: str) -> None:
    emit("login.success", outcome="success", actor=user_id, extra={"email": email})


def login_failure(*, email: str, reason: str) -> None:
    emit(
        "login.failure",
        outcome="failure",
        actor="anonymous",
        extra={"email": email, "reason": reason},
        level=logging.WARNING,
    )


def logout(*, user_id: str) -> None:
    emit("logout", outcome="success", actor=user_id)


def token_issued(*, user_id: str, token_prefix: str, name: str) -> None:
    emit(
        "token.issued",
        outcome="success",
        actor=user_id,
        extra={"token_prefix": token_prefix, "token_name": name},
    )


def token_revoked(*, user_id: str, token_prefix: str) -> None:
    emit(
        "token.revoked",
        outcome="success",
        actor=user_id,
        extra={"token_prefix": token_prefix},
    )


def token_deleted(*, user_id: str, token_prefix: str) -> None:
    emit(
        "token.deleted",
        outcome="success",
        actor=user_id,
        extra={"token_prefix": token_prefix},
    )


def token_authentication_failure(*, prefix: str, reason: str) -> None:
    emit(
        "token.auth_failed",
        outcome="failure",
        actor=f"prefix:{prefix}" if prefix else "anonymous",
        extra={"reason": reason},
        level=logging.WARNING,
    )


def study_created(*, study_id: str, user_id: str, name: str) -> None:
    emit(
        "study.created",
        outcome="success",
        actor=user_id,
        extra={"study_id": study_id, "study_name": name},
    )


def study_stopped(*, study_id: str, user_id: str) -> None:
    emit(
        "study.stopped",
        outcome="success",
        actor=user_id,
        extra={"study_id": study_id},
    )


def study_rebuild_started(*, study_id: str, user_id: Optional[str] = None) -> None:
    emit(
        "study.rebuild.started",
        outcome="success",
        actor=user_id or "system",
        extra={"study_id": study_id},
    )


def study_rebuild_finished(*, study_id: str, user_id: Optional[str] = None) -> None:
    emit(
        "study.rebuild.finished",
        outcome="success",
        actor=user_id or "system",
        extra={"study_id": study_id},
    )


__all__ = [
    "emit",
    "get_audit_logger",
    "login_failure",
    "login_success",
    "logout",
    "study_created",
    "study_rebuild_finished",
    "study_rebuild_started",
    "study_stopped",
    "token_authentication_failure",
    "token_deleted",
    "token_issued",
    "token_revoked",
]
