"""Client-side exception hierarchy.

Mirrors the orchestrator's failure modes on the server, but mapped to a
dedicated hierarchy so SDK users can ``except intopt.StudyStoppedError`` in
their own scripts without dragging server modules into their imports.

The mapping from HTTP status codes is one-to-one with the routes:

* ``404`` on ``/ask``, ``/enqueue``, ``/stop``, ``GET /studies/{id}`` →
  :class:`StudyNotFoundError`
* ``404`` on ``/tell`` → :class:`TrialNotFoundError`
* ``409`` on any mutating endpoint → :class:`ConfigDigestMismatchError`
* ``410`` on ``/ask`` or ``/enqueue`` → :class:`StudyStoppedError`
* ``503`` on ``/ask`` → :class:`StudyRebuildingError`
* anything else 4xx/5xx → :class:`IntelloptimAPIError`
"""

from __future__ import annotations

from typing import Optional


class IntelloptimError(Exception):
    """Base class for all intelloptim SDK exceptions."""


class IntelloptimAPIError(IntelloptimError):
    """A non-success HTTP response we did not translate to a more specific
    error type. Carries the status code and the server's ``detail`` field
    when available.
    """

    def __init__(self, status_code: int, detail: Optional[str] = None) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail or 'unknown error'}")


class StudyNotFoundError(IntelloptimError):
    """The server has no study under the supplied ``study_id``."""


class TrialNotFoundError(IntelloptimError):
    """The trial referenced in ``/tell`` is unknown to the server."""


class ConfigDigestMismatchError(IntelloptimError):
    """The locally-computed ``config_digest`` disagrees with what the
    server has for this study. Typically means two callers think they
    share a study but submitted different specs."""


class StudyStoppedError(IntelloptimError):
    """The study has been stopped; ``/ask`` and ``/enqueue`` are no
    longer accepted (but ``/tell`` for already-running trials still is).
    """


class StudyRebuildingError(IntelloptimError):
    """The study is rebuilding its sampler state and is not currently
    serving asks. The server included a ``Retry-After`` hint; back off
    and retry rather than treating this as a hard failure."""

    def __init__(
        self, message: str, retry_after_seconds: Optional[float] = None
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


__all__ = [
    "ConfigDigestMismatchError",
    "IntelloptimAPIError",
    "IntelloptimError",
    "StudyNotFoundError",
    "StudyRebuildingError",
    "StudyStoppedError",
    "TrialNotFoundError",
]
