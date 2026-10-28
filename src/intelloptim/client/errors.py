"""Client-side exception hierarchy.

Every exception raised by the SDK derives from :class:`IntelloptimError`, so a
single ``except IntelloptimError`` catches them all. They mirror the server's
failure modes but live in a hierarchy of their own, so user scripts can catch
them without importing any server module. Reach them through this module,
e.g. ``intelloptim.client.errors.StudyStoppedError``.

An error response from the server is translated by the HTTP status code:

- ``404`` on a ``/tell`` request → :class:`TrialNotFoundError`
- ``404`` on any other request → :class:`StudyNotFoundError`
- ``409`` → :class:`ConfigDigestMismatchError`
- ``410`` → :class:`StudyStoppedError`
- ``503`` → :class:`StudyRebuildingError`
- any other 4xx or 5xx, including ``401`` (invalid token) and ``429`` (quota
  exhausted on the server) → :class:`IntelloptimAPIError`

Three errors are raised locally, before any request is sent:

- :class:`MissingTokenError` — no API token could be found.
- :class:`InsufficientQuotaError` — the requested number of trials exceeds the
  remaining call budget.
- :class:`StudyLimitExceededError` — the active-study cap has been reached.
"""

from __future__ import annotations

from typing import Optional


class IntelloptimError(Exception):
    """Base class for all intelloptim SDK exceptions."""


class IntelloptimAPIError(IntelloptimError):
    """The server answered with an error that has no more specific exception.

    This is the catch-all for HTTP error responses that are not translated
    into one of the dedicated errors, such as ``401`` for an invalid token,
    ``429`` for an exhausted quota or any ``5xx`` other than ``503``.

    Args:
        status_code (int): HTTP status code of the response.
        detail (Optional[str]): The ``detail`` field of the response body, or
            its raw text, when available. Defaults to ``None``.

    Attributes:
        status_code (int): HTTP status code of the response.
        detail (Optional[str]): Server-provided explanation, if any.
    """

    def __init__(self, status_code: int, detail: Optional[str] = None) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail or 'unknown error'}")


class MissingTokenError(IntelloptimError):
    """Raised when no API token can be found in any of the supported sources.

    The client looks for a token in the ``token`` argument, the
    ``INTELLOPTIM_TOKEN`` environment variable and the file
    ``~/.config/intelloptim/token``, in that order, and fails when creating
    the :class:`~intelloptim.client.api.Client` if none of them provides one.
    """


class StudyNotFoundError(IntelloptimError):
    """Raised when the server has no study with the requested ``study_id``."""


class TrialNotFoundError(IntelloptimError):
    """Raised when the trial reported in a ``tell`` is unknown to the server."""


class ConfigDigestMismatchError(IntelloptimError):
    """Raised when the study configuration differs from the one on the server.

    Every study request carries a ``config_digest`` computed from the client's
    specification of the study. A mismatch usually means that two callers
    believe they share a study but submitted different specifications, for
    example a different parameter range or sampler setting.
    """


class StudyStoppedError(IntelloptimError):
    """Raised when asking for or enqueueing a trial on a stopped study.

    A stopped study no longer accepts ``ask`` or ``enqueue`` requests, but it
    still accepts the ``tell`` of trials that were already running.
    """


class StudyRebuildingError(IntelloptimError):
    """Raised when the study is rebuilding its sampler and cannot serve asks.

    This is transient rather than a hard failure: back off and retry. While a
    rebuild is in progress the server keeps accepting ``tell`` requests.

    Args:
        message (str): Description of the failure.
        retry_after_seconds (Optional[float]): Seconds to wait before retrying,
            taken from the ``Retry-After`` header of the response. ``None`` if
            the header is missing or is not a number of seconds. Defaults to
            ``None``.

    Attributes:
        retry_after_seconds (Optional[float]): Suggested wait before retrying.
    """

    def __init__(
        self, message: str, retry_after_seconds: Optional[float] = None
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class InsufficientQuotaError(IntelloptimError):
    """Raised before any trial runs when ``n_trials`` exceeds the call budget.

    This is a client-side pre-flight check, made by
    :meth:`~intelloptim.client.study.StudyHandle.optimize` against the quota
    snapshot fetched when the study was created. It prevents obvious
    over-budget runs without consuming any quota. The server enforces the
    limit independently on every ``ask`` and ``tell``.
    """


class StudyLimitExceededError(IntelloptimError):
    """Raised before contacting the server when the study cap is reached.

    :meth:`~intelloptim.client.api.Client.create_study` makes this check
    against the quota snapshot it fetches on first use. Delete an existing
    study to free a slot, or contact your admin to upgrade your role. The
    server enforces the same limit independently.
    """


__all__ = [
    "ConfigDigestMismatchError",
    "InsufficientQuotaError",
    "IntelloptimAPIError",
    "IntelloptimError",
    "MissingTokenError",
    "StudyLimitExceededError",
    "StudyNotFoundError",
    "StudyRebuildingError",
    "StudyStoppedError",
    "TrialNotFoundError",
]
