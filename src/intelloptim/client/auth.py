"""API-token resolution for the client.

The client authenticates with a bearer token, which :func:`resolve_token` looks
for in three places. The first one that yields a non-empty value wins:

1. The ``token=`` argument passed to :class:`~intelloptim.client.api.Client`.
2. The ``INTELLOPTIM_TOKEN`` environment variable.
3. The contents of ``~/.config/intelloptim/token``, if the file exists.

Surrounding whitespace is stripped from the token. If none of the sources
yields a token, :class:`~intelloptim.client.errors.MissingTokenError` is raised, so a missing token fails
at client creation rather than on the first request.
"""

from __future__ import annotations

import os
import stat
from typing import Optional

from intelloptim._internal.logging import get_logger
from intelloptim.client.errors import MissingTokenError

_log = get_logger(__name__)


def resolve_token(explicit: Optional[str] = None) -> str:
    """Return the API token, looking in the supported sources in order.

    The order is the explicit argument, the ``INTELLOPTIM_TOKEN`` environment
    variable, then the file ``~/.config/intelloptim/token``. Empty and
    whitespace-only values fall through to the next source, and the returned
    token has no surrounding whitespace. On POSIX systems, a warning is logged
    when the token file can be read or written by other users.

    Args:
        explicit (Optional[str]): Token passed via ``Client(..., token=...)``.
            Takes precedence over the environment variable and the file when
            set. Defaults to ``None``.

    Returns:
        str: The resolved API token.

    Raises:
        MissingTokenError: If no source provides a token.
        OSError: If the token file exists but cannot be read.
    """
    for candidate in (explicit, os.environ.get("INTELLOPTIM_TOKEN")):
        if candidate and candidate.strip():
            return candidate.strip()

    path = os.path.join(os.path.expanduser("~"), ".config", "intelloptim", "token")
    if os.path.exists(path):
        _warn_if_shared(path)
        with open(path, "r", encoding="utf-8") as f:
            on_disk = f.read().strip()
        if on_disk:
            return on_disk

    raise MissingTokenError(
        "No API token found. Pass token=... to Client(...), set "
        "INTELLOPTIM_TOKEN, or write the token to "
        "~/.config/intelloptim/token."
    )


def _warn_if_shared(path: str) -> None:
    """Log a warning if the token file at *path* is accessible to other users.

    Only meaningful on POSIX, where a token file should be readable by its
    owner alone (mode ``600``), like an SSH key. The token itself is never
    logged. Failures to inspect the file are ignored: reading it will raise
    the relevant error.
    """
    if os.name != "posix":
        return
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        _log.warning(
            f"token file is accessible by other users; restrict it with "
            f"'chmod 600 {path}'",
        )


__all__ = ["resolve_token"]
