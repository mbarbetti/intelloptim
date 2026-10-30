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

import errno
import os
import stat
from typing import Optional

from intelloptim.client.errors import InsecureTokenFileError, MissingTokenError


def resolve_token(explicit: Optional[str] = None) -> str:
    """Return the API token, looking in the supported sources in order.

    The order is the explicit argument, the ``INTELLOPTIM_TOKEN`` environment
    variable, then the file ``~/.config/intelloptim/token``. Empty and
    whitespace-only values fall through to the next source, and the returned
    token has no surrounding whitespace. On POSIX systems, an unsafe token
    file is rejected rather than read.

    Args:
        explicit (Optional[str]): Token passed via ``Client(..., token=...)``.
            Takes precedence over the environment variable and the file when
            set. Defaults to ``None``.

    Returns:
        str: The resolved API token.

    Raises:
        MissingTokenError: If no source provides a token.
        InsecureTokenFileError: If the token file is not a private regular file
            owned by the current user.
        OSError: If the token file exists but cannot be read.
    """
    for candidate in (explicit, os.environ.get("INTELLOPTIM_TOKEN")):
        if candidate and candidate.strip():
            return candidate.strip()

    path = os.path.join(os.path.expanduser("~"), ".config", "intelloptim", "token")
    try:
        on_disk = _read_token_file(path)
    except FileNotFoundError:
        on_disk = ""
    if on_disk:
        return on_disk

    raise MissingTokenError(
        "No API token found. Pass token=... to Client(...), set "
        "INTELLOPTIM_TOKEN, or write the token to "
        "~/.config/intelloptim/token."
    )


def _read_token_file(path: str) -> str:
    """Read a token file after verifying its ownership and permissions.

    On POSIX, open the file descriptor without following symlinks and inspect
    that descriptor to avoid a time-of-check/time-of-use race. Other platforms
    retain normal file semantics because ownership and mode bits are not
    portable there.
    """
    if os.name != "posix":
        with open(path, "r", encoding="utf-8") as token_file:
            return token_file.read().strip()

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise InsecureTokenFileError("token file must not be a symlink") from exc
        raise

    try:
        file_stat = os.fstat(fd)
        _validate_token_file(path, file_stat)
        with os.fdopen(fd, "r", encoding="utf-8") as token_file:
            fd = -1
            return token_file.read().strip()
    finally:
        if fd != -1:
            os.close(fd)


def _validate_token_file(path: str, file_stat: os.stat_result) -> None:
    """Reject a token file that is unsafe to read on a shared POSIX host."""
    if not stat.S_ISREG(file_stat.st_mode):
        raise InsecureTokenFileError("token file must be a regular file")
    if file_stat.st_uid != os.getuid():
        raise InsecureTokenFileError("token file must be owned by the current user")
    mode = stat.S_IMODE(file_stat.st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise InsecureTokenFileError(
            "token file must not grant permissions to group or other users"
        )

    directory_stat = os.stat(os.path.dirname(path))
    directory_mode = stat.S_IMODE(directory_stat.st_mode)
    if directory_stat.st_uid != os.getuid() or directory_mode & (
        stat.S_IWGRP | stat.S_IWOTH
    ):
        raise InsecureTokenFileError(
            "token directory must be owned by the current user and not writable "
            "by group or other users"
        )


__all__ = ["resolve_token"]
