"""API-token loading helpers.

Resolution order (first non-empty wins):

1. The ``token=`` argument passed to :class:`Client`.
2. The ``INTELLOPTIM_TOKEN`` environment variable.
3. The contents of ``~/.config/intelloptim/token`` if it exists.

If none of these yield a value, :func:`resolve_token` raises so the
caller fails fast at SDK init time rather than on first request.
"""

from __future__ import annotations

import os
from typing import Optional


class MissingTokenError(RuntimeError):
    """No API token could be resolved from any of the supported sources."""


def resolve_token(explicit: Optional[str] = None) -> str:
    """Return the first non-empty token in the documented resolution order.

    Args:
        explicit: Token passed via ``Client(..., token=...)``. Takes
            precedence over the environment and on-disk file when set.

    Raises:
        MissingTokenError: No token could be found anywhere.
    """
    if explicit:
        return explicit.strip()

    env_token = os.environ.get("INTELLOPTIM_TOKEN")
    if env_token:
        return env_token.strip()

    path = os.path.join(os.path.expanduser("~"), ".config", "intelloptim", "token")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            on_disk = f.read().strip()
        if on_disk:
            return on_disk

    raise MissingTokenError(
        "No API token found. Pass token=... to Client(...), set "
        "INTELLOPTIM_TOKEN, or write the token to "
        "~/.config/intelloptim/token."
    )


__all__ = ["MissingTokenError", "resolve_token"]
