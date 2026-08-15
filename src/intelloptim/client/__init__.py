"""Client SDK — the surface end-users import as ``intopt.Client`` etc.

Phase 2 ships the synchronous client. Direct imports::

    from intelloptim.client import Client, StudyHandle, Trial

Through the top-level alias::

    import intelloptim as intopt

    client = intopt.Client(url=..., token=...)
    study = client.create_study(...)
    for _ in range(50):
        trial = study.ask()
        study.tell(trial, my_objective(trial))
"""

from intelloptim.client.api import Client
from intelloptim.client.errors import (
    ConfigDigestMismatchError,
    IntelloptimAPIError,
    IntelloptimError,
    StudyNotFoundError,
    StudyRebuildingError,
    StudyStoppedError,
    TrialNotFoundError,
)
from intelloptim.client.study import StudyHandle
from intelloptim.client.trial import Trial

__all__ = [
    "Client",
    "ConfigDigestMismatchError",
    "IntelloptimAPIError",
    "IntelloptimError",
    "StudyHandle",
    "StudyNotFoundError",
    "StudyRebuildingError",
    "StudyStoppedError",
    "Trial",
    "TrialNotFoundError",
]
