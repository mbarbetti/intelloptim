"""Client SDK — the REST client, its handles and the errors it raises.

The package exposes what a user script touches:

- :class:`Client` — connects to a server and creates studies;
- :class:`StudyHandle` — the study returned by :meth:`Client.create_study`;
- :class:`Trial` — the values the server sampled for one evaluation;
- :class:`TrialState` — the lifecycle states of a trial, for ``tell``;
- :mod:`~intelloptim.client.errors` — the exception hierarchy rooted at
  ``IntelloptimError``, reached as ``intelloptim.client.errors.<Name>`` so that
  a script can catch any failure of the SDK. It also holds ``TrialPruned``,
  which an objective raises to end a pruned trial.

Everything else (token lookup, request plumbing) is internal. The package is
reached from the top level as ``intopt.client``::

    import intelloptim as intopt

    with intopt.client.Client(url="https://intelloptim.local") as client:
        study = client.create_study(...)
        study.optimize(my_objective, n_trials=50)
"""

from intelloptim._internal.schemas import TrialState
from intelloptim.client import errors
from intelloptim.client.api import Client
from intelloptim.client.study import StudyHandle
from intelloptim.client.trial import Trial

__all__ = ["Client", "StudyHandle", "Trial", "TrialState", "errors"]
