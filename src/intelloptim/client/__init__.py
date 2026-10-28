"""Client SDK — the REST client, its handles and the errors it raises.

The package exposes what a user script touches:

- :class:`Client` — connects to a server and creates studies;
- :class:`StudyHandle` — the study returned by :meth:`Client.create_study`;
- :class:`Trial` — the values the server sampled for one evaluation;
- :mod:`~intelloptim.client.errors` — the exception hierarchy rooted at
  ``IntelloptimError``, reached as ``intelloptim.client.errors.<Name>`` so that
  a script can catch any failure of the SDK.

Everything else (token lookup, request plumbing) is internal. ``Client`` and
``Trial`` are also available as ``intopt.Client`` and ``intopt.Trial``::

    import intelloptim as intopt

    with intopt.Client(url="https://intelloptim.local") as client:
        study = client.create_study(...)
        study.optimize(my_objective, n_trials=50)
"""

from intelloptim.client import errors
from intelloptim.client.api import Client
from intelloptim.client.study import StudyHandle
from intelloptim.client.trial import Trial

__all__ = ["Client", "StudyHandle", "Trial", "errors"]
