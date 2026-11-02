Worker command
==============

``intelloptim-worker`` is installed with the package. Its only subcommand,
``evaluator``, runs an objective against a server; see :doc:`../guides/cluster` for
how to use it on a batch system.

.. code-block:: console

   intelloptim-worker evaluator --server URL --import OBJECTIVE [OPTIONS]

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Option
     - Meaning
   * - ``--server TEXT``
     - Base URL of the intelloptim server. Required; falls back to the
       ``INTELLOPTIM_URL`` environment variable.
   * - ``--import MODULE:CALLABLE``
     - The objective to run: ``my_pkg.objectives:my_fn`` or
       ``/path/to/objective.py:my_fn``. The module is imported at start-up, which
       registers its ``@intopt.objective`` study spec. Required.
   * - ``--token TEXT``
     - API bearer token. Falls back to ``INTELLOPTIM_TOKEN`` and then to
       ``~/.config/intelloptim/token``. Prefer those over the option, which is
       visible in the process list.
   * - ``--spec-file FILE``
     - JSON file with the study spec. Not needed if the objective was decorated
       with ``study_spec={...}``.
   * - ``--n-trials INTEGER``
     - Number of trials this worker runs. Default: ``50``.
   * - ``--worker-id TEXT``
     - Worker id sent as ``X-Worker-Id``. Derived from the SLURM environment, or
       from the host name and process id, when not set.
   * - ``--site TEXT``
     - Site id sent as ``X-Site``, which groups workers by cluster or host.
       Derived from the SLURM environment, or from the host name, when not set.
   * - ``--loglevel [debug|info|warning|error]``
     - Logging level. Default: ``INFO``.
   * - ``--no-catch``
     - Stop the run when the objective raises, instead of marking that trial as
       failed and continuing.

Environment variables
---------------------

``INTELLOPTIM_URL`` and ``INTELLOPTIM_TOKEN`` provide the server and the token.
``INTELLOPTIM_LOG_LEVEL``, ``INTELLOPTIM_LOG_FORMAT`` (``json``, ``console`` or
``auto``) and ``INTELLOPTIM_LOG_FILE`` configure the logs.

Exit status
-----------

The worker exits with status ``1`` if the server does not answer the health probe
at start-up. An error in the objective import or an exhausted quota ends it with
the corresponding exception.
