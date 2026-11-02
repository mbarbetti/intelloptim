Running on a cluster
====================

Optimization campaigns are usually run on batch systems: many jobs, on many
nodes and sometimes on several sites, all working on one study. ``intelloptim``
is built for this. The server holds the study, each job is a client, and the
``intelloptim-worker`` command turns an objective into a job without a driver
script. This page uses SLURM, whose environment the worker understands, but the
idea applies to any scheduler.

The evaluator worker
--------------------

``intelloptim-worker evaluator`` does what you would otherwise write in a script:

1. imports your objective,
2. checks that the server answers,
3. creates the study described by the objective's spec, or joins it if it already
   exists,
4. runs the number of trials you ask for, and reports the best result.

Because a study is identified by its configuration, every worker that starts with
the same spec lands on the same study. There is nothing to coordinate between
jobs.

Write the objective and its study spec
--------------------------------------

Put the objective in a module the worker can import, and attach the study spec to
it with :func:`~intelloptim.objective`:

.. code-block:: python

   # objective.py
   import math

   import intelloptim as intopt


   @intopt.objective(
       "branin",
       study_spec={
           "study_name": "branin-demo",
           "direction": "minimize",
           "sampler_spec": {"kind": "tpe", "seed": 42},
           "pruner_spec": {"kind": "nop"},
           "parameters": [
               {"kind": "float", "name": "x", "low": -5.0, "high": 10.0},
               {"kind": "float", "name": "y", "low": 0.0, "high": 15.0},
           ],
       },
   )
   def branin(trial: intopt.client.Trial) -> float:
       x = trial.sampled_float("x")
       y = trial.sampled_float("y")
       b, c, t = 5.1 / (4 * math.pi**2), 5.0 / math.pi, 1.0 / (8 * math.pi)
       return (y - b * x**2 + c * x - 6.0) ** 2 + 10.0 * (1 - t) * math.cos(x) + 10.0

The spec is a plain dictionary with the same fields as the arguments of
:meth:`~intelloptim.client.Client.create_study`; the sampler and pruner are given
by their ``kind`` and settings. If you prefer to keep it out of the code, pass a
JSON file with ``--spec-file`` instead.

Try one worker by hand first, from a login node or a laptop:

.. code-block:: console

   intelloptim-worker evaluator \
       --server https://intelloptim.example.org \
       --import objective.py:branin \
       --n-trials 5

``--import`` takes ``module:function`` for an installed or importable module, or
``path/to/file.py:function`` for a file that is not installed, which is handy in
a project directory on a cluster.

Submit an array job
-------------------

A SLURM *array job* starts many identical tasks, and each one is a worker. This
script runs 20 tasks, at most 10 at a time, with 50 trials each:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=intopt-eval
   #SBATCH --array=1-20%10
   #SBATCH --time=04:00:00
   #SBATCH --cpus-per-task=1
   #SBATCH --mem=4G
   #SBATCH --output=logs/%x-%A-%a.out

   set -euo pipefail

   # An environment with intelloptim installed (here: conda)
   source "$(conda info --base)/etc/profile.d/conda.sh"
   conda activate intelloptim

   export INTELLOPTIM_URL="https://intelloptim.example.org"
   # The token is read from ~/.config/intelloptim/token (see below)

   intelloptim-worker evaluator \
       --import "$SLURM_SUBMIT_DIR/objective.py:branin" \
       --n-trials 50

.. code-block:: console

   mkdir -p logs
   sbatch evaluator.sbatch

All 1000 trials go to the same study. The array size and ``--n-trials`` decide how
many trials each task contributes, and ``%10`` caps how many tasks run at once.
Tasks that start late join the study as it is, and the sampler uses everything
finished so far.

Handle the token on a cluster
-----------------------------

The worker needs the API token, and the cluster is a shared place, so keep it out
of scripts and command lines (:doc:`authentication` explains why).

* Put it in ``~/.config/intelloptim/token`` with ``chmod 600``, on a file system
  that the compute nodes can read, such as your home directory. The client uses it
  automatically and checks its permissions.
* Or export ``INTELLOPTIM_TOKEN`` in your submission shell. SLURM passes the
  environment to the job by default.

Do not write the token value in the batch script, and do not pass it with
``--token``.

Identity of workers and sites
-----------------------------

Each request carries a **worker id** and a **site id**, so the server can tell
which process produced a trial and group trials by where they ran. They are
derived from the SLURM environment:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Where it runs
     - Worker id
     - Site id
   * - Array job
     - ``eval-slurm-<array job>-<task>``
     - ``slurm-<array job>``
   * - Plain SLURM job
     - ``eval-slurm-<job>``
     - ``slurm-<job>``
   * - Elsewhere
     - ``eval-<host>-<pid>``
     - the host name

Override them with ``--worker-id`` and ``--site``, which is how you label a
campaign on a scheduler other than SLURM, or name the cluster itself:

.. code-block:: console

   intelloptim-worker evaluator ... --site marconi --worker-id "$PBS_JOBID"

Several sites on one study
--------------------------

Nothing ties a study to a place. Submit the same worker command at several
sites, each with network access to the server and a valid token, and every one
contributes to the study. This is the case ``intelloptim`` descends from: opportunistic
resources at different providers working on a single optimization (see
:doc:`../index`).

Failures, preemption and budgets
--------------------------------

* **A failing objective** does not stop the worker: by default the trial is
  recorded as ``FAILED`` and the next one starts. Add ``--no-catch`` to stop at
  the first exception.
* **A killed task** (time limit, preemption, node failure) ends without telling
  the trial it was running, which then stays unfinished on the server. A task that
  is requeued starts a new run and contributes a new batch of trials, so size
  ``--time`` and ``--n-trials`` so that a task normally finishes.
* **The call quota** is checked before the first trial. If ``--n-trials`` exceeds
  what you have left, the worker stops with
  :class:`~intelloptim.client.errors.InsufficientQuotaError` and asks for nothing,
  so size the array against your budget. Both asks and tells count.
* **A rebuilding sampler** answers ``ask`` with a retry hint (see
  :doc:`sampling`). Because the worker runs ``optimize``, a rebuild ends the run
  with :class:`~intelloptim.client.errors.StudyRebuildingError`; for such samplers
  write the loop yourself as in :doc:`ask-tell-enqueue`.

Logging
-------

The worker logs one line per important event. Use ``--loglevel DEBUG`` for more.
Set ``INTELLOPTIM_LOG_FORMAT=json`` to get one JSON object per line, which suits
log collectors, and ``INTELLOPTIM_LOG_FILE`` to write to a file. Fields whose
names look like secrets, such as ``token`` or ``password``, are redacted.

All options are listed in :doc:`../api/worker`.
