Optuna, orchestrated by a server
================================

``intelloptim`` orchestrates Optuna-compatible optimization through a server.
If you know Optuna, the vocabulary is the same: a **study** is one optimization,
a **trial** is one evaluation of the objective, a **sampler** proposes the next
point and a **pruner** stops trials that are not going anywhere. What changes is
where these live. The study, its history and the samplers run on the server, and
every evaluator, wherever it runs, obtains a trial, evaluates it and reports the
result.

.. code-block:: text

   evaluator 1 ─┐                    ┌──────────────────────────┐
   evaluator 2 ─┼── ask / tell ─────►│ server: one study,       │
   evaluator N ─┘   enqueue / prune  │ its history and sampler  │
                                     └──────────────────────────┘

Optuna concepts and their counterparts
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - In Optuna
     - In ``intelloptim``
   * - ``optuna.create_study(...)``
     - :meth:`Client.create_study <intelloptim.client.Client.create_study>`,
       which is idempotent: calling it again, from any process, joins the same
       study.
   * - ``study.ask()`` and ``study.tell()``
     - :meth:`StudyHandle.ask <intelloptim.client.StudyHandle.ask>` and
       :meth:`StudyHandle.tell <intelloptim.client.StudyHandle.tell>`, over HTTP.
   * - ``study.optimize(func, n_trials)``
     - :meth:`StudyHandle.optimize <intelloptim.client.StudyHandle.optimize>`.
   * - ``study.enqueue_trial(params)``
     - :meth:`StudyHandle.enqueue <intelloptim.client.StudyHandle.enqueue>`,
       which can also record a point that was evaluated elsewhere.
   * - ``trial.suggest_float("x", ...)``
     - ``trial.sampled_float("x")``: the space is declared once, in
       ``create_study``.
   * - ``trial.report`` and ``trial.should_prune``
     - The same two calls on :class:`~intelloptim.client.Trial`. Reports are
       buffered and sent with ``should_prune``.
   * - ``optuna.samplers.*`` and ``optuna.pruners.*``
     - Declarative specs, such as ``intopt.samplers.TPESamplerSpec`` and
       ``intopt.pruners.MedianPrunerSpec``. The server builds the Optuna objects.
   * - A storage (SQLite, PostgreSQL)
     - The server. Evaluators never touch the database.

The one deliberate difference
-----------------------------

In Optuna, the search space is usually defined *inside* the objective, as it
runs, with ``suggest_*`` calls. Here it is declared once, when the study is
created:

.. code-block:: python

   study = client.create_study(
       study_name="pump-calibration",
       direction="minimize",
       parameters=[
           intopt.params.Float("k_friction", 1e-3, 1.0, log=True),
           intopt.params.Float("bypass_ratio", 0.0, 0.3),
       ],
       sampler_spec=intopt.samplers.TPESamplerSpec(seed=42),
   )

   def objective(trial: intopt.client.Trial) -> float:
       k = trial.sampled_float("k_friction")
       b = trial.sampled_float("bypass_ratio")
       return rms_error(k, b)

   study.optimize(objective, n_trials=200)

The server needs the whole space to sample it, to compare configurations and to
give the same study to every process that declares it. The price is that the
space cannot depend on values sampled earlier in the same trial. The objective
reads the sampled values by name with
:meth:`~intelloptim.client.Trial.sampled_float`,
:meth:`~intelloptim.client.Trial.sampled_int` and
:meth:`~intelloptim.client.Trial.sampled_categorical`.

What ``intelloptim`` adds to Optuna
-----------------------------------

* **One study, many machines.** Any process that can reach the server and holds
  a token can contribute trials. Nothing is shared except the server's URL and
  the token.
* **Studies with a stable identity.** The same declaration always resolves to
  the same study, and a process whose declaration differs gets an error rather
  than corrupting the study. See :doc:`ask-tell-enqueue`.
* **Samplers beyond Optuna's.** Besides the Optuna samplers there are built-in
  ones and quantum policy samplers, reached through the same API. See
  :doc:`sampling`.
* **Early stopping over HTTP,** which does not count against your call quota.
  See :doc:`pruning`.
* **Authentication, quotas and limits,** because the server is shared. See
  :doc:`authentication`.
* **A worker for batch systems,** so a SLURM array job can run an objective
  with no driver script. See :doc:`cluster`.
