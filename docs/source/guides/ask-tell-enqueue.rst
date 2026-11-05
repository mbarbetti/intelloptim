Studies, trials and the optimization loop
=========================================

A *study* collects the trials of one optimization. This page follows its life:
creating or joining it, running trials, driving the loop by hand, seeding it with
known points and stopping it.

Create or join a study
----------------------

:meth:`Client.create_study <intelloptim.client.Client.create_study>` declares the
whole study: its name, direction, search space, sampler and, optionally, pruner.

.. code-block:: python

   import intelloptim as intopt

   with intopt.client.Client(url="https://intelloptim.example.org") as client:
       study = client.create_study(
           study_name="pump-calibration",
           direction="minimize",
           parameters=[
               intopt.params.Float("gain", 1e-3, 1.0, log=True),
               intopt.params.Int("n_layers", 1, 8),
               intopt.params.Categorical("activation", ["relu", "tanh"]),
           ],
           sampler_spec=intopt.samplers.TPESamplerSpec(seed=42),
           pruner_spec=intopt.pruners.MedianPrunerSpec(n_startup_trials=5),
       )

The call is **idempotent and safe to run in parallel**. The study is identified
by a hash of the name, direction, parameters, sampler and pruner, together with
the identity of the caller (the user or group the token belongs to). Every
process that declares the same study therefore gets the same one, so a hundred
workers can each call ``create_study`` and then work on a single study, with
nothing to coordinate by hand. ``study.existed`` tells you whether the study was
already there.

Because the identity includes the configuration, changing a parameter range or
the sampler settings creates a *different* study. Every later request also
carries a digest of the configuration, and the server answers with an error if
it does not match the stored one (see :ref:`study-errors`), so two clients that
believe they share a study but differ are told so.

Run trials with ``optimize``
----------------------------

:meth:`StudyHandle.optimize <intelloptim.client.StudyHandle.optimize>` is the
usual loop. For each trial it asks the server for parameter values, calls your
objective with the :class:`~intelloptim.client.Trial` and tells the result:

.. code-block:: python

   def objective(trial: intopt.client.Trial) -> float:
       gain = trial.sampled_float("gain")
       layers = trial.sampled_int("n_layers")
       activation = trial.sampled_categorical("activation")
       return evaluate(gain, layers, activation)

   study.optimize(objective, n_trials=50)

* An exception raised by the objective marks that trial ``FAILED`` and the loop
  goes on, which suits evaluations that can crash (a diverging simulation, a
  preempted node). Pass ``catch=None`` to stop at the first exception instead,
  or a narrower exception class to catch only those.
* ``n_trials`` is the number of trials *this client* runs. Other clients of the
  same study run their own trials at the same time.
* Before the first trial, ``n_trials`` is compared with your remaining call
  budget, and :class:`~intelloptim.client.errors.InsufficientQuotaError` is
  raised without sending any request if it does not fit.
* To end a trial early, see :doc:`pruning`.

Read the outcome with ``study.best_trial``, which is ``None`` until one trial has
completed:

.. code-block:: python

   best = study.best_trial
   if best is not None:
       print(best.number, best.value, best.params)

Drive the loop by hand: ask and tell
------------------------------------

When the evaluation does not fit in a Python function, for example because it is
submitted to a queue and finishes later, call ``ask`` and ``tell`` yourself:

.. code-block:: python

   trial = study.ask()                               # parameters from the server
   value = evaluate(trial.sampled_float("gain"))
   study.tell(trial, value)                          # state COMPLETE by default

A trial that did not produce a value is reported with its state:

.. code-block:: python

   study.tell(trial, state=intopt.client.TrialState.FAILED)

``tell`` with ``COMPLETE`` requires a value, while ``FAILED`` and ``PRUNED`` do
not. The trial states are :class:`~intelloptim.client.TrialState`.

Seed a study with ``enqueue``
-----------------------------

:meth:`StudyHandle.enqueue <intelloptim.client.StudyHandle.enqueue>` adds a
trial whose parameter values you already know. It has two uses.

*Evaluate a given point next*, for example a known good starting configuration.
Without a value the point is queued, and the next ``ask`` of any client of the
study returns it:

.. code-block:: python

   study.enqueue({"gain": 0.05, "n_layers": 3, "activation": "relu"})

*Record a point that was evaluated elsewhere*, such as results of an earlier
campaign. With a value the trial is stored as ``COMPLETE`` at once, so the
sampler learns from it:

.. code-block:: python

   study.enqueue({"gain": 0.2, "n_layers": 2, "activation": "tanh"}, value=0.31)

The values should lie inside the declared search space; the server rejects a
recorded point outside it.

Stop a study
------------

``study.stop()`` closes a study to new work: ``ask`` and ``enqueue`` are then
refused with :class:`~intelloptim.client.errors.StudyStoppedError`. Trials that
are already running can still be told, so no result is lost.

.. _study-errors:

Errors you can handle
---------------------

Every exception raised by the SDK derives from
:class:`~intelloptim.client.errors.IntelloptimError`, and they are reached through
:mod:`intelloptim.client.errors`. The ones that call for a decision:

.. list-table::
   :header-rows: 1
   :widths: 14 38 48

   * - HTTP
     - Exception
     - What it means and what to do
   * - 409
     - :class:`~intelloptim.client.errors.ConfigDigestMismatchError`
     - The study on the server has another configuration than this client's.
       Check that every process declares the same parameters, sampler and pruner.
   * - 410
     - :class:`~intelloptim.client.errors.StudyStoppedError`
     - The study was stopped. Stop asking.
   * - 503
     - :class:`~intelloptim.client.errors.StudyRebuildingError`
     - The sampler is refitting, which is transient. Wait for
       ``retry_after_seconds`` and ask again. See :doc:`sampling`.
   * - 404
     - :class:`~intelloptim.client.errors.StudyNotFoundError` or
       :class:`~intelloptim.client.errors.TrialNotFoundError`
     - The study, or the trial told, is unknown to the server.
   * - 403
     - :class:`~intelloptim.client.errors.StudyLimitExceededError`
     - The study does not exist yet and you have reached the cap on active
       studies. A study that already exists is never refused, so you can always
       join it.
   * - 401, 429
     - :class:`~intelloptim.client.errors.IntelloptimAPIError`
     - An invalid token, or an exhausted quota; ``status_code`` and ``detail``
       tell which. See :doc:`authentication`.

One more is raised locally before any request:
:class:`~intelloptim.client.errors.InsufficientQuotaError`, described above.
