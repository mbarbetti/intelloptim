Pruning: stopping trials early
==============================

When an objective is iterative, such as a training loop, a trial often shows early
that it will not be competitive. *Pruning* lets the study stop it, which saves
the compute that the rest of the run would have used. Where sampling decides
*where to look* (:doc:`sampling`), pruning decides *whether to keep going*.

Pruning needs two things: a pruner declared in the study, and an objective that
reports intermediate values.

Declare a pruner
----------------

.. code-block:: python

   study = client.create_study(
       ...,
       pruner_spec=intopt.pruners.MedianPrunerSpec(n_startup_trials=5, n_warmup_steps=3),
   )

Without a ``pruner_spec`` no trial is ever pruned.

Report and ask
--------------

Inside the objective, call :meth:`trial.report <intelloptim.client.Trial.report>`
with the intermediate value and its step, and
:meth:`trial.should_prune <intelloptim.client.Trial.should_prune>` to ask the study.
When it returns ``True``, raise :class:`~intelloptim.client.errors.TrialPruned`:

.. code-block:: python

   def objective(trial: intopt.client.Trial) -> float:
       lr = trial.sampled_float("lr")
       for epoch in range(20):
           loss = train_one_epoch(lr)
           trial.report(loss, epoch)
           if trial.should_prune():
               raise intopt.client.errors.TrialPruned
       return loss

:meth:`~intelloptim.client.StudyHandle.optimize` recognises the exception, records
the trial as ``PRUNED`` with the last value you reported, and goes on to the next
trial. It is not logged as a failure, and the ``catch`` argument does not affect
it. ``TrialPruned`` deliberately does not derive from
:class:`~intelloptim.client.errors.IntelloptimError`, so a broad
``except IntelloptimError`` does not swallow it.

What is sent, and what it costs
-------------------------------

``report`` only stores the value locally. ``should_prune`` sends every value
reported since its previous call in a single request, and returns the answer for
all of them. If nothing new was reported it sends nothing. Pruning requests do
**not** count against the monthly call quota, but the server limits them to 60
per minute per API token. Report every step if useful, but check pruning at a
cadence no faster than once per second; unsent values remain buffered locally.
Each step can be reported once, and values must be finite. If the server returns
a rate-limit response, the SDK raises
:class:`~intelloptim.client.errors.PruneRateLimitError`, whose
``retry_after_seconds`` attribute tells the caller when to retry.

Choose a pruner
---------------

All pruners are Optuna-backed; see :doc:`../api/pruners` for every setting.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pruner
     - Rule
   * - ``MedianPrunerSpec``
     - Stops a trial whose best value so far is worse than the median of the
       completed trials at the same step. A solid default for iterative training.
   * - ``PercentilePrunerSpec``
     - The same idea with a chosen percentile: a low percentile prunes
       aggressively, a high one leniently.
   * - ``ThresholdPrunerSpec``
     - Stops a trial as soon as a value leaves the interval ``[lower, upper]`` or
       is ``nan``. For an absolute requirement, such as a loss that must stay
       below a bound.
   * - ``NopPrunerSpec``
     - Never prunes; the baseline.

The median and percentile rules need completed trials to compare against, so
nothing is pruned during the first ``n_startup_trials``, and
``n_warmup_steps`` protects the noisy first steps of every trial.
