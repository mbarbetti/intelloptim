Asking, telling, and pruning
----------------------------

Use :meth:`intelloptim.client.StudyHandle.optimize` for the usual sequential
loop. To control the lifecycle directly, call ``ask()`` and ``tell()``:

.. code-block:: python

   trial = study.ask()
   value = evaluate(trial.sampled_float("x"))
   study.tell(trial, value)

For iterative objectives, call ``trial.report`` and then
``trial.should_prune``. Raise
:class:`intelloptim.client.errors.TrialPruned` when it returns ``True``;
``optimize`` records the trial as pruned.
