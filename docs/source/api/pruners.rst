Pruners
=======

A pruner decides, from the intermediate values a trial reports, whether the trial
should stop early. Pass one to
:meth:`~intelloptim.client.Client.create_study` as ``pruner_spec``; without one,
nothing is pruned. The pruners are backed by Optuna, and how to use them is
described in :doc:`../guides/pruning`.

.. code-block:: python

   pruner_spec = intopt.pruners.MedianPrunerSpec(n_startup_trials=5)

.. currentmodule:: intelloptim.core.pruners

.. autoclass:: MedianPrunerSpec
   :no-members:

.. autoclass:: PercentilePrunerSpec
   :no-members:

.. autoclass:: ThresholdPrunerSpec
   :no-members:

.. autoclass:: NopPrunerSpec
   :no-members:
