Client SDK
==========

Everything here lives in ``intelloptim.client`` and is reached as
``intopt.client``.

.. code-block:: python

   import intelloptim as intopt

   with intopt.client.Client(url="https://intelloptim.example.org") as client:
       study = client.create_study(...)

.. currentmodule:: intelloptim.client

Client
------

.. autoclass:: Client
   :members: create_study, close, url

Studies
-------

.. autoclass:: StudyHandle
   :members: id, existed, ask, tell, enqueue, optimize, stop, best_trial

Trials
------

.. autoclass:: Trial
   :members: sampled_float, sampled_int, sampled_categorical, report,
             should_prune, intermediate_values, trial_id, number, params, value

.. autoclass:: TrialState
   :members:
   :undoc-members:

Errors
------

.. automodule:: intelloptim.client.errors
   :members:
   :show-inheritance:
