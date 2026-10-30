Evaluator worker
----------------

``intelloptim-worker evaluator`` imports an objective, resolves its study, and
runs a chosen number of trials. It is designed for batch systems such as SLURM
arrays, where workers sharing a study specification contribute to one study.

.. code-block:: console

   intelloptim-worker evaluator --server https://intelloptim.example.org \
       --import my_package.objectives:objective --n-trials 50
