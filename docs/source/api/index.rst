API reference
=============

The names a script uses to define and run a study. Everything is reached from
the package alias::

   import intelloptim as intopt

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Page
     - Contents
   * - :doc:`client`
     - ``intopt.client``: ``Client``, ``StudyHandle``, ``Trial``, ``TrialState``
       and the errors.
   * - :doc:`parameters`
     - ``intopt.params``, to declare the search space, and ``intopt.objective``.
   * - :doc:`samplers/index`
     - ``intopt.samplers``: Optuna, built-in and quantum samplers.
   * - :doc:`pruners`
     - ``intopt.pruners``: the early-stopping rules.
   * - :doc:`worker`
     - The ``intelloptim-worker`` command.

How these pieces fit together is explained in the :doc:`user guide
<../guides/optuna-and-intelloptim>`. The plugin interfaces for server operators
and sampler authors are described in :doc:`../guides/extending`.

.. toctree::
   :maxdepth: 2
   :hidden:

   client
   parameters
   samplers/index
   pruners
   worker
