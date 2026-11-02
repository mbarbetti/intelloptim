Samplers
========

A sampler spec tells the server how to propose the next trial; see
:doc:`../../guides/sampling` for what the families do and how to choose. Pass a
spec to :meth:`~intelloptim.client.Client.create_study` as ``sampler_spec``:

.. code-block:: python

   import intelloptim as intopt

   sampler_spec = intopt.samplers.TPESamplerSpec(seed=42)

The specs are validated when you create them, immutable, and reject unknown
fields, so a misconfigured sampler fails on your machine and not later on the
server. They are grouped by where the algorithm comes from:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Family
     - Specs
   * - :doc:`optuna`
     - ``RandomSamplerSpec``, ``QMCSamplerSpec``, ``TPESamplerSpec``,
       ``GPSamplerSpec``, ``NSGAIISamplerSpec``
   * - :doc:`builtin`
     - ``SASamplerSpec``, ``OWGSamplerSpec``
   * - :doc:`quantum`
     - ``SimQuantumGausSamplerSpec``, ``SimQuantumMixtureSamplerSpec``,
       ``QuantumGausSamplerSpec``, ``QuantumMixtureSamplerSpec``

.. toctree::
   :maxdepth: 1
   :hidden:

   optuna
   builtin
   quantum
