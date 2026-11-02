Optuna samplers
===============

Specs for samplers that the server builds from `Optuna <https://optuna.org/>`_.
Each links to the Optuna documentation of the sampler it configures. ``GPSamplerSpec``
needs the ``gp`` extra and ``QMCSamplerSpec`` the ``qmc`` extra on the machine that
hosts the sampler (see :doc:`../../installation`).

.. currentmodule:: intelloptim.core.samplers

.. autoclass:: RandomSamplerSpec
   :no-members:

.. autoclass:: QMCSamplerSpec
   :no-members:

.. autoclass:: TPESamplerSpec
   :no-members:

.. autoclass:: GPSamplerSpec
   :no-members:

.. autoclass:: NSGAIISamplerSpec
   :no-members:
