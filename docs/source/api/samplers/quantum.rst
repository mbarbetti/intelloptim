Quantum samplers
================

Policy samplers whose proposal distribution comes from a parameterized quantum
circuit. The circuit and a small classical head are trained with PPO on the
objective values of completed trials. Two algorithms exist, each as a simulated
and a hardware sampler:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - Simulated (ideal)
     - Hardware
   * - **Gaus**: one bounded Gaussian policy
     - ``SimQuantumGausSamplerSpec``
     - ``QuantumGausSamplerSpec``
   * - **Mixture**: a mixture of Gaussians whose weights are measurement
       probabilities
     - ``SimQuantumMixtureSamplerSpec``
     - ``QuantumMixtureSamplerSpec``

The simulated samplers need the ``quantum`` extra, and the hardware samplers
additionally ``quantum-ibm`` for IBM devices (see :doc:`../../installation`). They
support float, integer and categorical parameters. Hardware samplers see real
noise and cost queue time and money, so a spec must give an explicit shot budget
(``max_total_shots``) and a provider, and the worker has to allow hardware runs
explicitly (see :doc:`../../guides/extending`). A spec never contains credentials.

All four samplers rebuild periodically, so ``ask`` can be answered with a retry
hint during an update (see :doc:`../../guides/sampling`).

.. currentmodule:: intelloptim.core.samplers

The four samplers
-----------------

.. autoclass:: SimQuantumGausSamplerSpec
   :no-members:

.. autoclass:: SimQuantumMixtureSamplerSpec
   :no-members:

.. autoclass:: QuantumGausSamplerSpec
   :no-members:

.. autoclass:: QuantumMixtureSamplerSpec
   :no-members:

Settings
--------

The settings of a sampler are the union of the groups below: those shared by every
quantum sampler, those of its algorithm, and those of where it runs.

.. autoclass:: intelloptim.core.samplers.quantum.BaseQuantumSamplerSpec
   :no-members:

.. autoclass:: intelloptim.core.samplers.quantum.BaseQuantumGausSpec
   :no-members:

.. autoclass:: intelloptim.core.samplers.quantum.BaseQuantumMixtureSpec
   :no-members:

.. autoclass:: intelloptim.core.samplers.quantum._SimulatedFields
   :no-members:

.. autoclass:: intelloptim.core.samplers.quantum._HardwareFields
   :no-members:

Providers
---------

A hardware sampler runs its circuits on a provider.

.. autoclass:: IBMRuntimeProviderSpec
   :no-members:

.. autoclass:: PluginProviderSpec
   :no-members:
