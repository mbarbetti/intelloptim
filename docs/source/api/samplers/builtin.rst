Built-in samplers
=================

Algorithms implemented in ``intelloptim`` itself. They keep no state between
requests: every suggestion is derived from the study's completed trials, so they
are reproducible for a given history.

.. currentmodule:: intelloptim.core.samplers

.. autoclass:: SASamplerSpec
   :no-members:

.. autoclass:: OWGSamplerSpec
   :no-members:

Rebuild policy
--------------

``OWGSamplerSpec`` refits its model in periodic rebuild steps, as the quantum
samplers do. The policy that decides when is described by ``RebuildPolicy``;
its ``rebuild_every`` argument on the spec sets the interval in completed trials.
While a rebuild runs, ``ask`` is answered with a retry hint (see
:doc:`../../guides/sampling`).

.. autoclass:: intelloptim.core.samplers.base.RebuildPolicy
   :no-members:
