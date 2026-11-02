Search space and objectives
===========================

Parameters
----------

The parameters of a study are declared once, in
:meth:`~intelloptim.client.Client.create_study`, with the classes of
``intopt.params``. They are the same classes as ``FloatSpec``, ``IntSpec`` and
``CategoricalSpec``, under shorter names, and are documented here as such:

====================================  ============================
Name in your script                   Class
====================================  ============================
``intopt.params.Float``               :class:`FloatSpec`
``intopt.params.Int``                 :class:`IntSpec`
``intopt.params.Categorical``         :class:`CategoricalSpec`
====================================  ============================

.. currentmodule:: intelloptim.core.parameters

.. autoclass:: FloatSpec
   :no-members:

.. autoclass:: IntSpec
   :no-members:

.. autoclass:: CategoricalSpec
   :no-members:

.. currentmodule:: intelloptim

Objectives
----------

.. autofunction:: intelloptim.objective

.. autoexception:: intelloptim.core.objective.ObjectiveAlreadyRegisteredError
