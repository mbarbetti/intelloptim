Extending intelloptim
=====================

You do not need this page to run studies. It is for people who deploy a server or
write a package that plugs into one: overriding how a sampler is built, or adding
a quantum provider. Both use Python *entry points*, the standard way for an
installed package to announce a component, so a plugin is a normal package that
you install next to ``intelloptim`` on the machine that runs the sampler.

How a sampler spec becomes a sampler
------------------------------------

The sampler specs you pass to ``create_study`` are only configuration. On the
server, :func:`~intelloptim.algorithms.samplers.resolve_sampler` turns a spec into
a working Optuna sampler by looking up a *resolver* registered for the spec's
``kind``. A sampler that needs a package you have not installed raises
:class:`~intelloptim.algorithms.errors.SamplerDependencyError`, whose message names
the extra to install, for example ``pip install "intelloptim[gp]"``.

Override a sampler: the ``intelloptim.samplers`` entry point
------------------------------------------------------------

A package can replace the resolver of an existing sampler kind. It registers a
callable under the entry-point group ``intelloptim.samplers``, named after the
kind (a value of ``SamplerKind``, such as
``"tpe"``). The callable receives the validated spec and returns an
``optuna.samplers.BaseSampler``:

.. code-block:: toml

   # pyproject.toml of the plugin package
   [project.entry-points."intelloptim.samplers"]
   tpe = "my_plugin.samplers:resolve_tpe"

.. code-block:: python

   # my_plugin/samplers.py
   import optuna


   def resolve_tpe(spec) -> optuna.samplers.BaseSampler:
       return optuna.samplers.TPESampler(seed=spec.seed, multivariate=True)

The server calls
:func:`~intelloptim.algorithms.samplers.load_sampler_plugins` once at start-up. An
entry point with an unknown name, or one that fails to import, is skipped with a
warning, so a newer plugin does not stop an older server from starting. A plugin
registered for a kind that already has a resolver replaces it.

.. note::

   A server builds a fresh sampler on every ``ask``. A plugin sampler must
   therefore not rely on state kept in memory between asks: whatever it needs to
   remember has to be recomputed from the study's trials. The built-in samplers
   work this way.

Stateful samplers: the ``SamplerPlugin`` protocol
-------------------------------------------------

A sampler that learns from the trial history and refits periodically, as the
quantum and OWG samplers do, can implement
:class:`~intelloptim.algorithms.protocols.SamplerPlugin`. The protocol asks for
three operations: ``ask`` proposes parameter values, ``tell`` reports an evaluated
trial, and ``rebuild`` recomputes the internal state from the past trials (it can
do nothing for stateless samplers).

Add a quantum provider: the ``intelloptim.quantum_executors`` entry point
-------------------------------------------------------------------------

The quantum policy samplers run their circuits through an *executor* and know
nothing else about the device. A new provider, such as AWS or Quantinuum, is one
module that implements an executor and registers a factory under the entry-point
group ``intelloptim.quantum_executors``, named after the provider:

.. code-block:: toml

   [project.entry-points."intelloptim.quantum_executors"]
   my_provider = "my_plugin.executor:make_executor"

A study then selects it with a
:class:`~intelloptim.core.samplers.quantum.PluginProviderSpec` carrying the same
name and any non-secret ``options``, which the factory receives as keyword
arguments:

.. code-block:: python

   sampler_spec = intopt.samplers.QuantumGausSamplerSpec(
       shots=1000,
       max_total_shots=2_000_000,
       provider=intopt.samplers.PluginProviderSpec(
           provider="my_provider", options={"region": "eu-west-1"}
       ),
   )

Two rules keep this safe:

* **A spec never carries credentials.** Keys that look like secrets are rejected in
  ``options``. The plugin reads its credentials from the environment of the
  worker that runs it.
* **Hardware is opt-in on the worker.** The worker must have
  ``INTELLOPTIM_ALLOW_QUANTUM_HARDWARE=1`` set to run on a real device, so a study
  cannot spend hardware time on a worker that did not allow it.

Plugins are found on demand inside the worker, so they need no start-up call.

Reference
---------

.. autofunction:: intelloptim.algorithms.samplers.resolve_sampler

.. autofunction:: intelloptim.algorithms.samplers.load_sampler_plugins

.. autoclass:: intelloptim.algorithms.protocols.SamplerPlugin

.. autoexception:: intelloptim.algorithms.errors.SamplerDependencyError

.. autoexception:: intelloptim.algorithms.errors.SamplerNotImplementedError

.. autoclass:: intelloptim.algorithms.backend.quantum.commons.executor.QuantumExecutor
   :no-members:
