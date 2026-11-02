intelloptim
===========

**Distributed black-box optimization**: many evaluators, on any infrastructure,
collaborate on one study through a coordinating server. ``intelloptim`` is the
Python SDK and the sampler layer; it wraps `Optuna <https://optuna.org/>`_'s
ask/tell model and adds what a shared, multi-site service needs.

.. code-block:: python

   import intelloptim as intopt

   with intopt.client.Client(url="https://intelloptim.example.org") as client:
       study = client.create_study(
           study_name="pump-calibration",
           direction="minimize",
           parameters=[intopt.params.Float("gain", 1e-3, 1.0, log=True)],
           sampler_spec=intopt.samplers.TPESamplerSpec(seed=42),
       )
       study.optimize(lambda trial: evaluate(trial.sampled_float("gain")), 50)

Where it comes from
-------------------

``intelloptim`` is the evolution of **Hopaas**, *Hyperparameter OPtimization As
A Service*, described by M. Barbetti and L. Anderlini in
`arXiv:2301.05522 <https://arxiv.org/abs/2301.05522>`_ (*Hyperparameter
Optimization as a Service on INFN Cloud*, ACAT 2022).

Hopaas was built for a concrete problem. Tuning a machine-learning model means
running many independent trainings, the *trials*, and the more of them run in
parallel the more of the search space can be explored. In practice, the paper
points out, parallel tuning is "limited by the ability of accessing securely a
common database" with the earlier and ongoing trials, and the computing power
that is available is often opportunistic: GPUs offered by different providers,
each with its own restrictions and regulations. Existing hyperparameter
optimization services were hard to adopt in that setting.

Hopaas answered with a web service that coordinates the training instances
through a small REST API of three calls (``ask``, ``tell`` and
``should_prune``), using Bayesian optimization from Optuna. A computing node
only needs a network connection to the server, wherever it runs, and every node
that asks for a trial of the same study contributes to the same optimization.
A web dashboard tracks the studies, and users generate revocable API tokens.
The service ran on INFN Cloud and tuned the generative adversarial networks that
parameterize the response of the LHCb detector in its ultra-fast simulation,
with studies of hundreds of trials fed by more than twenty concurrent nodes at
INFN, CERN, CINECA and commercial clouds.

What intelloptim adds
---------------------

``intelloptim`` keeps that philosophy and generalizes it from hyperparameter
tuning to any small-to-moderate-dimensional black-box optimization problem, such
as the calibration of a system or the tuning of a model. It is also built to be
more robust and more secure.

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Topic
     - Hopaas, as published
     - ``intelloptim``
   * - Scope
     - Hyperparameter optimization of machine-learning models.
     - Any black-box objective: simulations, calibrations, models.
   * - Token
     - The API token is part of the request path.
     - The token is sent as a bearer header, never in the URL, and it is never
       part of a study's identity. The client refuses token files that other
       users can read.
   * - Samplers
     - Bayesian optimization through Optuna.
     - Optuna samplers, the built-in ``SA`` and ``OWG`` samplers and quantum
       policy samplers, all behind the same API.
   * - Calls
     - ``ask``, ``tell``, ``should_prune``.
     - The same, plus ``enqueue`` to seed or queue points and ``stop`` to close
       a study.

Beyond the table:

* **Studies are content-addressed.** The identity of a study is a hash of its
  name, direction, search space, sampler and pruner, scoped to the user or
  group of the caller. Independent processes that declare the same study land
  on the same one, and a process whose configuration differs is refused with a
  clear error instead of silently mixing results. See
  :doc:`guides/ask-tell-enqueue`.
* **A search space declared once.** The parameters are declared when the study
  is created, and objectives read the sampled values by name, which is what lets
  the server validate and share them. See :doc:`guides/optuna-and-intelloptim`.
* **Samplers that refit.** Samplers with an expensive fit, such as the quantum
  ones, are rebuilt in the background from the study history, and clients are
  told to retry meanwhile. See :doc:`guides/sampling`.
* **Quotas and limits** on calls and studies, reported with dedicated errors.
* **A ready-made worker** for batch systems such as SLURM, which takes the
  worker and site identity from the job environment. See :doc:`guides/cluster`.

The package contains the client, the search-space and sampler declarations and
the evaluator worker. The server is a separate project and is deployed
separately; this documentation covers what you need to use one.

.. note::

   ``intelloptim`` is in early development (alpha). The API may still change
   between minor versions, and the :doc:`changelog` lists every change.

.. toctree::
   :maxdepth: 1
   :caption: Start

   installation

.. toctree::
   :maxdepth: 2
   :caption: User guide

   guides/optuna-and-intelloptim
   guides/ask-tell-enqueue
   guides/sampling
   guides/pruning
   guides/authentication
   guides/cluster
   guides/extending

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog
