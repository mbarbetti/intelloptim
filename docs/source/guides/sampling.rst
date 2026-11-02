Sampling: how the space is explored
===================================

A **sampler** decides which point to try next. Each time a client asks for a
trial, the sampler looks at the search space you declared and at the trials
that have finished, and proposes parameter values. A good sampler balances two
needs: *exploring* regions that have not been tried and *exploiting* the regions
that already gave good results.

You choose the sampler once, when you create the study, with a *sampler spec*, a
small validated configuration object:

.. code-block:: python

   study = client.create_study(
       ...,
       sampler_spec=intopt.samplers.TPESamplerSpec(seed=42, n_startup_trials=20),
   )

The server builds the actual sampler from the spec, so the machine that runs your
objective needs nothing but the SDK. The sampler is part of the study's identity:
a different sampler, or different settings, is a different study (see
:doc:`ask-tell-enqueue`).

Three families
--------------

The specs are grouped by where the algorithm comes from. All of them are used the
same way and are reached through ``intopt.samplers``.

**Optuna samplers.** Thin specs over `Optuna's samplers <https://optuna.org/>`_;
see :doc:`../api/samplers/optuna`.

* ``RandomSamplerSpec``: independent uniform draws. A baseline, with no warm-up.
* ``QMCSamplerSpec``: low-discrepancy sequences (Sobol or Halton), which fill the
  space more evenly than random draws.
* ``TPESamplerSpec``: Tree-structured Parzen Estimator, which models where good
  and poor trials fall. The recommended default for most single-objective
  studies.
* ``GPSamplerSpec``: Gaussian-process Bayesian optimization, for
  low-dimensional objectives that are expensive to evaluate. Needs the ``gp``
  extra on the machine that hosts the sampler.
* ``NSGAIISamplerSpec``: the NSGA-II genetic algorithm, which evolves a
  population of candidates.

**Built-in samplers.** Algorithms implemented in ``intelloptim`` itself; see
:doc:`../api/samplers/builtin`.

* ``SASamplerSpec``: simulated annealing. Each trial is a neighbour of the
  current accepted point, and worse points are accepted with a probability that
  shrinks as the temperature cools, moving the search from exploration to
  exploitation. Suited to cheap objectives where a local, stochastic search is
  enough.
* ``OWGSamplerSpec``: objective-weighted Gaussian. It starts from a space-filling
  design and then samples from a Gaussian fitted to the completed trials, with
  better trials weighing more.

**Quantum samplers.** Policy samplers whose proposal distribution comes from a
parameterized quantum circuit, trained on the objective values of finished
trials; see :doc:`../api/samplers/quantum`. Each algorithm, ``Gaus`` and
``Mixture``, exists as a simulated sampler (``SimQuantumGausSamplerSpec``,
``SimQuantumMixtureSamplerSpec``) and as a hardware sampler
(``QuantumGausSamplerSpec``, ``QuantumMixtureSamplerSpec``). The simulated ones
need the ``quantum`` extra on the sampler's machine. The hardware ones run on a
real device, see its noise, and cost queue time and money, so they require an
explicit shot budget and a provider. A spec never carries credentials, which the
worker reads from its own environment.

Which sampler?
--------------

There is no best sampler for every problem, and no benchmark results are
published for these yet, so treat this as a starting point.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Your situation
     - A reasonable first choice
   * - You want a default and have no prior knowledge.
     - ``TPESamplerSpec``
   * - You have few trials, want to check the search space or need broad coverage.
     - ``RandomSamplerSpec`` or ``QMCSamplerSpec``
   * - Each evaluation is expensive and the space has few parameters.
     - ``GPSamplerSpec``
   * - Evaluations are cheap and a local search is acceptable.
     - ``SASamplerSpec``
   * - You study quantum policy samplers, or want to compare them with classical
       ones on the same problem.
     - the ``SimQuantum*`` specs first, hardware specs only when you need real
       noise
   * - You can afford a refit pause every few trials.
     - ``OWGSamplerSpec`` or a quantum sampler (see rebuilds below)

Warm-up
-------

Most model-based samplers start with a few trials drawn without a model, because
there is nothing to learn from yet. The number is ``n_startup_trials`` for TPE,
GP and SA, and OWG follows a space-filling design until its first rebuild. The
first trials of a study therefore look random: that is expected.

Reproducibility
---------------

Every spec takes a ``seed``. For the built-in and quantum samplers, ``seed=None``
seeds from the study name, so a study stays reproducible. They keep no hidden
state between requests: the suggestions are derived from the study's history, so
the same history gives the same suggestions. For the Optuna samplers,
``seed=None`` lets Optuna pick a seed, so pass one when you need repeatable runs.

Samplers that refit: rebuilds
-----------------------------

``OWGSamplerSpec`` and the quantum samplers learn from the completed trials in
periodic *rebuild* steps rather than after every trial, because a fit can be
slow. They are configured with ``rebuild_every``, the number of completed trials
between rebuilds (see :class:`~intelloptim.core.samplers.base.RebuildPolicy`).
Between two rebuilds, trials are drawn from the last model, which lags slightly
behind the study.

While a rebuild is running, the study cannot serve ``ask``. The server answers
``503`` with a ``Retry-After`` hint and the client raises
:class:`~intelloptim.client.errors.StudyRebuildingError`. This is transient:
wait ``retry_after_seconds`` and ask again. ``tell`` keeps working during a
rebuild, so results are never lost.

.. code-block:: python

   import time

   from intelloptim.client.errors import StudyRebuildingError

   while True:
       try:
           trial = study.ask()
           break
       except StudyRebuildingError as exc:
           time.sleep(exc.retry_after_seconds or 5)

:meth:`~intelloptim.client.StudyHandle.optimize` does not retry for you: the error
ends the run. For a sampler that rebuilds, drive the loop by hand, as in
:doc:`ask-tell-enqueue`, with the retry around ``ask``, then ``tell`` the result.
