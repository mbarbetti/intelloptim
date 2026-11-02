Installation
============

``intelloptim`` requires Python 3.11 or newer. Install the SDK from PyPI:

.. code-block:: console

   pip install intelloptim

This is all you need to write an objective and run a study against a server.
Samplers with heavier requirements are optional extras:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Extra
     - Adds
     - Needed for
   * - ``gp``
     - ``torch``, ``scipy``
     - the Gaussian-process sampler
   * - ``qmc``
     - ``scipy``
     - the quasi-Monte Carlo sampler
   * - ``quantum``
     - ``qiskit``
     - the simulated quantum samplers
   * - ``quantum-ibm``
     - ``qiskit-ibm-runtime``
     - quantum samplers on IBM hardware
   * - ``samplers``
     - ``gp``, ``qmc`` and ``quantum``
     - everything a server needs to resolve any sampler

The extras install the packages that a sampler needs where the sampler *runs*.
With a client-only setup you do not need them, because the sampler is created
by the server; the extras matter on the machine that hosts the server or the
sampler workers. A sampler whose package is missing raises an error that names
the extra to install. For example:

.. code-block:: console

   pip install "intelloptim[gp]"

Check the installation:

.. code-block:: console

   python -c "import intelloptim as intopt; print(intopt.__version__)"

What you also need
------------------

* **A server.** The SDK talks to an intelloptim server over HTTP(S). The server
  is not part of this package, so ask the person who runs it for its URL.
* **An API token**, issued by that server. See :doc:`guides/authentication` for
  how the client finds and protects it.

Then continue with :doc:`guides/optuna-and-intelloptim`.
