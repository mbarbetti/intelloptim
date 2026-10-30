Getting started
---------------

Install the base SDK:

.. code-block:: console

   pip install intelloptim

The SDK requires an intelloptim server URL and an API token. Declare the search
space once, then read server-sampled values in the objective:

.. code-block:: python

   import intelloptim as intopt

   with intopt.client.Client(url="https://intelloptim.example.org") as client:
       study = client.create_study(
           study_name="pump-calibration",
           direction="minimize",
           parameters=[intopt.params.Float("gain", 1e-3, 1.0, log=True)],
           sampler_spec=intopt.samplers.TPESamplerSpec(seed=42),
       )

       def objective(trial: intopt.client.Trial) -> float:
           return evaluate_gain(trial.sampled_float("gain"))

       study.optimize(objective, n_trials=50)
