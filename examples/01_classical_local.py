"""Single-machine smoke run against a local intelloptim server.

Prerequisites
-------------

Bring up an intelloptim server in one terminal::

    $ intelloptim server --storage sqlite:///./dev.db

In another terminal, export your token and run::

    $ export INTELLOPTIM_TOKEN=intopt_...
    $ python examples/01_classical_local.py

Against the Docker Compose stack add::

    $ export INTELLOPTIM_URL=https://localhost
    $ export INTELLOPTIM_SSL_VERIFY=false

What it does
------------

Minimises ``(x - 3)**2 + (y - 2)**2`` on ``x in [-10, 10], y in [0, 5]``
using a TPE sampler running inside the server. The whole loop is over
HTTP — every ``study.ask()`` is a POST, every ``study.tell()`` is a POST,
and the script can be killed mid-run and re-launched against the same
study by simply pointing at the same server with the same parameters.

Two evaluators in parallel
--------------------------

To see federation in action, run this same script in two terminals
simultaneously (with the same token and server URL). Both will land on
the same content-addressed study and contribute trials cooperatively.
"""

from __future__ import annotations

import os

import intelloptim as intopt


def objective(trial: intopt.Trial) -> float:
    x = trial.suggest_float("x", -10.0, 10.0)
    y = trial.suggest_float("y", 0.0, 5.0)
    return (x - 3.0) ** 2 + (y - 2.0) ** 2


def main() -> None:
    url = os.environ.get("INTELLOPTIM_URL", "http://127.0.0.1:8000")
    verify = os.environ.get("INTELLOPTIM_SSL_VERIFY", "true").lower() != "false"

    with intopt.Client(url=url, verify=verify) as client:
        study = client.create_study(
            name="classical-local-demo",
            direction="minimize",
            sampler=intopt.samplers.TPE(seed=42),
            parameters=[
                intopt.params.Float("x", -10.0, 10.0),
                intopt.params.Float("y", 0.0, 5.0),
            ],
        )
        print(f"study {study.id} ({'joined' if study.existed else 'created'})")

        study.optimize(objective, n_trials=50)

        best = study.best_trial
        if best is None:
            print("no completed trials yet")
            return
        print(
            f"best trial #{best.number}: value={best.value:.6f}, params={best.params}"
        )


if __name__ == "__main__":
    main()
