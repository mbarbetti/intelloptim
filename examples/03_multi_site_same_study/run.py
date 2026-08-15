"""Multi-site federation demo: two terminals, one cooperative study.

Run this script in two separate terminals on (possibly) two separate
machines, pointing at the same intelloptim server with the same API
token. Both invocations resolve to the *same* content-addressed
``study_id`` and contribute trials cooperatively to one Optuna study on
the server.

Usage
-----

In every terminal that should participate::

    export INTELLOPTIM_URL=http://intelloptim.local:8000
    export INTELLOPTIM_TOKEN=$(cat /path/to/shared/token)
    python examples/03_multi_site_same_study/run.py --site alpha --n-trials 25

Each instance prints the resolved ``study_id`` and ``existed`` flag.
The first to start sees ``created``; every subsequent one sees
``joined``. Stop a process at any time and restart it — it will pick
up where it left off because the study identity is content-addressed.

What this demonstrates
----------------------

The whole point of intelloptim's federation primitive: independent
clients with no shared state beyond the API token and the spec land on
the same study by construction. Useful for:

* HPC + cloud co-training (one site has GPU, another has CPU-only
  pre-processing).
* On-prem + lab-machine collaboration where neither holds the truth.
* Restart-friendly evaluators that may die mid-run.
"""

from __future__ import annotations

import argparse
import os
import time

import intelloptim as intopt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--url",
        default=os.environ.get("INTELLOPTIM_URL", "http://127.0.0.1:8000"),
    )
    p.add_argument("--site", required=True, help="Label for log lines")
    p.add_argument(
        "--n-trials", type=int, default=25, help="Trials this process contributes"
    )
    p.add_argument("--study-name", default="multi-site-demo", help="Shared study label")
    return p.parse_args()


def objective(trial: intopt.Trial) -> float:
    """Minimise the Branin function on a tiny 2-d toy."""
    x = trial.suggest_float("x", -5.0, 10.0)
    y = trial.suggest_float("y", 0.0, 15.0)
    a, b, c = 1.0, 5.1 / (4 * 3.141592653589793**2), 5.0 / 3.141592653589793
    r, s, t = 6.0, 10.0, 1.0 / (8 * 3.141592653589793)
    return (
        a * (y - b * x**2 + c * x - r) ** 2
        + s * (1 - t) * float(__import__("math").cos(x))
        + s
    )


def main() -> None:
    args = parse_args()
    verify = os.environ.get("INTELLOPTIM_SSL_VERIFY", "true").lower() != "false"

    with intopt.Client(url=args.url, verify=verify) as client:
        study = client.create_study(
            name=args.study_name,
            direction="minimize",
            sampler=intopt.samplers.TPE(seed=42),
            parameters=[
                intopt.params.Float("x", -5.0, 10.0),
                intopt.params.Float("y", 0.0, 15.0),
            ],
        )
        print(
            f"[{args.site}] study {study.id} "
            f"({'joined' if study.existed else 'created'})"
        )

        for i in range(args.n_trials):
            trial = study.ask(worker_id=args.site)
            value = objective(trial)
            study.tell(trial, value)
            print(
                f"[{args.site}] trial #{trial.number} → x={trial.params['x']:.4f}, "
                f"y={trial.params['y']:.4f}, value={value:.4f}"
            )
            # Tiny pause to keep the two terminals visibly interleaving.
            time.sleep(0.05)

        best = study.best_trial
        if best is not None:
            print(
                f"[{args.site}] best so far: trial #{best.number}, "
                f"value={best.value:.4f}, params={best.params}"
            )


if __name__ == "__main__":
    main()
