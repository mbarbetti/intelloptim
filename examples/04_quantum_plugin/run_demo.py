"""Minimal demo: run a study with the QuantumQAOA sampler via the dummy plugin.

Prerequisites:
    pip install -e examples/04_quantum_plugin/
    intelloptim server --storage sqlite:///./dev.db   (separate terminal)
    export INTELLOPTIM_TOKEN=<token from intelloptim token issue ...>
    export INTELLOPTIM_URL=http://127.0.0.1:8000
"""

from __future__ import annotations

import os

import intelloptim as intopt


def objective(trial: intopt.Trial) -> float:
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    return (x - 1.0) ** 2 + (y - 2.0) ** 2


def main() -> None:
    url = os.environ.get("INTELLOPTIM_URL", "http://127.0.0.1:8000")
    token = os.environ.get("INTELLOPTIM_TOKEN")
    verify = os.environ.get("INTELLOPTIM_SSL_VERIFY", "true").lower() != "false"

    with intopt.Client(url=url, token=token, verify=verify) as client:
        study = client.create_study(
            name="quantum-demo",
            direction="minimize",
            sampler=intopt.samplers.QuantumQAOA(reps=2, seed=42),
            parameters=[
                intopt.params.Float("x", -5.0, 5.0),
                intopt.params.Float("y", -5.0, 5.0),
            ],
        )
        print(f"study_id={study.id!r}  existed={study.existed}")

        study.optimize(objective, n_trials=20, catch=Exception)

        best = study.best_trial
        if best is not None:
            print(
                f"best trial #{best.number}: value={best.value:.4f}  params={best.params}"
            )
        else:
            print("no completed trials")


if __name__ == "__main__":
    main()
