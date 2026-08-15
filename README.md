<div align="center">
    <img src="https://raw.githubusercontent.com/mbarbetti/intelloptim/main/.github/images/intelloptim-logo-white.png" width="800"/>
</div>

# intelloptim

A general black-box optimization framework, built on top of
[Optuna](https://optuna.org/) and exposed as a service. Distributed
evaluators on any infrastructure (HPC, cloud, on-prem) collaborate on a
single study via a coordinating server, with a pluggable sampler layer
that treats classical and quantum samplers uniformly at the API surface.

```python
import intelloptim as intopt

client = intopt.Client(url="https://intelloptim.local", token=...)

study = client.create_study(
    name="pump-cal-2026-Q2",
    direction="minimize",
    sampler=intopt.samplers.TPE(seed=42),
    parameters=[
        intopt.params.Float("k_friction", 1e-3, 1.0, log=True),
        intopt.params.Float("bypass_ratio", 0.0, 0.3),
    ],
)

def objective(trial: intopt.Trial) -> float:
    k = trial.suggest_float("k_friction", 1e-3, 1.0, log=True)
    b = trial.suggest_float("bypass_ratio", 0.0, 0.3)
    return rms_error_against_telemetry(k, b)

study.optimize(objective, n_trials=200)
print(study.best_trial)
```

## Lineage

`intelloptim` is an evolution of **Hopaas** — *Hyperparameter Optimization
as a Service* (Barbetti & Anderlini, 2023, [arXiv:2301.05522](https://arxiv.org/abs/2301.05522)).
It generalises the philosophy beyond hyperparameter tuning to any
small-to-moderate-dimensional black-box optimization problem (system
calibration, model tuning, etc.) and adds:

- Federation by content-addressed study identity — independent client
  processes on different infrastructures collaborate on the same study by
  sharing only an API token.
- A separate optimizer compute plane (Celery + Redis) for samplers whose
  `ask` step is expensive (heavy Bayesian-optimization surrogates,
  hybrid quantum-classical loops).
- Pause / rebuild semantics for samplers that periodically refit on
  accumulated trials.
- Quantum-classical sampler interchangeability at the API surface.
- A live monitoring frontend (React) with login, API token CRUD, and
  Optuna-dashboard-style plots.

See `docs/architecture.md` for the design details.

## Status

Early development — see `docs/` and the project plan for the current phase.
