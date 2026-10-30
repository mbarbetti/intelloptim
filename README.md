<!--
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mbarbetti/intelloptim/main/docs/images/intelloptim-logo-w-bkg.png">
    <img alt="intelloptim" src="https://raw.githubusercontent.com/mbarbetti/intelloptim/main/docs/images/intelloptim-logo-black.png" width="600">
  </picture>
</div>
-->

# intelloptim

[![CI](https://github.com/mbarbetti/intelloptim/actions/workflows/ci.yml/badge.svg)](https://github.com/mbarbetti/intelloptim/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/intelloptim.svg)](https://pypi.org/project/intelloptim/)
[![Python](https://img.shields.io/pypi/pyversions/intelloptim.svg)](https://pypi.org/project/intelloptim/)
[![License](https://img.shields.io/pypi/l/intelloptim.svg)](https://github.com/mbarbetti/intelloptim/blob/main/LICENSE)
[![Documentation](https://readthedocs.org/projects/intelloptim/badge/?version=latest)](https://intelloptim.readthedocs.io)

Distributed black-box optimization built on [Optuna](https://optuna.org/)'s
ask/tell API. Evaluators on any infrastructure (HPC, cloud, on-prem)
collaborate on a single study through a coordinating server, and the sampler
layer treats classical and quantum samplers uniformly.

This package is the client SDK and the sampler layer. It talks to an
intelloptim server over HTTP; the server itself is not part of this package.

## Installation

```bash
pip install intelloptim
```

Samplers with heavier requirements are optional extras:

| Extra | Adds | For |
|---|---|---|
| `gp` | `torch`, `scipy` | the Gaussian-process sampler |
| `qmc` | `scipy` | the quasi-Monte Carlo sampler |
| `quantum` | `qiskit` | the simulated quantum samplers |
| `quantum-ibm` | `qiskit-ibm-runtime` | quantum samplers on IBM hardware |
| `samplers` | `gp`, `qmc`, `quantum` | everything a server needs to resolve any sampler |

Python 3.11 or newer is required.

## Quick start

```python
import intelloptim as intopt

client = intopt.client.Client(url="https://intelloptim.local", token=...)

study = client.create_study(
    study_name="pump-cal-2026-Q2",
    direction="minimize",
    sampler_spec=intopt.samplers.TPESamplerSpec(seed=42),
    parameters=[
        intopt.params.Float("k_friction", 1e-3, 1.0, log=True),
        intopt.params.Float("bypass_ratio", 0.0, 0.3),
    ],
)

def objective(trial: intopt.client.Trial) -> float:
    k = trial.sampled_float("k_friction")
    b = trial.sampled_float("bypass_ratio")
    return rms_error_against_telemetry(k, b)

study.optimize(objective, n_trials=200)
print(study.best_trial)
```

The search space is declared once, in `create_study`. The objective only
reads the values the server sampled, by parameter name (`sampled_float`,
`sampled_int`, `sampled_categorical`).

Training in steps? Report each intermediate value and let the study's pruner
decide whether to go on (declare one with `pruner_spec=` in `create_study`):

```python
def objective(trial: intopt.client.Trial) -> float:
    for epoch in range(20):
        loss = train_one_epoch(trial.sampled_float("lr"))
        trial.report(loss, epoch)
        if trial.should_prune():
            raise intopt.client.errors.TrialPruned
    return loss
```

`should_prune()` sends the values reported since the last call in a single
request and does not count against your call quota.

**Coming from Optuna?** Here the search space is defined once, when the study
is created, and not dynamically inside the objective as Optuna's `suggest_*`
methods allow. Declare the parameters in `create_study` and read them back with
the `sampled_*` accessors.

## Lineage

`intelloptim` is an evolution of **Hopaas** — *Hyperparameter Optimization
as a Service* (Barbetti & Anderlini, 2023, [arXiv:2301.05522](https://arxiv.org/abs/2301.05522)).
It generalises the philosophy beyond hyperparameter tuning to any
small-to-moderate-dimensional black-box optimization problem (system
calibration, model tuning, etc.) and adds:

- Federation by content-addressed study identity — independent client
  processes on different infrastructures collaborate on the same study by
  sharing only an API token.
- A separate optimizer compute plane for samplers whose `ask` step is
  expensive (heavy Bayesian-optimization surrogates, hybrid quantum-classical
  loops).
- Pause / rebuild semantics for samplers that periodically refit on
  accumulated trials.
- Quantum-classical sampler interchangeability at the API surface.

## Status

Early development (alpha): the API may still change between minor versions.
See [CHANGELOG.md](CHANGELOG.md) for what changed in each release and
[CONTRIBUTING.md](CONTRIBUTING.md) to get involved.

## License

Apache-2.0, see [LICENSE](LICENSE).

## Citation

If you use `intelloptim` in academic work, please cite the Hopaas paper above.
