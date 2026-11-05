<!--
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mbarbetti/intelloptim/main/docs/images/intelloptim-logo-w-bkg.png">
    <img alt="intelloptim" src="https://raw.githubusercontent.com/mbarbetti/intelloptim/main/docs/images/intelloptim-logo-black.png" width="600">
  </picture>
</div>
-->

# intelloptim

[![Python](https://img.shields.io/pypi/pyversions/intelloptim.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/intelloptim.svg)](https://pypi.org/project/intelloptim/)
[![License](https://img.shields.io/pypi/l/intelloptim.svg)](https://github.com/mbarbetti/intelloptim/blob/main/LICENSE)
[![CI](https://github.com/mbarbetti/intelloptim/actions/workflows/ci.yml/badge.svg)](https://github.com/mbarbetti/intelloptim/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mbarbetti/intelloptim/graph/badge.svg)](https://codecov.io/gh/mbarbetti/intelloptim)
[![Documentation](https://img.shields.io/readthedocs/intelloptim?logo=read-the-docs)](https://intelloptim.readthedocs.io/en/stable)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23880962.svg)](https://doi.org/10.5281/zenodo.23880962)

Server-orchestrated, data-driven optimization with classical and quantum
samplers. Evaluators on any infrastructure (HPC, cloud, on-prem) collaborate on
a single study through a coordinating server.

This package is the client SDK and the sampler layer. It talks to an
intelloptim server over HTTP; the server itself is not part of this package.
Full documentation: **[intelloptim.readthedocs.io](https://intelloptim.readthedocs.io/en/stable)**.

## 🌱 Where it comes from

`intelloptim` is the evolution of **Hopaas**, *Hyperparameter OPtimization As A
Service* (Barbetti & Anderlini, [arXiv:2301.05522](https://arxiv.org/abs/2301.05522)).
Hopaas was built to coordinate many trainings of a machine-learning model on
opportunistic resources from different providers, which share nothing but a
network connection to a server that holds the study. It tuned the generative
models behind the LHCb ultra-fast simulation, with studies fed by more than
twenty concurrent nodes.

`intelloptim` keeps that philosophy and generalizes it to small-to-moderate-
dimensional optimization problems, such as system calibration and model tuning,
with more robustness and security:

- bearer-token authentication (the token never appears in a URL or in a study's
  identity) and a client that refuses unsafe token files;
- studies identified by their content and scoped to a user or group, so
  independent processes join the same study, and a mismatching configuration is
  refused instead of silently mixed;
- more samplers (Optuna, built-in and quantum policy samplers) behind one API,
  with background rebuilds for the ones that refit;
- `enqueue`, `stop`, call quotas and a worker for SLURM.

## 📦 Installation

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

Python 3.11 or newer is required. You also need the URL of a server and an API
token.

## 🧭 Server-orchestrated optimization

``intelloptim`` keeps Optuna's study, trial, sampler and pruner concepts, while
the server holds the shared study so that many machines can contribute to it.

```python
import intelloptim as intopt

with intopt.client.Client(url="https://intelloptim.example.org") as client:
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

**Coming from Optuna?** The search space is declared once, in `create_study`,
and not inside the objective with `suggest_*`. The objective reads the sampled
values by name (`sampled_float`, `sampled_int`, `sampled_categorical`).
[More](https://intelloptim.readthedocs.io/en/latest/guides/optuna-and-intelloptim.html)

- **ask / tell:** `optimize` runs the loop; `study.ask()` and `study.tell()` give
  you control of it. `create_study` is idempotent, so any process that declares
  the same study joins it.
  [More](https://intelloptim.readthedocs.io/en/latest/guides/ask-tell-enqueue.html)
- **enqueue:** `study.enqueue({...})` queues a point to evaluate next, and
  `study.enqueue({...}, value=...)` records one that was evaluated elsewhere.
- **prune:** report intermediate values and let the study's pruner stop a
  hopeless trial; it does not count against your call quota.
  [More](https://intelloptim.readthedocs.io/en/latest/guides/pruning.html)

  ```python
  def objective(trial: intopt.client.Trial) -> float:
      for epoch in range(20):
          loss = train_one_epoch(trial.sampled_float("lr"))
          trial.report(loss, epoch)
          if trial.should_prune():
              raise intopt.client.errors.TrialPruned
      return loss
  ```

- **samplers:** three families behind `sampler_spec=`: **Optuna** (Random, QMC,
  TPE, GP, NSGA-II), **built-in** (simulated annealing `SA`, objective-weighted
  Gaussian `OWG`) and **quantum** policy samplers, simulated or on hardware.
  [More](https://intelloptim.readthedocs.io/en/latest/guides/sampling.html)

## 🔐 Authentication

Every request carries an API token. The client takes it from the `token=`
argument, then from the `INTELLOPTIM_TOKEN` environment variable, then from
`~/.config/intelloptim/token`, which it only reads if it is a regular file
owned by you with no group or other permissions. Treat the token as a password:
never put it in code, command lines, study specifications or logs, and use
HTTPS for any non-local server.
[More](https://intelloptim.readthedocs.io/en/latest/guides/authentication.html)

## 🖥️ Running on a cluster

`intelloptim-worker` runs an objective on a batch system without a driver
script. Each task of a SLURM array job imports the objective, joins the study
and runs its trials; worker and site ids come from the SLURM environment.

```bash
intelloptim-worker evaluator \
    --server https://intelloptim.example.org \
    --import objective.py:my_objective \
    --n-trials 50
```

[More](https://intelloptim.readthedocs.io/en/latest/guides/cluster.html), with a
complete array-job script.

## 🚧 Status

Early development (alpha): the API may still change between minor versions.
See [CHANGELOG.md](https://github.com/mbarbetti/intelloptim/blob/main/CHANGELOG.md)
for what changed in each release and
[CONTRIBUTING.md](https://github.com/mbarbetti/intelloptim/blob/main/CONTRIBUTING.md)
to get involved.

## 📄 License

Apache-2.0, see [LICENSE](https://github.com/mbarbetti/intelloptim/blob/main/LICENSE).

## 📚 Citation

If you use `intelloptim` in academic work, please cite the Hopaas paper:
M. Barbetti and L. Anderlini, *Hyperparameter Optimization as a Service on INFN
Cloud*, [arXiv:2301.05522](https://arxiv.org/abs/2301.05522).
