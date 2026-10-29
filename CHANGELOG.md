# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html); while the
version is below 1.0, minor releases may contain breaking changes.

## [Unreleased]

### Changed

- **Breaking:** the client objects are reached through `intopt.client`:
  `intopt.client.Client`, `intopt.client.Trial`, `intopt.client.TrialState` and
  `intopt.client.errors.TrialPruned` replace `intopt.Client`, `intopt.Trial`,
  `intopt.TrialState` and `intopt.TrialPruned`. `intopt.client` and
  `intopt.algorithms` are loaded on first access, so `import intelloptim` no
  longer imports Optuna.
- `intelloptim.worker` no longer re-exports `main`; the console script is
  `intelloptim.worker.cli:main`.
- `intelloptim.worker.slurm` now only provides `worker_id()` and `site_id()`; the
  sbatch rendering and submission helpers and the sbatch template moved to the
  `02_distributed_slurm` example.
- The evaluator finds the study spec attached with `@intopt.objective` from the
  imported function, whatever name it was registered under
  (`get_study_spec_for`).

### Removed

- The top-level `intelloptim` command, which only printed the version; use
  `intelloptim-worker` or `python -c "import intelloptim; print(intelloptim.__version__)"`.

### Added

- `StudyHandle.optimize(..., worker_id=...)` sends the worker id with every ask;
  the evaluator uses it.

## [0.1.0] - Unreleased

First public release.

### Added

- Client SDK (`Client`, studies, trials) for an intelloptim server: ask/tell,
  enqueue, early stopping with `Trial.report` and `Trial.should_prune`.
- Declarative search spaces (`intelloptim.params`) with float, int and
  categorical parameters.
- Sampler and pruner specifications backed by Optuna (Random, TPE, GP, NSGA-II,
  QMC; Median, Percentile, Threshold, Nop) and the built-in samplers SA and OWG.
- Quantum policy samplers (Gaussian and mixture variants) with a simulated
  version and a hardware version, an executor plugin mechanism
  (`intelloptim.quantum_executors`) and an IBM Runtime executor.
- `intelloptim-worker` command-line tool.
- Optional extras: `gp`, `qmc`, `quantum`, `quantum-ibm`, `samplers`.

[Unreleased]: https://github.com/mbarbetti/intelloptim/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mbarbetti/intelloptim/releases/tag/v0.1.0
