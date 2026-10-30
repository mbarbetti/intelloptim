# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html); while the
version is below 1.0, minor releases may contain breaking changes.

## [Unreleased]

### Added

- Sphinx and Read the Docs configuration for a public guide and API reference,
  built with warnings treated as errors, a package metadata documentation URL,
  and a README build-status badge.

### Fixed

- Encode opaque study and trial identifiers as URL path segments, close the
  evaluator HTTP client, avoid collisions when importing same-named objective
  files, and reject invalid `optimize(n_trials=...)` values.

### Security

- Reject unsafe local token files, including symlinks and files accessible by
  group or other users, and redact sensitive fields from structured logging.
- Add CodeQL static analysis and PR dependency review.

## [0.1.0] - 2026-10-30

First public release.

### Added

- Client SDK in `intelloptim.client` (`Client`, `StudyHandle`, `Trial`,
  `TrialState` and the `errors` module) for an intelloptim server: ask/tell,
  enqueue, early stopping with `Trial.report` and `Trial.should_prune`
  (`TrialPruned` ends a pruned trial). `intelloptim.client` and
  `intelloptim.algorithms` are loaded on first access, so `import intelloptim`
  stays light.
- Declarative search spaces (`intelloptim.params`) with float, int and
  categorical parameters.
- Sampler and pruner specifications backed by Optuna (Random, TPE, GP, NSGA-II,
  QMC; Median, Percentile, Threshold, Nop) and the built-in samplers SA and OWG.
- Quantum policy samplers (Gaussian and mixture variants) with a simulated
  version and a hardware version, an executor plugin mechanism
  (`intelloptim.quantum_executors`) and an IBM Runtime executor.
- `intelloptim-worker evaluator` command, which runs an objective registered with
  `@intopt.objective` against a server, for example as a SLURM array job. Each
  worker sends its worker and site ids (`StudyHandle.optimize(...,
  worker_id=...)`), derived from the SLURM environment.
- Optional extras: `gp`, `qmc`, `quantum`, `quantum-ibm`, `samplers`.

[Unreleased]: https://github.com/mbarbetti/intelloptim/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mbarbetti/intelloptim/releases/tag/v0.1.0
