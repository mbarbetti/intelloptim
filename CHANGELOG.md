# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html); while the
version is below 1.0, minor releases may contain breaking changes.

## [Unreleased]

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
- `intelloptim` and `intelloptim-worker` command-line tools.
- Optional extras: `gp`, `qmc`, `quantum`, `quantum-ibm`, `samplers`.

[Unreleased]: https://github.com/mbarbetti/intelloptim/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mbarbetti/intelloptim/releases/tag/v0.1.0
