# intelloptim — project guidance

## What this repo is

The client SDK and the algorithm layer of **intelloptim**, a general black-box
optimization framework built on Optuna's ask/tell API (an evolution of
**Hopaas**, Barbetti & Anderlini, 2023,
[arXiv:2301.05522](https://arxiv.org/abs/2301.05522)). It contains the user-facing
client, the pure domain types, the wire protocol, and the sampler/pruner
resolution including the in-repo sampler runtimes. Imported as
`import intelloptim as intopt`.

Users install it to talk to a running intelloptim server and to write
objectives. The server and the dashboard are separate projects that depend on
this package for its domain types, wire schemas, hashing and algorithms.

## Mental map

```
src/intelloptim/
  _internal/      wire schemas (incl. TrialState), hashing (study identity),
                  logging, audit helpers
  core/           pure domain types — no I/O, only pydantic
    samplers/       base.py (SamplerKind, rebuild policy, base specs),
                    optuna.py (Random/TPE/GP/NSGA-II/QMC specs),
                    builtin.py (SA/OWG specs), quantum.py (quantum policy
                    specs: Gaus/Mixture, simulated "Sim" and hardware)
    pruners/        base.py (PrunerKind), optuna.py (Median/Percentile/
                    Threshold/Nop specs)
    parameters.py, objective.py
  algorithms/     spec -> optuna resolvers (samplers.py, pruners.py) and
                  load_sampler_plugins() entry-point discovery
    backend/        runtime for non-Optuna samplers (builtin/: sa.py, owg.py;
                    quantum/: gaus.py, mixture.py, commons/ for the
                    vendor-neutral policy, learner and executors, qiskit/ for
                    the only Qiskit code)
    protocols.py, errors.py, space.py
  client/         REST/WS SDK (Client, StudyHandle, Trial, errors) — httpx
  worker/         `intelloptim-worker` CLI (evaluator subcommand, SLURM helpers)
  params.py       public facade over core/parameters (`intopt.params.Float`)
  cli.py          minimal top-level `intelloptim` CLI
```

## Hard layering rules

1. **`core/` is pure**: stdlib and pydantic only; no `httpx`, `optuna`, I/O or
   networking.
2. **`algorithms/` is the only place Optuna's sampler/pruner classes are
   imported.** `core/` holds specs; `algorithms/` resolves them.
3. **`_internal/schemas.py` is the only place wire models are defined.**
4. **`_internal/hashing.py` output is byte-stable.** Golden tests pin it;
   changing the hashed payload changes every study id.
5. **`client/` is the only module that talks to a server over HTTP.**
6. **Quantum samplers** (`SimQuantumGaus`, `QuantumGaus`, `SimQuantumMixture`,
   `QuantumMixture`) live in-repo: specs in `core/samplers/quantum.py`, runtime
   in `algorithms/backend/quantum/`. The `Sim` samplers are ideal; the others
   run on hardware and are separate kinds on purpose. Specs never carry
   credentials, and only `algorithms/backend/quantum/qiskit/` imports Qiskit.

## Wire protocol and stability

- The wire models and `API_VERSION` live in `_internal/schemas.py`; the sampler
  and pruner unions are `SamplerSpec` and `PrunerSpec`. Changing any of them is
  a versioned event that the server and the dashboard must follow.
- Study-scoped requests carry `study_id` in the URL and `config_digest` in the
  body; **409** on mismatch (`ConfigDigestMismatchError`).
- **503 + `Retry-After`** from `/ask` while a sampler rebuilds
  (`StudyRebuildingError`). **410** from `/ask` and `/enqueue` when the study is
  stopped (`StudyStoppedError`). **429** from `/ask` or `/tell` when the quota is
  exhausted.
- `POST /studies/{id}/trials/{tid}/should_prune` takes the intermediate values
  not yet sent and returns `{"should_prune": bool}`; it is not quota-counted.
- Study identity: `study_id = sha256(study_key, study_name, direction,
  canonicalized_params, sampler_spec, pruner_spec)`, always computed with
  `study_identity(study_key, request)`; `study_key` is an opaque per-user or
  per-group key resolved by the server, never the API token.
- `trial_id` is an opaque handle; `number` is the position in the study.

## Samplers

To add a **built-in** sampler: add a spec class in `core/samplers/`, add its
`kind` to `SamplerKind`, add it to the `SamplerSpec` union, and register a
resolver in `algorithms/samplers.py:_RESOLVERS`. If it is not a thin wrapper
over an Optuna sampler, put its runtime in `algorithms/backend/`. Pruners follow
the same pattern.

Built-in samplers must be **stateless**: the server builds a fresh sampler on
every ask, so anything an in-memory sampler would remember must be rebuilt from
the study history (with RNGs derived from the seed and trial number) or from a
model persisted in the study by `rebuild(study)` (see `OWG`, `SA` and the
quantum samplers).

External packages can override the resolver of an existing kind through the
`intelloptim.samplers` entry-point group, and add a quantum provider through
`intelloptim.quantum_executors`.

Ships in-repo: `Random`, `TPE`, `GP`, `NSGA-II`, `QMC` (Optuna-backed; `GP`
needs the `gp` extra, `QMC` the `qmc` extra), `SA`, `OWG`, and the quantum
policy samplers (extras `quantum`, `quantum-ibm`). A missing optional package
raises `SamplerDependencyError` naming the extra to install.

## SDK contract

```python
import intelloptim as intopt
with intopt.Client(url="...", token=...) as client:
    study = client.create_study(
        study_name="...", direction="minimize",
        sampler_spec=intopt.samplers.TPESamplerSpec(seed=42),
        parameters=[intopt.params.Float("x", -10, 10)],
    )
    study.optimize(my_objective, n_trials=50, catch=Exception)
    best = study.best_trial
```

- The search space is declared once, in `create_study(parameters=[...])`.
  Objectives read the sampled values by name: `trial.sampled_float("x")`,
  `sampled_int`, `sampled_categorical`. There is no `suggest_*`.
- `tell(state=COMPLETE)` requires `value`; `PRUNED`/`FAILED` accept `None`.
- Early stopping: `trial.report(value, step)` buffers locally,
  `trial.should_prune()` sends the new values, and the objective raises
  `intopt.TrialPruned`, which `optimize` records as `PRUNED`.
- `create_study` is idempotent and parallel-safe; `best_trial` is `None` until
  one trial completes. `optimize(n_trials=N)` raises `InsufficientQuotaError`
  before any call if `N` exceeds the remaining calls.

Client errors: `ConfigDigestMismatchError` (409), `StudyRebuildingError` (503),
`StudyStoppedError` (410), `MissingTokenError`, `InsufficientQuotaError`,
`StudyLimitExceededError`.

## Testing

- `tests/unit/` tests pure logic and never imports third-party quantum SDKs; the
  quantum samplers run against a numpy executor in `tests/support/`.
  `tests/integration/` (marker `qiskit`) uses real Qiskit and a stubbed IBM
  runtime. Tests marked `hardware` hit a real device, are opt-in, and must never
  run in automation.
- `client/` tests use `httpx.MockTransport`; there is no dependency on a server.
- Every feature or fix ships with `pytest` tests (happy path and edge cases).

## Style

Black (88 columns, double quotes), Ruff, `mypy --strict`. Type hints with
`typing.List/Dict/Optional/Union`, `os.path` instead of `pathlib`, Google-style
docstrings with typed `Args:`, `Returns:`, `Raises:`. Comments only for a
non-obvious *why*. Docstrings describe behaviour for users, not development
rules.

## Do not modify without explicit instruction

- `_internal/hashing.py` and `study_identity` (study ids and digests depend on
  the exact bytes hashed).
- The wire models in `_internal/schemas.py` (versioned change, bump
  `API_VERSION`).

## Commands

```bash
pip install -e ".[dev]"
pytest tests/unit -q
pytest tests/integration -q     # needs the quantum extra
ruff check src/ tests/
black --check src/ tests/
mypy src/intelloptim/
nox                             # the same checks, across Python versions
```

## Development notes

The maintainer keeps additional design notes in `.claude/notes/`. They are not
part of the repository; ask for them when working in these areas:

- samplers and benchmarking (statelessness rules, sampler choice, honest
  benchmarking)
- quantum samplers (design, cohort/snapshot mechanics, hardware guards, limits)
- sampler plugins (entry-point contract, adding a built-in sampler kind)
- documentation conventions (docstring style, checklist after public-surface
  changes)
