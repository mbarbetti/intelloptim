# intelloptim (public SDK + algorithm layer) — guidance for Claude

## What this repo is

The **open-source** half of intelloptim: the user-facing client SDK, the
pure domain kernel, the wire protocol, and the optimization **algorithm
layer** (sampler/pruner resolution + the sampler-plugin contract). A general
black-box optimization framework built on Optuna's ask/tell API. Evolution
of **Hopaas** (Barbetti & Anderlini, 2023, [arXiv:2301.05522](https://arxiv.org/abs/2301.05522)).
Imported as `import intelloptim as intopt`.

Users `pip install intelloptim` to talk to a running service and to write
objectives. The service itself (FastAPI orchestrator, auth, dashboard,
deployment) is the private `intelloptim-server` / `intelloptim-web` repos.

## Related repositories

- **`intelloptim-server`** (private) — the FastAPI service, auth, storage,
  Celery, the optimizer worker, and the `deploy/` stack. It **depends on
  this package** and imports algorithm resolution from
  `intelloptim.algorithms`. It never vendors or forks this kernel.
- **`intelloptim-web`** (private) — the React dashboard. Its TypeScript wire
  types are hand-written mirrors of this package's `_internal/schemas.py`.

### Cross-repo consistency contract (identical in all three repos)

The **shared source of truth lives here, in the public repo**:
- `_internal/schemas.py` — the pydantic wire models + `API_VERSION`.
- `_internal/hashing.py` — `study_identity` / content-addressed `study_id`
  + `config_digest`. **Byte-stable** across processes or federation breaks
  silently.
- the sampler union `SamplerSpec` (`core/samplers/__init__.py`, consumed by
  `_internal/schemas.py`) and the pruner union `PrunerSpec`
  (`core/pruners/__init__.py`).

Changing any of these is a **coordinated, versioned event**: bump
`API_VERSION`, and update the server (which imports these) and the web app
(which mirrors the wire types by hand). Adding a new sampler *kind* touches
`SamplerKind` + `SamplerSpec` + `_RESOLVERS` — all live in *this* repo, so
it's a single-repo change here, then a server bump.

## Mental map

```
src/intelloptim/
  _internal/      wire schemas, hashing (federation-critical), logging, audit
  core/           pure domain types — no I/O, only pydantic
    samplers/       base.py (SamplerKind, rebuild policy, base specs),
                    optuna.py (Random/TPE/GP/NSGA-II/QMC specs),
                    builtin.py (SA/OWG specs), quantum.py (QAOA/VQE stubs)
    pruners/        base.py (PrunerKind), optuna.py (Median/Percentile/
                    Threshold/Nop specs)
    parameters.py, trial.py (TrialState), objective.py
  algorithms/     spec -> optuna resolvers (samplers.py, pruners.py) +
                  load_sampler_plugins() entry-point discovery. Needs optuna.
    backend/        runtime for non-Optuna samplers (builtin/: one module
                    per sampler, sa.py and owg.py; quantum.py)
    protocols.py    SamplerPlugin protocol; errors.py: resolver errors
  client/         REST/WS SDK (Client, StudyHandle, Trial, errors) — httpx
  worker/         `intelloptim-worker` CLI (cli.py) with the `evaluator`
                  subcommand only (evaluator_cli.py, registry.py, slurm.py)
  params.py       public facade over core/parameters: the `...Spec` classes
                  without the suffix (`intopt.params.Float(...)`). Samplers
                  and pruners are reached through core only: `intopt.samplers`
                  and `intopt.pruners` are the core packages (`...Spec` names)
  cli.py          minimal top-level `intelloptim` CLI
```

## Hard layering rules

1. **`core/` must be pure.** No `httpx`, `fastapi`, `celery`, `optuna`,
   no I/O or networking. Only stdlib + pydantic.
2. **`algorithms/` is the only place Optuna's sampler/pruner *classes* are
   imported.** `core/` holds specs; `algorithms/` resolves them.
3. **`_internal/schemas.py` is the only place pydantic wire models cross
   process boundaries.**
4. **`_internal/hashing.py` must produce byte-stable output.**
5. **`client/` is the only module that talks to a server over HTTP.**
6. **`core/samplers/quantum.py` is constructible but unimplemented.**
   Do not implement quantum samplers here — they ship as external plugins.

## Wire protocol invariants

- Study-scoped requests carry `study_id` in the URL and `config_digest` in
  the body; **409** on mismatch → `ConfigDigestMismatchError`.
- **503 + `Retry-After`** from `/ask` → `StudyRebuildingError`.
- **`/ask` and `/enqueue` return 410** when STOPPED → `StudyStoppedError`.
- **429** from `/ask` or `/tell` → quota exhausted for this period.
- Study identity: `study_id = sha256(study_key, name, canonicalized_params,
  sampler_spec, pruner_spec, direction)` — always via
  `study_identity(study_key, request)`. `study_key` is an opaque UUID4 per
  user (or group) resolved server-side; multiple tokens owned by the same
  user always produce the same `study_id` for the same spec.
- `Client.create_study` accepts the server-returned `study_id` directly —
  no local recomputation or agreement check (server is authoritative).
- `GET /auth/me` is called lazily on first `create_study`; the response is
  cached in `Client._who_am_i` (`WhoAmIResponse`) for quota pre-flight
  checks. A `WARNING` is logged when the budget is limited.

## Sampler plugin contract

To add a **built-in** sampler: add a spec class in `core/samplers/`, add its
`kind` to `SamplerKind` (`core/samplers/base.py`), add it to the `SamplerSpec`
union (`core/samplers/__init__.py`), and register a resolver in
`algorithms/samplers.py:_RESOLVERS`. If it is not a thin wrapper over an
Optuna sampler, put its runtime in `algorithms/backend/` (see `SASampler`).
Pruners follow the same pattern (`PrunerKind` / `PrunerSpec` /
`algorithms/pruners.py:_RESOLVERS`).

Built-in samplers must be **stateless**: the orchestrator builds a fresh
sampler on every ask, so anything an in-memory sampler would remember (e.g.
the SA temperature and current point) must be rebuilt from the study's trial
history, with RNGs derived from the seed and trial number.
`algorithms/backend/builtin/sa.py:SASampler` is the reference implementation.

To add an **external** sampler (e.g. quantum): register under the
`intelloptim.samplers` entry-point group; the entry-point *name* must equal
a `SamplerKind` value and the callable resolves `(spec) ->
optuna.samplers.BaseSampler`. `load_sampler_plugins()` discovers these.
Contract + reference plugin: `docs/samplers/quantum_plugin.md` and
`examples/04_quantum_plugin/`.

Ships in-repo: `Random`, `TPE`, `GP`, `NSGA-II`, `QMC` (Optuna-backed),
`SA` (simulated annealing) and `OWG` (objective-weighted Gaussian), both in
`algorithms/backend/builtin/`, and `QuantumQAOA`/`QuantumVQE`
(constructible stubs).

`OWG` is the test vehicle for the rebuild plane
(`BaseRebuildSamplerSpec`: `runs_remote=True` plus a `RebuildPolicy`). Its
model is fitted by `sampler.rebuild(study)` — called by the server's
`rebuild_sampler` task — and persisted in the study's user attrs
(`intopt:owg`); asks only read it. Rebuildable
samplers should follow this pattern, since asks never share an instance.

## SDK contract

```python
import intelloptim as intopt
with intopt.Client(url="...", token=...) as client:
    study = client.create_study(
        name="...", direction="minimize",
        sampler=intopt.samplers.TPESamplerSpec(seed=42),
        parameters=[intopt.params.Float("x", -10, 10)],
    )
    study.optimize(my_objective, n_trials=50, catch=Exception)
    best = study.best_trial
```
- `tell(state=COMPLETE)` requires `value`; `PRUNED`/`FAILED` accept `None`.
- `create_study` is idempotent and parallel-safe; `best_trial` is `None`
  until at least one trial completes.
- `optimize(n_trials=N)` raises `InsufficientQuotaError` before any server
  call if `N > calls_remaining` (pre-flight from cached `/auth/me`).
- `create_study` raises `StudyLimitExceededError` before posting if the user
  is at the active-study cap (pre-flight from cached `/auth/me`).

## Error taxonomy (client-side)

| Error | When |
|---|---|
| `ConfigDigestMismatchError` | 409 — study config drifted between clients |
| `StudyRebuildingError` | 503 — sampler rebuild in progress |
| `StudyStoppedError` | 410 — study is stopped |
| `InsufficientQuotaError` | Pre-flight: n_trials > calls_remaining |
| `StudyLimitExceededError` | Pre-flight: at active-study cap |

## Testing

- `client/` tests exercise the HTTP/wire layer **without a server** — via
  `httpx.MockTransport` (`test_error_mapping`) or pure logic
  (`test_trial`, `test_auth`). The real client↔server round-trip lives in
  the private `intelloptim-server` integration suite. **Do not add a
  dependency on the private server here.**
- Trial params in tests: `trial.suggest_float("x", -1, 1)` or
  `trial.params["x"]`.
- Every feature or bug fix ships with `pytest` tests (happy path + edge
  cases) under `tests/unit/`, written before the task is declared done.

## Style

Global preferences apply (Black 88-col, double quotes, `typing.List/Dict/
Optional`, `os.path`, type hints, **Google-style docstrings** with `Args:` /
`Returns:` / `Raises:`). Wire schemas in `_internal/schemas.py`;
no comments unless *why* is non-obvious. Keep `mypy --strict` clean.

## Do not modify without explicit instruction

- `_internal/hashing.py` + `study_identity` (federation breaks silently)
- `_internal/schemas.py` wire models (versioned change; bump `API_VERSION`)
- `core/samplers/quantum.py`; the Optuna study-name → `study_id` convention
- `study_identity` signature — takes `study_key: str` (opaque UUID4), NOT a
  raw bearer token. The `study_key` is resolved server-side from the token.
  Swapping back to raw-token hashing silently breaks federation.

## Common commands

`PYBIN` (path to the project's conda env `bin/`) is machine-specific and lives
in the gitignored `.claude/local.md`.

```bash
$PYBIN/pip install -e ".[client,dev]"
$PYBIN/pytest tests/unit -q
$PYBIN/ruff check src/ tests/
$PYBIN/black --check src/ tests/ examples/
$PYBIN/mypy src/intelloptim/
```

## Project skills

- `optimization-quantum.md` — `core/samplers/`, `algorithms/`, quantum
- `docs-loop.md` — docstring + docs/ review after public-surface changes
