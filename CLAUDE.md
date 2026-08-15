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
- the sampler union (declared in `core/study.py`, `_internal/schemas.py`,
  and `algorithms/samplers.py:ConcreteSampler`).

Changing any of these is a **coordinated, versioned event**: bump
`API_VERSION`, and update the server (which imports these) and the web app
(which mirrors the wire types by hand). Adding a new sampler *kind* touches
`SamplerKind` + `SamplerSpec` + `ConcreteSampler` — all three now live in
*this* repo, so it's a single-repo change here, then a server bump.

## Mental map

```
src/intelloptim/
  _internal/      wire schemas, hashing (federation-critical), logging, audit
  core/           pure domain types (parameters, trial, study, pruners,
                  objective, samplers/) — no I/O, only pydantic
  algorithms/     spec -> optuna resolvers (samplers.py, pruners.py) +
                  load_sampler_plugins() entry-point discovery. Needs optuna.
  client/         REST/WS SDK (Client, StudyHandle, Trial, errors) — httpx
  worker/         evaluator CLI (evaluator_cli.py, registry.py, slurm.py);
                  `intelloptim-worker evaluator` subcommand only
  params.py / pruners.py / samplers.py   re-export shims over core/
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
6. **`core/samplers/quantum/_stubs.py` is constructible but unimplemented.**
   Do not implement quantum samplers here — they ship as external plugins.

## Wire protocol invariants

- Study-scoped requests carry `study_id` in the URL and `config_digest` in
  the body; **409** on mismatch → `ConfigDigestMismatchError`.
- **503 + `Retry-After`** from `/ask` → `StudyRebuildingError`.
- **`/ask` and `/enqueue` return 410** when STOPPED → `StudyStoppedError`.
- Study identity: `study_id = sha256(token, name, canonicalized_params,
  sampler_spec, pruner_spec, direction)` — always via
  `study_identity(token, request)`. Token rotation starts a fresh study
  (raw token mixes into the hash — intentional; do not hash on `user_id`).
- `Client.create_study` double-checks the server-returned `study_id`
  against the locally-computed one.

## Sampler plugin contract

To add a **built-in** sampler: add a spec class in `core/samplers/`, add its
`kind` to `SamplerKind`, add it to `SamplerSpec` (`_internal/schemas.py`) and
`ConcreteSampler` (`algorithms/samplers.py`), and register a resolver in
`algorithms/samplers.py:_RESOLVERS`.

To add an **external** sampler (e.g. quantum): register under the
`intelloptim.samplers` entry-point group; the entry-point *name* must equal
a `SamplerKind` value and the callable resolves `(spec) ->
optuna.samplers.BaseSampler`. `load_sampler_plugins()` discovers these.
Contract + reference plugin: `docs/samplers/quantum_plugin.md` and
`examples/04_quantum_plugin/`.

Ships in-repo: `TPE` (inline), `RemoteRandom` (Celery-dispatched synthetic,
the server's test vehicle), `QuantumQAOA`/`QuantumVQE` (constructible stubs).

## SDK contract

```python
import intelloptim as intopt
with intopt.Client(url="...", token=...) as client:
    study = client.create_study(
        name="...", direction="minimize",
        sampler=intopt.samplers.TPE(seed=42),
        parameters=[intopt.params.Float("x", -10, 10)],
    )
    study.optimize(my_objective, n_trials=50, catch=Exception)
    best = study.best_trial
```
- `tell(state=COMPLETE)` requires `value`; `PRUNED`/`FAILED` accept `None`.
- `create_study` is idempotent and parallel-safe; `best_trial` is `None`
  until at least one trial completes.

## Testing

- `client/` tests exercise the HTTP/wire layer **without a server** — via
  `httpx.MockTransport` (`test_error_mapping`) or pure logic
  (`test_trial`, `test_auth`). The real client↔server round-trip lives in
  the private `intelloptim-server` integration suite. **Do not add a
  dependency on the private server here.**
- Trial params in tests: `trial.suggest_float("x", -1, 1)` or
  `trial.params["x"]`.

## Style

Global preferences apply (Black 88-col, double quotes, `typing.List/Dict/
Optional`, `os.path`, type hints). Wire schemas in `_internal/schemas.py`;
no comments unless *why* is non-obvious. Keep `mypy --strict` clean.

## Do not modify without explicit instruction

- `_internal/hashing.py` + `study_identity` (federation breaks silently)
- `_internal/schemas.py` wire models (versioned change; bump `API_VERSION`)
- `core/samplers/quantum/_stubs.py`; the Optuna study-name → `study_id`
  convention; `Client.create_study` `study_id` agreement check
- Token-based `study_id` hash (user-based silently breaks federation)

## Common commands

```bash
PYBIN=/path/to/conda/envs/intelloptim/bin
$PYBIN/pip install -e ".[client,dev]"
$PYBIN/pytest tests/unit -q
$PYBIN/ruff check src/ tests/
$PYBIN/black --check src/ tests/ examples/
$PYBIN/mypy src/intelloptim/
```

## Project skills

- `optimization-quantum.md` — `core/samplers/`, `algorithms/`, quantum
- `docs-loop.md` — docstring + docs/ review after public-surface changes
