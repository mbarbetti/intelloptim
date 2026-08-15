# Quantum sampler plugin contract

intelloptim ships two quantum sampler *spec classes* (`QuantumQAOA`,
`QuantumVQE`) that are fully constructible and participate in the canonical
study hash, but have no built-in runtime implementation. The runtime is
supplied by an external Python package registered via the
`intelloptim.samplers` entry-point group.

This document describes the full contract an implementation must satisfy
and shows a worked example (the reference dummy plugin in
`examples/04_quantum_plugin/`).

---

## How it works

At startup `intelloptim.server.samplers.load_sampler_plugins()` calls
`importlib.metadata.entry_points(group="intelloptim.samplers")` and loads
each entry point. The entry-point **name** must equal a
`SamplerKind` string value (e.g. `"quantum_qaoa"`); the loaded object
replaces the default `SamplerNotImplementedError` stub in the resolver
registry.

```
User code                  Server
---------                  ------
QuantumQAOA(reps=3)  ──▶  load_sampler_plugins()
  (spec object)               entry_points("intelloptim.samplers")
                               → quantum_qaoa = my_pkg.sampler:resolve_qaoa
                              _RESOLVERS[SamplerKind.QUANTUM_QAOA] = resolve_qaoa

POST /studies/resolve ──▶  resolve_sampler(spec)
                               → resolve_qaoa(spec)
                               → optuna.samplers.BaseSampler instance
```

---

## Plugin contract

### 1. Resolver function

Each entry point must be a callable with the signature:

```python
def resolve_<kind>(spec: <SpecClass>) -> optuna.samplers.BaseSampler:
    ...
```

| Argument | Type | Description |
|----------|------|-------------|
| `spec`   | `QuantumQAOA` / `QuantumVQE` | The spec object as constructed by the user. Contains backend-specific fields (`reps`, `optimizer`, `shots`, `seed`, …). |

Return an `optuna.samplers.BaseSampler` instance. The server plugs it
directly into Optuna's study via `optuna.create_study(sampler=...)`.

### 2. Entry-point declaration (`pyproject.toml`)

```toml
[project.entry-points."intelloptim.samplers"]
quantum_qaoa = "my_quantum_pkg.sampler:resolve_qaoa"
quantum_vqe  = "my_quantum_pkg.sampler:resolve_vqe"
```

The key (left of `=`) is the `SamplerKind` value. Valid values:

| Key | Spec class |
|-----|-----------|
| `quantum_qaoa` | `intelloptim.core.samplers.quantum._stubs.QuantumQAOA` |
| `quantum_vqe`  | `intelloptim.core.samplers.quantum._stubs.QuantumVQE`  |

An unrecognised key is silently skipped; a load failure (e.g. missing
dependency) logs a warning and leaves the stub in place.

### 3. Spec fields (read-only in the resolver)

**`QuantumQAOA`**

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `reps` | `int` | `1` | QAOA depth `p` |
| `optimizer` | `str` | `"COBYLA"` | Classical variational optimizer |
| `shots` | `int` | `1024` | Measurement shots per circuit |
| `seed` | `Optional[int]` | `None` | Random seed |

**`QuantumVQE`**

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `ansatz` | `str` | `"RealAmplitudes"` | Variational circuit ansatz |
| `optimizer` | `str` | `"COBYLA"` | Classical optimizer |
| `shots` | `int` | `1024` | Measurement shots per circuit |
| `seed` | `Optional[int]` | `None` | Random seed |

---

## Worked example

The reference dummy plugin lives in `examples/04_quantum_plugin/`. Install it
and start the server to see the wiring in action:

```bash
pip install -e examples/04_quantum_plugin/
intelloptim server --storage sqlite:///./dev.db
# server log: "sampler plugin registered  kind=quantum_qaoa"
# server log: "sampler plugin registered  kind=quantum_vqe"

python examples/04_quantum_plugin/run_demo.py
```

The plugin resolves both specs to `optuna.samplers.RandomSampler` — not a
real quantum implementation, but sufficient to prove the entry-point chain
from user spec to Optuna sampler to `/ask` response is intact.

---

## Adding a brand-new sampler kind

The quantum stubs ship pre-declared in `intelloptim._internal.schemas.SamplerSpec`
so the server can deserialise study specs that reference them. A sampler kind
that is **not** in that union will cause a validation error on `POST /studies/resolve`.

To add a new kind you must coordinate with the intelloptim server:

1. Add a new value to `SamplerKind` in `core/samplers/base.py`.
2. Add the new spec class to `SamplerSpec` in `_internal/schemas.py`
   (discriminated union — bump `API_VERSION`).
3. Ship your resolver via the entry-point group as described above.

Because steps 1–2 require a server update, adding a new kind is a versioned
change. Implementing an *existing* quantum kind (QAOA, VQE) only requires
step 3 and works with any intelloptim version that already declares the kind.

---

## Testing your plugin

Use the dummy plugin as a template: implement `resolve_qaoa`/`resolve_vqe`,
install the package with `pip install -e .`, then run the integration test
suite against a local server:

```bash
intelloptim server --storage sqlite:///./dev.db &
pytest tests/integration/ -k quantum
```

For isolated unit testing, mock `entry_points` and call
`load_sampler_plugins()` directly — see
`tests/unit/server/test_plugin_loading.py` for the pattern.
