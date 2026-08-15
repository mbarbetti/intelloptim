# Example 04 — Quantum sampler plugin (entry-point wiring demo)

Demonstrates how to package an external quantum sampler so that the
intelloptim server discovers and loads it automatically at startup via
Python's entry-point mechanism.

The plugin in this directory is **not** a real quantum implementation.
Both `QuantumQAOA` and `QuantumVQE` resolve to `optuna.samplers.RandomSampler`
so you can validate the full wiring path without Qiskit, PennyLane, or
quantum hardware.

## What this shows

- The `intelloptim.samplers` entry-point group contract.
- How `load_sampler_plugins()` maps entry-point names to `SamplerKind` values.
- How the server replaces the built-in `SamplerNotImplementedError` stubs
  with a live resolver at startup.

## Install and run

```bash
# From the repo root — install the plugin into your intelloptim conda env
pip install -e examples/04_quantum_plugin/

# Start the server (SQLite for local use)
intelloptim server --storage sqlite:///./dev.db

# The server log will show:
#   sampler plugin registered  kind=quantum_qaoa
#   sampler plugin registered  kind=quantum_vqe

# Run a study with the QuantumQAOA sampler
python examples/04_quantum_plugin/run_demo.py
```

## Adapting to a real quantum backend

1. Replace the `RandomSampler` bodies in `sampler.py` with your actual
   implementation (Qiskit Runtime Sampler, PennyLane QNode, etc.).
2. Add your quantum SDK to `[project.dependencies]` in `pyproject.toml`.
3. Keep the entry-point names (`quantum_qaoa`, `quantum_vqe`) unchanged
   — they must match the `SamplerKind` string values in
   `intelloptim.core.samplers.base.SamplerKind`.

See `docs/samplers/quantum_plugin.md` for the full contract specification.
