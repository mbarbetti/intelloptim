# Multi-site federation demo

Two independent Python processes — on the same machine, on different
machines, on different infrastructures — driving the **same** Optuna
study via one intelloptim server, with no coordination beyond a shared
API token.

## What this shows

intelloptim derives a `study_id` from
`sha256(token, name, parameters, sampler, pruner, direction)`. Any two
callers who present the same spec resolve to the same study by
construction — the first creates it, subsequent ones join it. This is
the federation primitive in action.

## Running

You'll need one intelloptim server running somewhere reachable. For a
local demo, in **terminal 0**:

```bash
intelloptim server --storage sqlite:///./dev.db --host 127.0.0.1 --port 8000
```

Then in **terminal 1** (the "alpha" site):

```bash
export INTELLOPTIM_URL=http://127.0.0.1:8000
export INTELLOPTIM_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

python examples/03_multi_site_same_study/run.py --site alpha --n-trials 25
```

In **terminal 2** (the "beta" site) — note: **same** token:

```bash
export INTELLOPTIM_URL=http://127.0.0.1:8000
export INTELLOPTIM_TOKEN=...   # paste the same value from terminal 1

python examples/03_multi_site_same_study/run.py --site beta --n-trials 25
```

Both terminals print the resolved `study_id` and either `created`
(first to start) or `joined` (every subsequent invocation). They then
take turns asking for trials and telling results to the same server-
side study.

## What to look for

- Both terminals report the **same** `sha256:...` `study_id`.
- The second invocation says **joined**, not created.
- Trial numbers from the two terminals interleave (the server hands
  out a fresh number on each `/ask`, no matter which client asks).
- The TPE sampler on the server learns from trials reported by **both**
  sites — so terminal beta benefits from terminal alpha's tells, and
  vice versa.

## Variations to try

- **Change the token in one terminal** → different `study_id`, the two
  runs no longer collaborate.
- **Change a parameter range or the sampler seed** → different
  `study_id`, by design.
- **Kill one terminal mid-run** → restart it with the same args; it
  rejoins the existing study and keeps contributing.

## Where this fits in the architecture

This is the human-runnable companion to
`tests/integration/test_subprocess_federation.py`, which exercises the
same primitive end-to-end in CI via `subprocess.run` against a real
uvicorn server.
