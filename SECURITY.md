# Security policy

## Supported versions

`intelloptim` is in alpha. Security fixes are made on the latest release only.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem. Report it
privately with GitHub's
[private vulnerability reporting](https://github.com/mbarbetti/intelloptim/security/advisories/new)
on this repository. Include the affected version, a description of the problem
and, if possible, steps to reproduce it. You can expect an acknowledgement
within a few days.

## Scope notes

- API tokens are credentials: never put one in code, a spec, a log or an issue.
  Study specs and quantum sampler specs deliberately have no credential fields.
- Quantum hardware access is opt-in and its credentials are managed by the
  server; see the documentation of the quantum samplers.
