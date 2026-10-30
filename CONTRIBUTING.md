# Contributing to intelloptim

Thanks for your interest. Bug reports, fixes, documentation and new samplers are
all welcome.

## Workflow

`main` is protected: nothing is pushed to it directly.

1. Open an issue for anything non-trivial, so the approach can be agreed first.
2. Create a branch from `main` (`feat/short-name`, `fix/short-name`).
3. Make the change with tests, and add an entry under `[Unreleased]` in
   `CHANGELOG.md` (label the PR `skip-changelog` if users are not affected).
4. Open a pull request. The PR title becomes the commit message on `main`
   (squash merge), so write it as a
   [Conventional Commit](https://www.conventionalcommits.org/): `feat(client): ...`,
   `fix(algorithms): ...`, `docs: ...`.
5. The `ci-ok` check must pass before the PR can be merged.

## Development setup

```bash
git clone https://github.com/mbarbetti/intelloptim
cd intelloptim
pip install -e ".[dev]"
pre-commit install        # optional: formatting and lint hooks on commit
```

Use a virtual or conda environment of your choice; Python 3.11 or newer.

## Checks

```bash
pytest tests/unit -q               # fast, no optional dependencies
pytest tests/integration -q        # real Qiskit, needs the quantum extra
ruff check src tests
black --check src tests
mypy src/intelloptim
nox                                # all of the above, in isolated environments
nox -s tests-3.14                  # the tests on one Python version
nox -s docs                         # strict Sphinx documentation build
```

Tests marked `hardware` talk to a real quantum device. They are skipped unless
you opt in explicitly, and they never run in CI.

## Documentation publishing

Documentation is built for every pull request by CI. Read the Docs publishes
the strict Sphinx build from `main` and provides versioned documentation for
release tags.

## Code conventions

- Black (88 columns), Ruff and `mypy --strict` must pass.
- Type hints everywhere, using `typing.List`, `Dict`, `Optional` and `Union`;
  `os.path` instead of `pathlib`.
- Google-style docstrings with typed `Args:`, `Returns:` and `Raises:`. A
  docstring tells a user what the code does and when to use it.
- Respect the layering in `CLAUDE.md`: `core/` stays free of I/O and of Optuna,
  Qiskit is imported only under `algorithms/backend/quantum/qiskit/`, and the
  hashing in `_internal/hashing.py` is byte-stable.
- Never put tokens, credentials or private data in code, tests, examples or
  logs.

## Releases (maintainers)

Versions come from git tags. After the pull request with the changelog entry is
merged:

```bash
scripts/release.sh 0.1.0rc1     # pre-release: published to TestPyPI
scripts/release.sh 0.1.0        # release: published to PyPI after approval
```

The script runs the checks locally, asks for confirmation, then pushes the tag
`vX.Y.Z`. The release workflow builds the distributions, publishes them with
PyPI Trusted Publishing and creates the GitHub release from the changelog.

## License

By contributing you agree that your contribution is licensed under the
Apache License 2.0, like the rest of the project.
