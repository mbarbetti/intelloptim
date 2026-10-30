"""Local and CI checks: the same sessions run on a laptop and in GitHub Actions.

Locally the sessions run in conda environments (``nox -r`` reuses them); CI
overrides the backend with ``--default-venv-backend virtualenv``.

    nox                    lint, typecheck, tests on the default Python, build
    nox -s tests           tests on every supported Python
    nox -s tests-3.14      tests on one Python
    nox -s build           build the sdist and wheel, check them, smoke-install
    nox -s docs            build the Sphinx documentation with warnings as errors
"""

from __future__ import annotations

import glob
import os

import nox

PYTHONS = ["3.11", "3.12", "3.13", "3.14", "3.15"]
DEFAULT_PYTHON = "3.12"

# New interpreters (3.14, 3.15) reach conda-forge first; the default channels
# may not have them yet. The variable is only a default: set your own to override.
os.environ.setdefault("CONDA_CHANNELS", "conda-forge")

nox.options.default_venv_backend = "conda"
nox.options.sessions = ["lint", "typecheck", f"tests-{DEFAULT_PYTHON}", "build"]
nox.options.error_on_external_run = True


@nox.session(python=DEFAULT_PYTHON)
def lint(session: nox.Session) -> None:
    """Ruff and Black, as in the pre-commit hooks."""
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "src", "tests", "noxfile.py", "scripts")
    session.run("black", "--check", "src", "tests", "noxfile.py", "scripts")


@nox.session(python=DEFAULT_PYTHON)
def typecheck(session: nox.Session) -> None:
    """Strict mypy on the package."""
    session.install("-e", ".[dev]")
    session.run("mypy", "src/intelloptim")


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    """Unit and integration tests; tests that need a real device never run."""
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "tests",
        "-q",
        "-m",
        "not hardware",
        *session.posargs,
        env={
            "INTELLOPTIM_ALLOW_QUANTUM_HARDWARE": "",
            "INTELLOPTIM_RUN_HARDWARE_TESTS": "",
        },
    )


@nox.session(python=DEFAULT_PYTHON)
def build(session: nox.Session) -> None:
    """Build, validate and smoke-test the distributions."""
    session.install("build", "twine")
    dist = os.path.join(session.create_tmp(), "dist")
    session.run("python", "-m", "build", "--outdir", dist)
    session.run("twine", "check", "--strict", *glob.glob(os.path.join(dist, "*")))
    session.run("python", "scripts/release_tools.py", "check-contents", dist)
    wheel = glob.glob(os.path.join(dist, "*.whl"))[0]
    session.install(wheel)
    session.run(
        "python",
        "-c",
        "import intelloptim; print('intelloptim', intelloptim.__version__)",
    )
    session.run("intelloptim-worker", "--help")


@nox.session(python=DEFAULT_PYTHON)
def docs(session: nox.Session) -> None:
    """Build the public API documentation with strict warning handling."""
    session.install("-e", ".[docs]")
    session.run(
        "sphinx-build",
        "-W",
        "--keep-going",
        "-b",
        "html",
        "docs/source",
        "docs/_build/html",
    )
