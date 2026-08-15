"""``intelloptim-worker`` console script.

A thin Click dispatcher with one role:

* ``evaluator`` — convenience loop that imports a user-registered objective
  and drives ``study.optimize()`` over HTTP. Designed to be submitted as
  a SLURM array job so N processes collaborate on one study.

The heavier optimizer role (a Celery worker for ``runs_remote`` samplers)
lives in the private ``intelloptim-server`` package, not here — it needs
the server's Celery app, config, and task definitions.

The evaluator defers its heavy imports until the subcommand is selected so
a minimal install can run ``intelloptim-worker --help`` without pulling in
httpx.
"""

from __future__ import annotations

from typing import Optional

import click


@click.group(name="intelloptim-worker")
def main() -> None:
    """Run an intelloptim evaluator worker."""


@main.command("evaluator")
@click.option(
    "--server",
    "server",
    required=True,
    envvar="INTELLOPTIM_URL",
    help="Base URL of the intelloptim server (or set INTELLOPTIM_URL).",
)
@click.option(
    "--import",
    "import_spec",
    required=True,
    metavar="MODULE:CALLABLE",
    help=(
        "Dotted import path to the objective callable, e.g. "
        "'my_pkg.objectives:my_fn'. The module is imported at startup "
        "which triggers any @intopt.objective registration."
    ),
)
@click.option(
    "--token",
    "token",
    default=None,
    envvar="INTELLOPTIM_TOKEN",
    help="API bearer token. Falls back to INTELLOPTIM_TOKEN / ~/.config/intelloptim/token.",
)
@click.option(
    "--spec-file",
    "spec_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a JSON file containing the StudyResolveRequest payload. "
        "Not required when the objective was decorated with study_spec={{...}}."
    ),
)
@click.option(
    "--n-trials",
    "n_trials",
    type=int,
    default=50,
    show_default=True,
    help="Number of trials this worker process should contribute.",
)
@click.option(
    "--worker-id",
    "wid",
    default=None,
    help=(
        "Override the worker identifier sent as X-Worker-Id. "
        "Auto-derived from SLURM env vars when not set."
    ),
)
@click.option(
    "--site",
    "site",
    default=None,
    help=(
        "Override the site identifier sent as X-Site (groups workers by "
        "cluster/host in the dashboard). Auto-derived from SLURM env vars "
        "or hostname when not set."
    ),
)
@click.option(
    "--loglevel",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
@click.option(
    "--no-catch",
    "no_catch",
    is_flag=True,
    default=False,
    help="Re-raise exceptions from the objective instead of logging and continuing.",
)
def evaluator(
    server: str,
    import_spec: str,
    token: Optional[str],
    spec_file: Optional[str],
    n_trials: int,
    wid: Optional[str],
    site: Optional[str],
    loglevel: str,
    no_catch: bool,
) -> None:
    """Run an evaluator loop: import an objective and drive study.optimize()."""
    from intelloptim.worker.evaluator_cli import run_evaluator

    run_evaluator(
        server=server,
        import_spec=import_spec,
        token=token,
        spec_file=spec_file,
        n_trials=n_trials,
        wid=wid,
        site=site,
        loglevel=loglevel,
        catch_exceptions=not no_catch,
    )


__all__ = ["main"]
