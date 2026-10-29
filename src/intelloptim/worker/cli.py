"""The ``intelloptim-worker`` console script.

It has one subcommand, ``evaluator``, which imports an objective and runs
``study.optimize()`` against a server. Started as the tasks of a SLURM array
job, many such processes collaborate on one study. See
:mod:`intelloptim.worker.evaluator_cli` for what a worker does.

The imports the evaluator needs are deferred until the subcommand runs, so
``intelloptim-worker --help`` stays fast.
"""

from __future__ import annotations

from typing import Optional

import click


@click.group(name="intelloptim-worker")
def main() -> None:
    """Run intelloptim workers."""


@main.command("evaluator")
@click.option(
    "--server",
    "server",
    required=True,
    envvar="INTELLOPTIM_URL",
    help="Base URL of the intelloptim server. Falls back to INTELLOPTIM_URL.",
)
@click.option(
    "--import",
    "import_spec",
    required=True,
    metavar="MODULE:CALLABLE",
    help=(
        "Objective to run: 'my_pkg.objectives:my_fn' or "
        "'/path/to/objective.py:my_fn'. The module is imported at startup, "
        "which registers its @intopt.objective study spec."
    ),
)
@click.option(
    "--token",
    "token",
    default=None,
    envvar="INTELLOPTIM_TOKEN",
    help=(
        "API bearer token. Falls back to INTELLOPTIM_TOKEN and then to "
        "~/.config/intelloptim/token."
    ),
)
@click.option(
    "--spec-file",
    "spec_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "JSON file with the study spec (a StudyResolveRequest payload). "
        "Not needed if the objective was decorated with study_spec={...}."
    ),
)
@click.option(
    "--n-trials",
    "n_trials",
    type=int,
    default=50,
    show_default=True,
    help="Number of trials this worker runs.",
)
@click.option(
    "--worker-id",
    "wid",
    default=None,
    help=(
        "Worker id sent as X-Worker-Id. Derived from the SLURM environment, "
        "or from the host name and process id, when not set."
    ),
)
@click.option(
    "--site",
    "site",
    default=None,
    help=(
        "Site id sent as X-Site, which groups workers by cluster or host. "
        "Derived from the SLURM environment, or from the host name, when "
        "not set."
    ),
)
@click.option(
    "--loglevel",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging level.",
)
@click.option(
    "--no-catch",
    "no_catch",
    is_flag=True,
    default=False,
    help=(
        "Stop the run when the objective raises, instead of marking that "
        "trial as failed and continuing."
    ),
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
    """Run an evaluator worker: import an objective and optimise it."""
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
