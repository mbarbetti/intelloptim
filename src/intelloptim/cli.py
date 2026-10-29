"""The top-level ``intelloptim`` command.

It only reports the installed version. Running objectives from a batch system
is the job of the separate ``intelloptim-worker`` command.
"""

from __future__ import annotations

import click

from intelloptim.version import __version__


@click.group(name="intelloptim")
@click.version_option(version=__version__, prog_name="intelloptim")
def main() -> None:
    """intelloptim — black-box optimization SDK and algorithm layer."""


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["main"]
