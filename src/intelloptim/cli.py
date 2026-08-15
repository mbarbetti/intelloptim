"""Top-level ``intelloptim`` CLI.

The public package ships a minimal CLI. The server-admin subcommands
(``server`` / ``user`` / ``token``) and the optimizer worker live in the
private ``intelloptim-server`` package, which exposes its own
``intelloptim-server`` console script. The evaluator worker is the separate
``intelloptim-worker`` console script.
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
