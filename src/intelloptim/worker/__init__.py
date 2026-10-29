"""The ``intelloptim-worker`` command-line tool.

``intelloptim-worker evaluator`` runs an objective against a server without a
driver script: it imports the objective, creates or joins the study and calls
``study.optimize()``. It is meant for batch systems, for example a SLURM array
job in which many processes contribute trials to the same study. Each process
identifies itself to the server with a worker id and a site id derived from the
SLURM environment (see :mod:`intelloptim.worker.slurm`).
"""

from intelloptim.worker.cli import main

__all__ = ["main"]
