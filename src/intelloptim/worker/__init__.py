"""Evaluator worker entrypoint.

Houses the convenience evaluator CLI shim: it imports a
``@intopt.objective``-registered callable and drives ``study.optimize()``
over HTTP, so HPC (e.g. SLURM array) submissions don't have to ship custom
Python. The heavier optimizer-role Celery worker for ``runs_remote``
samplers lives in the private ``intelloptim-server`` package.
"""

from intelloptim.worker.cli import main

__all__ = ["main"]
